import argparse
import csv
import inspect
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from utils.config import load_config


KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0
MODULE_TYPES = ("attn", "mlp")


class ScalarResidualGate(nn.Module):
    def __init__(self, init_bias):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def gate_values_scalar(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, module_output):
        gate_value = torch.sigmoid(self.gate_logit).to(
            device=module_output.device,
            dtype=module_output.dtype,
        )
        gate_values = torch.ones(
            module_output.shape[0],
            module_output.shape[1],
            1,
            device=module_output.device,
            dtype=module_output.dtype,
        ) * gate_value
        return module_output * gate_value, gate_values


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model):
    return next(model.parameters()).device


def module_name(layer_idx, module_type):
    return f"L{layer_idx:02d}.{module_type}"


def get_module_names(model):
    names = []
    for layer_idx in range(len(model.model.layers)):
        names.append(module_name(layer_idx, "attn"))
        names.append(module_name(layer_idx, "mlp"))
    return names


def iter_gate_modules(model):
    for layer in model.model.layers:
        yield layer.attn_gate
        yield layer.mlp_gate


def get_all_module_gates(model):
    return torch.stack([gate.gate_values_scalar() for gate in iter_gate_modules(model)])


def get_gate_values_cpu(model):
    return [float(g.detach().float().cpu()) for g in get_all_module_gates(model)]


def make_qwen_gated_forward(layer, layer_idx, parent_model):
    supported_attn_args = set(inspect.signature(layer.self_attn.forward).parameters)

    def qwen_gated_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states_norm = layer.input_layernorm(hidden_states)
        attn_kwargs = {
            "hidden_states": hidden_states_norm,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        attn_kwargs.update(kwargs)
        attn_kwargs = {key: value for key, value in attn_kwargs.items() if key in supported_attn_args}
        attn_outputs = layer.self_attn(**attn_kwargs)
        attn_output = attn_outputs[0]
        if getattr(parent_model, "_cg_intervention", None) == (layer_idx, "attn"):
            attn_output = torch.zeros_like(attn_output)
        gated_attn, attn_gate_values = layer.attn_gate(attn_output)
        hidden_states = residual + gated_attn

        residual = hidden_states
        hidden_states_norm = layer.post_attention_layernorm(hidden_states)
        mlp_output = layer.mlp(hidden_states_norm)
        if getattr(parent_model, "_cg_intervention", None) == (layer_idx, "mlp"):
            mlp_output = torch.zeros_like(mlp_output)
        gated_mlp, mlp_gate_values = layer.mlp_gate(mlp_output)
        hidden_states = residual + gated_mlp

        layer.last_attn_gate = attn_gate_values
        layer.last_mlp_gate = mlp_gate_values

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[-1],)
        return outputs

    return qwen_gated_forward


def add_causalgates_to_qwen(model, config):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a Qwen/Llama-style model with model.layers")

    init_bias = float(config["gate"]["init_bias"])
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise ValueError(f"Layer {layer_idx} does not expose self_attn/mlp")
        device = next(layer.parameters()).device
        layer.attn_gate = ScalarResidualGate(init_bias).to(device=device)
        layer.mlp_gate = ScalarResidualGate(init_bias).to(device=device)
        layer.forward = make_qwen_gated_forward(layer, layer_idx, model)

    model._cg_intervention = None
    return model


def freeze_backbone_train_gates(model):
    for param in model.parameters():
        param.requires_grad = False
    for gate in iter_gate_modules(model):
        for param in gate.parameters():
            param.requires_grad = True
    return model


def load_qwen_with_gates(config):
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = config.get("model", {}).get("torch_dtype", "float16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get("system", {}).get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = add_causalgates_to_qwen(model, config)
    model = freeze_backbone_train_gates(model)
    return model, tokenizer


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)


def build_wikitext_loader(config, tokenizer, split, num_samples=None, shuffle=False):
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, config["data"]["max_length"]),
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=1, shuffle=shuffle, collate_fn=collator)


def build_c4_loader(config, tokenizer, num_samples):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for example in dataset:
        text = example.get("text", "")
        if len(text.strip()) > 50:
            texts.append({"text": text})
        if len(texts) >= num_samples:
            break
    tokenized = [
        tokenize_fn(example, tokenizer, config["data"]["max_length"])
        for example in texts
    ]
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return [collator([item]) for item in tokenized]


def compute_kl_delta(original_logits, intervened_logits):
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()
    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean").detach()


@torch.no_grad()
def compute_all_module_deltas(model, batch, original_logits):
    deltas = []
    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        for module_type in MODULE_TYPES:
            model._cg_intervention = (layer_idx, module_type)
            intervened_outputs = model(**batch, use_cache=False)
            intervened_logits = intervened_outputs.logits.detach()
            model._cg_intervention = None
            deltas.append(compute_kl_delta(original_logits, intervened_logits))
    return torch.stack(deltas)


def causal_ranking_loss(gates, targets, margin=0.05, num_pairs=128):
    gates = gates.float()
    targets = targets.detach().float()
    num_modules = gates.shape[0]
    idx_a = torch.randint(0, num_modules, (num_pairs,), device=gates.device)
    idx_b = torch.randint(0, num_modules, (num_pairs,), device=gates.device)
    target_diff = targets[idx_a] - targets[idx_b]
    gate_diff = gates[idx_a] - gates[idx_b]
    non_tied = target_diff.abs() > 1e-6
    if non_tied.sum().item() == 0:
        return gates.new_tensor(0.0)
    signs = target_diff[non_tied].sign()
    ordered_gate_diff = signs * gate_diff[non_tied]
    return F.relu(margin - ordered_gate_diff).mean()


def gate_sparsity_loss(model):
    return get_all_module_gates(model).float().pow(2).mean()


def apply_binary_gate_mask(model, kept_indices):
    kept_indices = set(kept_indices)
    for idx, gate in enumerate(iter_gate_modules(model)):
        gate.gate_logit.data.fill_(KEEP_LOGIT if idx in kept_indices else SKIP_LOGIT)


def apply_all_modules_open(model):
    apply_binary_gate_mask(model, set(range(sum(1 for _ in iter_gate_modules(model)))))


@torch.no_grad()
def evaluate_ppl(model, loader):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(model_device(model)) for key, value in batch.items()}
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(mean_nll)


def train_gates(model, train_loader, config):
    gate_params = [param for gate in iter_gate_modules(model) for param in gate.parameters()]
    optimizer = torch.optim.AdamW(
        gate_params,
        lr=float(config["training"]["learning_rate"]),
        eps=float(config["training"].get("adam_eps", 1e-6)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    lambda_lm = float(config["loss"].get("lambda_lm", 0.0))
    lambda_sparsity = float(config["loss"]["lambda_sparsity"])
    lambda_causal = float(config["causal"]["lambda_causal"])
    lambda_rank = float(config["causal"].get("lambda_rank", 0.0))
    target_floor = float(config["causal"].get("target_floor", 0.0))
    ema_beta = float(config["causal"].get("ema_beta", 0.9))
    rank_margin = float(config["causal"].get("rank_margin", 0.05))
    rank_pairs = int(config["causal"].get("rank_pairs", 128))
    grad_accum_steps = int(config["training"]["grad_accum_steps"])
    max_steps = int(config["training"].get("max_steps", 1000))
    max_grad_norm = float(config["training"]["max_grad_norm"])
    log_every = int(config.get("logging", {}).get("log_every", 50))

    optimizer.zero_grad()
    ema_targets = None
    avg_gate_grad_norm = 0.0
    model.train()
    for step, batch in enumerate(tqdm(train_loader, desc="Training Qwen3B CausalGate")):
        batch = {key: value.to(model_device(model)) for key, value in batch.items()}
        model._cg_intervention = None
        outputs = model(**batch, use_cache=False)
        lm_loss = outputs.loss
        original_logits = outputs.logits.detach()

        deltas = compute_all_module_deltas(model, batch, original_logits)
        normalized_targets = deltas / (deltas.max() + 1e-8)
        current_targets = target_floor + (1.0 - target_floor) * normalized_targets
        current_targets = current_targets.to(model_device(model))

        if ema_targets is None:
            ema_targets = current_targets.detach().clone()
        else:
            ema_targets.mul_(ema_beta).add_(current_targets.detach(), alpha=1.0 - ema_beta)
        targets = ema_targets.detach()

        gates = get_all_module_gates(model)
        causal_loss = F.mse_loss(gates.float(), targets.float())
        rank_loss = causal_ranking_loss(gates, targets, margin=rank_margin, num_pairs=rank_pairs)
        sparse_loss = gate_sparsity_loss(model)
        loss = (
            lambda_lm * lm_loss
            + lambda_sparsity * sparse_loss
            + lambda_causal * causal_loss
            + lambda_rank * rank_loss
        )
        (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            total_grad_norm = 0.0
            num_grads = 0
            for gate in iter_gate_modules(model):
                if gate.gate_logit.grad is not None:
                    total_grad_norm += float(gate.gate_logit.grad.abs().item())
                    num_grads += 1
            avg_gate_grad_norm = total_grad_norm / max(num_grads, 1)
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            gate_range = float(gates.max() - gates.min())
            print(
                f"step={step} lm_loss={lm_loss.item():.4f} "
                f"sparsity={float(sparse_loss):.4f} causal_loss={float(causal_loss):.4f} "
                f"rank_loss={float(rank_loss):.4f} delta_max={float(deltas.max()):.4f} "
                f"delta_mean={float(deltas.mean()):.4f} target_mean={float(current_targets.mean()):.4f} "
                f"ema_target_mean={float(targets.mean()):.4f} gate_mean={float(gates.mean()):.4f} "
                f"gate_min={float(gates.min()):.4f} gate_max={float(gates.max()):.4f} "
                f"gate_range={gate_range:.4f} grad_norm={avg_gate_grad_norm:.8f}"
            )

        if step >= max_steps:
            break
    model._cg_intervention = None


def save_gate_tables(model, gate_values_path, ranking_path):
    module_names = get_module_names(model)
    gate_values = get_gate_values_cpu(model)
    rows = [{"module": name, "gate": gate} for name, gate in zip(module_names, gate_values)]
    ranked = sorted(rows, key=lambda row: row["gate"], reverse=True)

    Path(gate_values_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(gate_values_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "gate"])
        writer.writeheader()
        writer.writerows(rows)

    Path(ranking_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(ranking_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "module", "gate"])
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, **row})


def save_gate_state(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for layer_idx, layer in enumerate(model.model.layers):
        state[f"model.layers.{layer_idx}.attn_gate.gate_logit"] = layer.attn_gate.gate_logit.detach().cpu()
        state[f"model.layers.{layer_idx}.mlp_gate.gate_logit"] = layer.mlp_gate.gate_logit.detach().cpu()
    torch.save(state, output_path)


def evaluate_budgets(model, wikitext_loader, c4_loader, results_csv):
    module_names = get_module_names(model)
    gate_values = get_gate_values_cpu(model)
    num_modules = len(gate_values)
    ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)
    rows = []

    for target_saved in [0.0, 0.05, 0.10]:
        if target_saved == 0.0:
            skipped = 0
            kept = num_modules
            apply_all_modules_open(model)
        else:
            skipped = max(0, min(num_modules - 1, round(num_modules * target_saved)))
            kept = num_modules - skipped
            apply_binary_gate_mask(model, set(ranked_indices[:kept]))

        wikitext_ppl = evaluate_ppl(model, wikitext_loader)
        c4_ppl = evaluate_ppl(model, c4_loader)
        realized_saved = skipped / num_modules
        rows.append({
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "target_saved": target_saved,
            "realized_saved": realized_saved,
            "wikitext_ppl": wikitext_ppl,
            "c4_ppl": c4_ppl,
            "num_modules": num_modules,
            "num_skipped": skipped,
        })
        skipped_names = [module_names[idx] for idx in ranked_indices[kept:]]
        print(
            f"target_saved={target_saved:.2f} realized_saved={realized_saved:.4f} "
            f"num_modules={num_modules} skipped={skipped} skipped_modules={skipped_names}"
        )

    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(results_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CausalGate on Qwen2.5-3B-Instruct.")
    parser.add_argument("--config", default="utils/qwen3b_gate.yaml")
    parser.add_argument("--results-csv", default="results/qwen3b_causalgate.csv")
    parser.add_argument("--gate-values-csv", default="outputs/qwen3b_gate_values.csv")
    parser.add_argument("--module-ranking-csv", default="outputs/qwen3b_module_ranking.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)

    model, tokenizer = load_qwen_with_gates(config)
    num_layers = len(model.model.layers)
    num_modules = sum(1 for _ in iter_gate_modules(model))
    print(f"Loaded {config['model']['name']}")
    print(f"Detected layers={num_layers}, attention_modules={num_layers}, mlp_modules={num_layers}, total_modules={num_modules}")
    if num_modules != 2 * num_layers:
        raise RuntimeError(f"Gate attachment verification failed: expected {2 * num_layers}, found {num_modules}")

    train_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=config["data"]["split"],
        num_samples=None,
        shuffle=True,
    )
    wikitext_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=config["data"].get("eval_split", "test"),
        num_samples=int(config["data"].get("wikitext_eval_samples", 128)),
        shuffle=False,
    )
    c4_loader = build_c4_loader(
        config,
        tokenizer,
        num_samples=int(config["data"].get("c4_eval_samples", 128)),
    )

    train_gates(model, train_loader, config)
    save_dir = Path(config.get("logging", {}).get("save_dir", "outputs/qwen3b_causalgate"))
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    save_gate_state(model, save_dir / "qwen3b_gate_state.pt")

    save_gate_tables(model, args.gate_values_csv, args.module_ranking_csv)
    evaluate_budgets(model, wikitext_loader, c4_loader, args.results_csv)
    print(f"Saved results to {args.results_csv}")
    print(f"Saved gate values to {args.gate_values_csv}")
    print(f"Saved module ranking to {args.module_ranking_csv}")


if __name__ == "__main__":
    main()
