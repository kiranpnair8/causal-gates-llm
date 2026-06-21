import argparse
import csv
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.qwen3b_causalgate import (
    add_causalgates_to_qwen,
    apply_all_modules_open,
    apply_binary_gate_mask,
    build_c4_loader,
    build_wikitext_loader,
    causal_ranking_loss,
    compute_all_module_deltas,
    evaluate_ppl,
    freeze_backbone_train_gates,
    get_module_names,
    iter_gate_modules,
    save_gate_state,
)
from utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_llama_with_gates(config):
    model_name = config["model"]["name"]
    token = os.environ.get("HF_TOKEN")
    auth_kwargs = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = config.get("model", {}).get("torch_dtype", "float16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get("system", {}).get("device_map", "auto"),
        low_cpu_mem_usage=True,
        **auth_kwargs,
    )
    model.config.use_cache = False

    # The Qwen helper is architecture-generic: it discovers model.layers and
    # wraps each Llama attention and MLP residual branch independently.
    model = add_causalgates_to_qwen(model, config)
    model = freeze_backbone_train_gates(model)
    return model, tokenizer


def verify_gate_attachment(model):
    num_layers = len(model.model.layers)
    gates = list(iter_gate_modules(model))
    expected_modules = 2 * num_layers
    if len(gates) != expected_modules:
        raise RuntimeError(
            f"Gate attachment failed: expected {expected_modules} gates, found {len(gates)}"
        )
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "attn_gate") or not hasattr(layer, "mlp_gate"):
            raise RuntimeError(f"Layer {layer_idx} is missing an attention or MLP gate")
    print(
        f"Verified layers={num_layers}, attention_modules={num_layers}, "
        f"mlp_modules={num_layers}, total_modules={len(gates)}"
    )
    return num_layers, len(gates)


def get_gate_values_cpu(model):
    return [
        float(gate.gate_values_scalar().detach().float().cpu())
        for gate in iter_gate_modules(model)
    ]


def get_all_module_gates(model, target_device):
    return torch.stack([
        gate.gate_values_scalar().to(target_device)
        for gate in iter_gate_modules(model)
    ])


def train_gates_multidevice(model, train_loader, config):
    gate_params = [
        param
        for gate in iter_gate_modules(model)
        for param in gate.parameters()
    ]
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

    input_device = next(model.parameters()).device
    optimizer.zero_grad()
    ema_targets = None
    avg_gate_grad_norm = 0.0
    model.train()

    for step, batch in enumerate(tqdm(train_loader, desc="Training Llama-3.1-8B CausalGate")):
        batch = {key: value.to(input_device) for key, value in batch.items()}
        model._cg_intervention = None
        outputs = model(**batch, use_cache=False)
        lm_loss = outputs.loss
        loss_device = lm_loss.device
        original_logits = outputs.logits.detach()

        deltas = compute_all_module_deltas(model, batch, original_logits).to(loss_device)
        normalized_targets = deltas / (deltas.max() + 1e-8)
        current_targets = target_floor + (1.0 - target_floor) * normalized_targets

        if ema_targets is None:
            ema_targets = current_targets.detach().clone()
        else:
            ema_targets.mul_(ema_beta).add_(
                current_targets.detach(),
                alpha=1.0 - ema_beta,
            )
        targets = ema_targets.detach()

        gates = get_all_module_gates(model, loss_device)
        causal_loss = F.mse_loss(gates.float(), targets.float())
        rank_loss = causal_ranking_loss(
            gates,
            targets,
            margin=rank_margin,
            num_pairs=rank_pairs,
        )
        sparse_loss = gates.float().pow(2).mean()
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
            print(
                f"step={step} lm_loss={lm_loss.item():.4f} "
                f"sparsity={float(sparse_loss):.4f} "
                f"causal_loss={float(causal_loss):.4f} "
                f"rank_loss={float(rank_loss):.4f} "
                f"delta_max={float(deltas.max()):.4f} "
                f"delta_mean={float(deltas.mean()):.4f} "
                f"target_mean={float(current_targets.mean()):.4f} "
                f"ema_target_mean={float(targets.mean()):.4f} "
                f"gate_mean={float(gates.mean()):.4f} "
                f"gate_min={float(gates.min()):.4f} "
                f"gate_max={float(gates.max()):.4f} "
                f"gate_range={float(gates.max() - gates.min()):.4f} "
                f"grad_norm={avg_gate_grad_norm:.8f}"
            )

        if step >= max_steps:
            break
    model._cg_intervention = None


def save_gate_tables(model, gate_values_path, ranking_path):
    module_names = get_module_names(model)
    gate_values = get_gate_values_cpu(model)
    rows = [
        {"module": name, "gate": gate}
        for name, gate in zip(module_names, gate_values)
    ]
    ranked = sorted(rows, key=lambda row: row["gate"], reverse=True)

    gate_values_path = Path(gate_values_path)
    gate_values_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_values_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "gate"])
        writer.writeheader()
        writer.writerows(rows)

    ranking_path = Path(ranking_path)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "module", "gate"])
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, **row})


def evaluate_budgets(model, config, wikitext_loader, c4_loader, results_csv):
    module_names = get_module_names(model)
    learned_gate_values = get_gate_values_cpu(model)
    num_modules = len(learned_gate_values)
    ranked_indices = sorted(
        range(num_modules),
        key=lambda idx: learned_gate_values[idx],
        reverse=True,
    )
    rows = []

    for target_saved in (0.0, 0.05, 0.10):
        if target_saved == 0.0:
            num_skipped = 0
            keep_count = num_modules
            apply_all_modules_open(model)
        else:
            num_skipped = max(
                0,
                min(num_modules - 1, round(num_modules * target_saved)),
            )
            keep_count = num_modules - num_skipped
            apply_binary_gate_mask(model, set(ranked_indices[:keep_count]))

        realized_saved = num_skipped / num_modules
        wikitext_ppl = evaluate_ppl(model, wikitext_loader)
        c4_ppl = evaluate_ppl(model, c4_loader)
        skipped_names = [module_names[idx] for idx in ranked_indices[keep_count:]]
        rows.append({
            "model": config["model"]["name"],
            "target_saved": target_saved,
            "realized_saved": realized_saved,
            "wikitext_ppl": wikitext_ppl,
            "c4_ppl": c4_ppl,
            "num_modules": num_modules,
            "num_skipped": num_skipped,
        })
        print(
            f"target_saved={target_saved:.2f} realized_saved={realized_saved:.4f} "
            f"num_modules={num_modules} num_skipped={num_skipped} "
            f"skipped_modules={skipped_names}"
        )

    output_path = Path(results_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate CausalGate on Llama-3.1-8B-Instruct."
    )
    parser.add_argument("--config", default="utils/llama31_8b_gate.yaml")
    parser.add_argument(
        "--results-csv",
        default="results/llama31_8b_causalgate.csv",
    )
    parser.add_argument(
        "--gate-values-csv",
        default="outputs/llama31_8b_gate_values.csv",
    )
    parser.add_argument(
        "--module-ranking-csv",
        default="outputs/llama31_8b_module_ranking.csv",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)

    model, tokenizer = load_llama_with_gates(config)
    num_layers, num_modules = verify_gate_attachment(model)
    print(
        f"Loaded {config['model']['name']} in fp16 with use_cache=False; "
        f"detected {num_layers} layers and {num_modules} gated modules."
    )

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

    train_gates_multidevice(model, train_loader, config)

    save_dir = Path(
        config.get("logging", {}).get(
            "save_dir",
            "outputs/llama31_8b_causalgate",
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    save_gate_state(model, save_dir / "llama31_8b_gate_state.pt")
    save_gate_tables(model, args.gate_values_csv, args.module_ranking_csv)

    evaluate_budgets(model, config, wikitext_loader, c4_loader, args.results_csv)
    print(f"Saved results to {args.results_csv}")
    print(f"Saved gate values to {args.gate_values_csv}")
    print(f"Saved module ranking to {args.module_ranking_csv}")


if __name__ == "__main__":
    main()
