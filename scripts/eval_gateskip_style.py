import argparse
import csv
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model):
    return next(model.parameters()).device


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)


def load_tinyllama(config):
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = config.get("model", {}).get("torch_dtype", "float16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get("system", {}).get("device_map", "auto"),
    )
    model.config.use_cache = False
    return model, tokenizer


class GateSkipBranchWrapper(nn.Module):
    def __init__(self, original_module, parent_model, module_name, hidden_size, init_bias, init_std):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "parent_model", parent_model)
        self.module_name = module_name
        self.gate = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.gate.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.gate.bias, init_bias)
        self.gate.to(device=next(original_module.parameters()).device, dtype=torch.float32)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_module(hidden_states, *args, **kwargs)
        is_tuple = isinstance(outputs, tuple)
        branch_output = outputs[0] if is_tuple else outputs

        gate_value = torch.sigmoid(self.gate(hidden_states.detach().float()))
        self._record_gate_values(gate_value)

        gated_output = branch_output * gate_value.to(dtype=branch_output.dtype)
        skip_mask = self._make_skip_mask(gate_value.squeeze(-1), hidden_states)
        if skip_mask is not None:
            gated_output = gated_output.masked_fill(skip_mask.unsqueeze(-1), 0.0)

        if is_tuple:
            return (gated_output,) + outputs[1:]
        return gated_output

    def _record_gate_values(self, gate_value):
        if not torch.is_grad_enabled():
            return
        valid_mask = self._valid_mask(gate_value.shape[:2], gate_value.device)
        valid_gates = gate_value.squeeze(-1)[valid_mask]
        if valid_gates.numel() > 0:
            self.parent_model._gateskip_gate_values.append(valid_gates)

    def _valid_mask(self, shape, device):
        current_mask = getattr(self.parent_model, "_gateskip_current_attention_mask", None)
        if current_mask is not None and tuple(current_mask.shape) == tuple(shape):
            return current_mask.to(device=device, dtype=torch.bool)
        return torch.ones(shape, device=device, dtype=torch.bool)

    def _make_skip_mask(self, token_importance, hidden_states):
        skip_ratio = getattr(self.parent_model, "_gateskip_skip_ratio", None)
        if skip_ratio is None or skip_ratio <= 0.0:
            return None

        valid_mask = self._valid_mask(token_importance.shape, token_importance.device)
        valid_scores = token_importance[valid_mask]
        valid_count = int(valid_scores.numel())
        skip_count = int(round(valid_count * skip_ratio))
        if skip_count <= 0:
            self._update_stats(valid_count, 0)
            return torch.zeros_like(valid_mask)
        skip_count = min(skip_count, valid_count)

        threshold = torch.topk(valid_scores, k=skip_count, largest=False).values.max()
        skip_mask = (token_importance <= threshold) & valid_mask

        actual_skip = int(skip_mask.sum().item())
        if actual_skip > skip_count:
            valid_positions = valid_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
            valid_flat_scores = token_importance.flatten()[valid_positions]
            selected = torch.topk(valid_flat_scores, k=skip_count, largest=False).indices
            exact_flat = torch.zeros_like(token_importance.flatten(), dtype=torch.bool)
            exact_flat[valid_positions[selected]] = True
            skip_mask = exact_flat.view_as(token_importance)
            actual_skip = skip_count

        self._update_stats(valid_count, actual_skip)
        return skip_mask

    def _update_stats(self, total, skipped):
        self.parent_model._gateskip_total_valid_tokens += float(total)
        self.parent_model._gateskip_skipped_valid_tokens += float(skipped)


def iter_gateskip_wrappers(model):
    for layer in model.model.layers:
        if isinstance(layer.self_attn, GateSkipBranchWrapper):
            yield layer.self_attn
        if isinstance(layer.mlp, GateSkipBranchWrapper):
            yield layer.mlp


def add_gateskip_wrappers(model, args):
    hidden_size = model.config.hidden_size
    for layer in model.model.layers:
        layer.self_attn = GateSkipBranchWrapper(
            layer.self_attn,
            model,
            "attn",
            hidden_size,
            init_bias=args.gate_init_bias,
            init_std=args.gate_init_std,
        )
        layer.mlp = GateSkipBranchWrapper(
            layer.mlp,
            model,
            "mlp",
            hidden_size,
            init_bias=args.gate_init_bias,
            init_std=args.gate_init_std,
        )
    reset_gateskip_state(model)
    return model


def freeze_backbone_train_gates(model):
    for param in model.parameters():
        param.requires_grad = False
    for wrapper in iter_gateskip_wrappers(model):
        for param in wrapper.gate.parameters():
            param.requires_grad = True


def gateskip_parameters(model):
    for wrapper in iter_gateskip_wrappers(model):
        yield from wrapper.gate.parameters()


def reset_gateskip_stats(model):
    model._gateskip_total_valid_tokens = 0.0
    model._gateskip_skipped_valid_tokens = 0.0


def clear_gateskip_gate_values(model):
    model._gateskip_gate_values = []


def reset_gateskip_state(model):
    model._gateskip_skip_ratio = None
    model._gateskip_current_attention_mask = None
    clear_gateskip_gate_values(model)
    reset_gateskip_stats(model)


def set_gateskip_batch_mask(model, attention_mask):
    model._gateskip_current_attention_mask = attention_mask.detach() if attention_mask is not None else None


def get_gateskip_saved(model):
    total = getattr(model, "_gateskip_total_valid_tokens", 0.0)
    skipped = getattr(model, "_gateskip_skipped_valid_tokens", 0.0)
    return skipped / max(total, 1.0)


def gate_regularizers(model, eps=1e-6):
    if not model._gateskip_gate_values:
        device = model_device(model)
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    gates = torch.cat(model._gateskip_gate_values)
    sparsity = gates.abs().mean()
    entropy = (-(gates * torch.log(gates + eps)) - ((1.0 - gates) * torch.log(1.0 - gates + eps))).mean()
    return sparsity, entropy


def build_wikitext_loader(config, tokenizer, split, num_samples, batch_size, shuffle):
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, config["data"]["max_length"]),
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def train_gates(model, train_loader, args):
    freeze_backbone_train_gates(model)
    model.train()
    model._gateskip_skip_ratio = None
    optimizer = torch.optim.AdamW(gateskip_parameters(model), lr=args.learning_rate, weight_decay=0.0)

    step = 0
    pbar = tqdm(total=args.train_steps)
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            batch = {k: v.to(model_device(model)) for k, v in batch.items()}
            set_gateskip_batch_mask(model, batch.get("attention_mask"))
            clear_gateskip_gate_values(model)

            outputs = model(**batch, use_cache=False)
            sparsity, entropy = gate_regularizers(model)
            loss = outputs.loss + args.sparsity_loss_weight * sparsity + args.entropy_loss_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gateskip_parameters(model)), args.max_grad_norm)
            optimizer.step()

            if step % args.log_every == 0:
                print(
                    f"step={step} lm_loss={outputs.loss.item():.4f} "
                    f"sparsity={sparsity.item():.4f} entropy={entropy.item():.4f} total_loss={loss.item():.4f}"
                )
            step += 1
            pbar.update(1)
    pbar.close()


def save_gates(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {f"wrapper_{idx}.gate.{name}": value.detach().cpu() for idx, wrapper in enumerate(iter_gateskip_wrappers(model)) for name, value in wrapper.gate.state_dict().items()}
    torch.save(state, path)
    print(f"Saved GateSkip gate checkpoint to {path}")


def load_gates(model, path):
    state = torch.load(path, map_location="cpu")
    wrappers = list(iter_gateskip_wrappers(model))
    for idx, wrapper in enumerate(wrappers):
        prefix = f"wrapper_{idx}.gate."
        gate_state = {k.replace(prefix, ""): v for k, v in state.items() if k.startswith(prefix)}
        wrapper.gate.load_state_dict(gate_state)
    print(f"Loaded GateSkip gate checkpoint from {path}")


@torch.no_grad()
def evaluate_wikitext_ppl(model, loader, skip_ratio):
    model.eval()
    model._gateskip_skip_ratio = skip_ratio
    reset_gateskip_stats(model)
    total_nll = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {k: v.to(model_device(model)) for k, v in batch.items()}
        set_gateskip_batch_mask(model, batch.get("attention_mask"))
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(mean_nll), get_gateskip_saved(model)


def get_hellaswag_context(example):
    ctx = example.get("ctx", "")
    return ctx if ctx else f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()


def make_choice_tensors(tokenizer, context, ending, max_length, device):
    context = context.strip()
    ending = ending.strip()
    if not ending.startswith(" "):
        ending = " " + ending
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(context + ending, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    context_len = min(len(context_ids), max(0, len(full_ids) - 1))
    labels = torch.tensor(full_ids, dtype=torch.long)
    labels[:context_len] = -100
    if int((labels != -100).sum().item()) == 0 and len(labels) > 0:
        labels[-1] = full_ids[-1]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels.unsqueeze(0).to(device),
    }


@torch.no_grad()
def score_choice(model, tensors):
    set_gateskip_batch_mask(model, tensors.get("attention_mask"))
    outputs = model(**tensors, use_cache=False)
    token_count = int((tensors["labels"][:, 1:] != -100).sum().item())
    return float(outputs.loss.item()) * token_count / max(token_count, 1)


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, dataset_name, num_samples, max_length, skip_ratio):
    model.eval()
    model._gateskip_skip_ratio = skip_ratio
    reset_gateskip_stats(model)

    if dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    elif dataset_name == "commonsense_qa":
        dataset = load_dataset("commonsense_qa", split="validation")
    else:
        raise ValueError(dataset_name)

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    correct = 0
    total = 0

    for example in dataset:
        if dataset_name == "hellaswag":
            context = get_hellaswag_context(example)
            endings = example["endings"]
            label = int(example["label"])
        elif dataset_name == "piqa":
            context = example["goal"]
            endings = [example["sol1"], example["sol2"]]
            label = int(example["label"])
        else:
            context = example["question"]
            endings = example["choices"]["text"]
            label = example["choices"]["label"].index(example["answerKey"])

        scores = []
        for ending in endings:
            tensors = make_choice_tensors(tokenizer, context, ending, max_length, model_device(model))
            scores.append(score_choice(model, tensors))
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
        total += 1

    return correct / max(total, 1), correct, total, get_gateskip_saved(model)


def make_row(policy, target_saved, realized_saved, wikitext_nll, wikitext_ppl, hs, piqa, csqa):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "wikitext_mean_nll": wikitext_nll,
        "wikitext_ppl": wikitext_ppl,
        "hellaswag_acc": hs[0],
        "hellaswag_correct": hs[1],
        "hellaswag_total": hs[2],
        "piqa_acc": piqa[0],
        "piqa_correct": piqa[1],
        "piqa_total": piqa[2],
        "commonsenseqa_acc": csqa[0],
        "commonsenseqa_correct": csqa[1],
        "commonsenseqa_total": csqa[2],
    }


def evaluate_suite(model, tokenizer, wikitext_loader, args, target_saved):
    w_nll, w_ppl, w_saved = evaluate_wikitext_ppl(model, wikitext_loader, target_saved)
    hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", args.hellaswag_samples, args.max_length, target_saved)
    piqa = evaluate_multiple_choice(model, tokenizer, "piqa", args.piqa_samples, args.max_length, target_saved)
    csqa = evaluate_multiple_choice(model, tokenizer, "commonsense_qa", args.commonsenseqa_samples, args.max_length, target_saved)
    realized = (w_saved + hs[3] + piqa[3] + csqa[3]) / 4.0
    return realized, w_nll, w_ppl, hs, piqa, csqa


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nGateSkip-Style Tradeoff")
    print("| policy | target_saved | realized_saved | WikiText PPL | HellaSwag | PIQA | CSQA |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} | "
            f"{row['piqa_acc']:.4f} | {row['commonsenseqa_acc']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a standalone GateSkip-style residual gating baseline.")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sparsity-loss-weight", type=float, default=0.01)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.0)
    parser.add_argument("--gate-init-bias", type=float, default=5.0)
    parser.add_argument("--gate-init-std", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/gateskip_style_tradeoff.csv")
    parser.add_argument("--save-gates", default="outputs/gateskip_style_gates.pt")
    parser.add_argument("--load-gates", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama(config)
    add_gateskip_wrappers(model, args)

    train_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=config["data"].get("split", "train"),
        num_samples=args.train_samples,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=config["data"].get("eval_split", "test"),
        num_samples=args.wikitext_samples,
        batch_size=1,
        shuffle=False,
    )

    if args.load_gates:
        load_gates(model, args.load_gates)

    if not args.skip_training:
        train_gates(model, train_loader, args)
        save_gates(model, args.save_gates)

    rows = []
    realized, w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, eval_loader, args, target_saved=0.0)
    rows.append(make_row("full_model_with_trained_gates", 0.0, realized, w_nll, w_ppl, hs, piqa, csqa))

    for target_saved in args.target_saved:
        realized, w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, eval_loader, args, target_saved=target_saved)
        rows.append(make_row("gateskip_style", target_saved, realized, w_nll, w_ppl, hs, piqa, csqa))

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved GateSkip-style tradeoff table to {args.output_csv}")


if __name__ == "__main__":
    main()
