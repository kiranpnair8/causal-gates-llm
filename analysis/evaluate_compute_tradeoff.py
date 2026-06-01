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
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from models.load_model import load_tinyllama_with_gates
from utils.config import load_config


MODULES = ("attn", "mlp")
KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def module_name(layer_idx, module):
    return f"L{layer_idx:02d}.{module}"


def get_module_names(model):
    names = []

    for layer_idx in range(len(model.model.layers)):
        for module in MODULES:
            names.append(module_name(layer_idx, module))

    return names


def iter_gate_modules(model):
    for layer in model.model.layers:
        yield layer.attn_gate
        yield layer.mlp_gate


def get_gate_values(model):
    return [float(gate.gate_values_scalar().detach().float().cpu()) for gate in iter_gate_modules(model)]


def get_gate_logits(model):
    return [gate.gate_logit.detach().clone() for gate in iter_gate_modules(model)]


def restore_gate_logits(model, logits):
    for gate, logit in zip(iter_gate_modules(model), logits):
        gate.gate_logit.data.copy_(logit.to(gate.gate_logit.device))


def apply_binary_gate_mask(model, kept_indices):
    kept_indices = set(kept_indices)

    for idx, gate in enumerate(iter_gate_modules(model)):
        value = KEEP_LOGIT if idx in kept_indices else SKIP_LOGIT
        gate.gate_logit.data.fill_(value)


def load_gate_checkpoint(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
        )

    gate_state = {
        key: value
        for key, value in state_dict.items()
        if key.endswith("attn_gate.gate_logit") or key.endswith("mlp_gate.gate_logit")
    }

    if not gate_state:
        raise ValueError(f"No gate_logit parameters found in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(gate_state, strict=False)
    print(f"Loaded {len(gate_state)} trained gate parameters from {checkpoint_dir}")
    print(f"Ignored missing backbone keys: {len(missing)}; unexpected keys: {len(unexpected)}")


def build_wikitext_loader(config, tokenizer, num_samples):
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config"],
        split=config["data"].get("eval_split", "test"),
    )
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, config["data"]["max_length"]),
        remove_columns=dataset.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return DataLoader(
        tokenized,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )


@torch.no_grad()
def evaluate_wikitext_ppl(model, loader):
    total_nll = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch)
        labels = batch.get("labels")

        token_count = int((labels != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    mean_nll = total_nll / total_tokens
    return mean_nll, math.exp(mean_nll)


def get_hellaswag_context(example):
    ctx = example.get("ctx", "")
    if ctx:
        return ctx

    ctx_a = example.get("ctx_a", "")
    ctx_b = example.get("ctx_b", "")
    return f"{ctx_a} {ctx_b}".strip()


@torch.no_grad()
def score_ending(model, tokenizer, context, ending, max_length):
    context = context.strip()
    ending = ending.strip()

    if not ending.startswith(" "):
        ending = " " + ending

    context_ids = tokenizer(
        context,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        context + ending,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    if len(full_ids) <= 1:
        return float("inf")

    context_len = min(len(context_ids), len(full_ids) - 1)
    labels = torch.tensor(full_ids, dtype=torch.long)
    labels[:context_len] = -100

    if int((labels != -100).sum().item()) == 0:
        labels[-1] = full_ids[-1]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    labels = labels.unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return float(outputs.loss.item())


@torch.no_grad()
def evaluate_hellaswag_accuracy(model, tokenizer, num_samples, max_length):
    dataset = load_dataset("hellaswag", split="validation")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0

    for example in dataset:
        context = get_hellaswag_context(example)
        endings = example["endings"]
        label = int(example["label"])

        scores = [
            score_ending(model, tokenizer, context, ending, max_length)
            for ending in endings
        ]
        prediction = min(range(len(scores)), key=lambda idx: scores[idx])

        correct += int(prediction == label)
        total += 1

    return correct / total, correct, total


def make_row(policy, keep_ratio, kept_count, skipped_count, wikitext_nll, wikitext_ppl, hellaswag_acc, hellaswag_correct, hellaswag_total):
    return {
        "policy": policy,
        "keep_ratio": keep_ratio,
        "compute_saved": 1.0 - keep_ratio,
        "kept_modules": kept_count,
        "skipped_modules": skipped_count,
        "wikitext_mean_nll": wikitext_nll,
        "wikitext_ppl": wikitext_ppl,
        "hellaswag_acc": hellaswag_acc,
        "hellaswag_correct": hellaswag_correct,
        "hellaswag_total": hellaswag_total,
    }


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "keep_ratio",
                "compute_saved",
                "kept_modules",
                "skipped_modules",
                "wikitext_mean_nll",
                "wikitext_ppl",
                "hellaswag_acc",
                "hellaswag_correct",
                "hellaswag_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nCompute Tradeoff Evaluation")
    print("| policy | keep | saved | kept | skipped | WikiText PPL | HellaSwag acc |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['policy']} | {row['keep_ratio']:.2f} | {row['compute_saved']:.2f} | "
            f"{row['kept_modules']} | {row['skipped_modules']} | "
            f"{row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} |"
        )


def evaluate_policy(model, tokenizer, wikitext_loader, args, policy, keep_ratio, kept_indices, num_modules):
    apply_binary_gate_mask(model, kept_indices)
    wikitext_nll, wikitext_ppl = evaluate_wikitext_ppl(model, wikitext_loader)
    hs_acc, hs_correct, hs_total = evaluate_hellaswag_accuracy(
        model,
        tokenizer,
        args.hellaswag_samples,
        args.max_length,
    )

    return make_row(
        policy,
        keep_ratio,
        len(kept_indices),
        num_modules - len(kept_indices),
        wikitext_nll,
        wikitext_ppl,
        hs_acc,
        hs_correct,
        hs_total,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate compute-saved vs quality tradeoff on WikiText PPL and HellaSwag accuracy."
    )
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.95, 0.90, 0.85, 0.80, 0.75],
    )
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/compute_tradeoff_wikitext_hellaswag.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    wikitext_loader = build_wikitext_loader(config, tokenizer, args.wikitext_samples)
    module_names = get_module_names(model)
    num_modules = len(module_names)
    trained_logits = get_gate_logits(model)
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(
        range(num_modules),
        key=lambda idx: gate_values[idx],
        reverse=True,
    )

    rows = []

    restore_gate_logits(model, trained_logits)
    wikitext_nll, wikitext_ppl = evaluate_wikitext_ppl(model, wikitext_loader)
    hs_acc, hs_correct, hs_total = evaluate_hellaswag_accuracy(
        model,
        tokenizer,
        args.hellaswag_samples,
        args.max_length,
    )
    rows.append(
        make_row(
            "learned_soft_gates",
            1.0,
            num_modules,
            0,
            wikitext_nll,
            wikitext_ppl,
            hs_acc,
            hs_correct,
            hs_total,
        )
    )

    for keep_ratio in args.keep_ratios:
        keep_count = max(1, min(num_modules, round(num_modules * keep_ratio)))
        kept_indices = set(gate_ranked_indices[:keep_count])
        rows.append(
            evaluate_policy(
                model,
                tokenizer,
                wikitext_loader,
                args,
                "gate_topk_binary",
                keep_count / num_modules,
                kept_indices,
                num_modules,
            )
        )

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved compute tradeoff table to {args.output_csv}")


if __name__ == "__main__":
    main()
