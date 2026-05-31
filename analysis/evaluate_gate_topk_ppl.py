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
    values = []

    for gate in iter_gate_modules(model):
        values.append(float(gate.gate_values_scalar().detach().float().cpu()))

    return values


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


def build_loader(config, tokenizer, num_samples):
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config"],
        split=config["data"]["split"],
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
def evaluate_ppl(model, loader):
    total_nll = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch)
        labels = batch.get("labels")

        if labels is None:
            raise ValueError("Batch is missing labels; use a causal LM data collator")

        token_count = int((labels != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No valid LM tokens found for evaluation")

    mean_nll = total_nll / total_tokens
    return mean_nll, math.exp(mean_nll)


def make_row(policy, keep_ratio, kept_indices, mean_nll, ppl, module_names):
    kept = [module_names[idx] for idx in sorted(kept_indices)]
    skipped = [name for idx, name in enumerate(module_names) if idx not in set(kept_indices)]

    return {
        "policy": policy,
        "keep_ratio": keep_ratio,
        "kept_modules": len(kept),
        "skipped_modules": len(skipped),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "kept_module_names": ";".join(kept),
        "skipped_module_names": ";".join(skipped),
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
                "kept_modules",
                "skipped_modules",
                "mean_nll",
                "ppl",
                "kept_module_names",
                "skipped_module_names",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nTop-k Gate PPL Evaluation")
    print("| policy | keep_ratio | kept | skipped | mean_nll | ppl |")
    print("|---|---:|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['policy']} | {row['keep_ratio']:.2f} | "
            f"{row['kept_modules']} | {row['skipped_modules']} | "
            f"{row['mean_nll']:.4f} | {row['ppl']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether learned high-gate modules preserve perplexity under top-k module skipping."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/tinyllama_gated",
        help="Directory containing trained gated model checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25],
    )
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-csv",
        default="outputs/gate_topk_ppl.csv",
        help="Where to save PPL results",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    loader = build_loader(config, tokenizer, args.num_samples)
    module_names = get_module_names(model)
    num_modules = len(module_names)
    trained_logits = get_gate_logits(model)
    learned_gate_values = get_gate_values(model)
    ranked_indices = sorted(
        range(num_modules),
        key=lambda idx: learned_gate_values[idx],
        reverse=True,
    )

    rows = []

    restore_gate_logits(model, trained_logits)
    mean_nll, ppl = evaluate_ppl(model, loader)
    rows.append(
        make_row(
            "learned_soft_gates",
            1.0,
            set(range(num_modules)),
            mean_nll,
            ppl,
            module_names,
        )
    )

    apply_binary_gate_mask(model, set(range(num_modules)))
    mean_nll, ppl = evaluate_ppl(model, loader)
    rows.append(
        make_row(
            "all_modules_binary",
            1.0,
            set(range(num_modules)),
            mean_nll,
            ppl,
            module_names,
        )
    )

    for keep_ratio in args.keep_ratios:
        keep_count = max(1, min(num_modules, round(num_modules * keep_ratio)))
        kept_indices = set(ranked_indices[:keep_count])

        apply_binary_gate_mask(model, kept_indices)
        mean_nll, ppl = evaluate_ppl(model, loader)
        rows.append(
            make_row(
                "gate_topk_binary",
                keep_count / num_modules,
                kept_indices,
                mean_nll,
                ppl,
                module_names,
            )
        )

        if keep_count < num_modules:
            random_nlls = []
            random_ppls = []

            for trial in range(args.random_trials):
                kept_indices = set(random.sample(range(num_modules), keep_count))
                apply_binary_gate_mask(model, kept_indices)
                mean_nll, ppl = evaluate_ppl(model, loader)
                random_nlls.append(mean_nll)
                random_ppls.append(ppl)
                rows.append(
                    make_row(
                        f"random_binary_trial_{trial + 1}",
                        keep_count / num_modules,
                        kept_indices,
                        mean_nll,
                        ppl,
                        module_names,
                    )
                )

            avg_nll = sum(random_nlls) / len(random_nlls)
            avg_ppl = sum(random_ppls) / len(random_ppls)
            rows.append(
                make_row(
                    "random_binary_avg",
                    keep_count / num_modules,
                    set(),
                    avg_nll,
                    avg_ppl,
                    module_names,
                )
            )

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved PPL table to {args.output_csv}")


if __name__ == "__main__":
    main()
