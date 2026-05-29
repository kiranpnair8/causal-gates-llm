import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from models.load_model import load_tinyllama_with_gates
from models.intervention import set_intervention, clear_intervention
from utils.config import load_config


MODULES = ("attn", "mlp")


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


def get_all_module_gates(model):
    gates = []

    for layer in model.model.layers:
        gates.append(layer.attn_gate.gate_values_scalar())
        gates.append(layer.mlp_gate.gate_values_scalar())

    return torch.stack(gates).detach().float().cpu()


def compute_kl_delta(original_logits, intervened_logits):
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()

    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)

    return F.kl_div(log_p, q, reduction="batchmean")


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0

    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank

        i = j + 1

    return torch.tensor(ranks, dtype=torch.float32)


def pearson_corr(x, y):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()

    if denom.item() <= 1e-8:
        return 0.0

    return float((x * y).sum() / denom)


def spearman_corr(x, y):
    return pearson_corr(rankdata(x), rankdata(y))


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
    loaded_keys = len(gate_state)
    print(f"Loaded {loaded_keys} trained gate parameters from {checkpoint_dir}")
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
def compute_average_module_deltas(model, loader):
    num_layers = len(model.model.layers)
    total_deltas = torch.zeros(num_layers * len(MODULES), dtype=torch.float32)
    num_batches = 0

    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}

        clear_intervention()
        original_outputs = model(**batch)
        original_logits = original_outputs.logits.detach()

        deltas = []
        for layer_idx in range(num_layers):
            for module in MODULES:
                set_intervention(
                    layer_idx=layer_idx,
                    module=module,
                    mode="module",
                )

                intervened_outputs = model(**batch)
                intervened_logits = intervened_outputs.logits.detach()
                clear_intervention()

                delta = compute_kl_delta(original_logits, intervened_logits)
                deltas.append(delta.detach().cpu().float())

        total_deltas += torch.stack(deltas)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("No batches available for causal scan")

    return total_deltas / num_batches


def overlap(sorted_by_gate, sorted_by_delta, top_k, reverse=True):
    gate_modules = [row["module"] for row in sorted_by_gate[:top_k]]
    delta_modules = [row["module"] for row in sorted_by_delta[:top_k]]
    shared = sorted(set(gate_modules) & set(delta_modules))

    return gate_modules, delta_modules, shared


def print_rank_list(title, rows, key, top_k):
    print(f"\n{title}")
    for row in rows[:top_k]:
        print(
            f"{row['module']:8s} gate={row['gate']:.4f} delta={row['delta']:.6f} {key}_rank={row[key]}"
        )


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module", "gate", "delta", "gate_rank", "delta_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate learned gates with fresh module-level causal deltas."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/tinyllama_gated",
        help="Directory containing trained gated model checkpoint",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of text samples to average for the fresh causal scan",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-csv",
        default="outputs/gate_causal_correlation.csv",
        help="Where to save per-module gate/delta rankings",
    )
    args = parser.parse_args()

    config = load_config("utils/gate.yaml")
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    loader = build_loader(config, tokenizer, args.num_samples)
    gates = get_all_module_gates(model)
    deltas = compute_average_module_deltas(model, loader)
    names = get_module_names(model)

    rows = [
        {
            "module": name,
            "gate": float(gate),
            "delta": float(delta),
        }
        for name, gate, delta in zip(names, gates, deltas)
    ]

    by_gate = sorted(rows, key=lambda row: row["gate"], reverse=True)
    by_delta = sorted(rows, key=lambda row: row["delta"], reverse=True)
    by_gate_low = sorted(rows, key=lambda row: row["gate"])
    by_delta_low = sorted(rows, key=lambda row: row["delta"])

    for rank, row in enumerate(by_gate, start=1):
        row["gate_rank"] = rank
    for rank, row in enumerate(by_delta, start=1):
        row["delta_rank"] = rank

    pearson = pearson_corr(gates, deltas)
    spearman = spearman_corr(gates, deltas)

    gate_top, delta_top, top_shared = overlap(by_gate, by_delta, args.top_k)
    gate_bottom, delta_bottom, bottom_shared = overlap(
        by_gate_low,
        by_delta_low,
        args.top_k,
        reverse=False,
    )

    print("\nGate vs Fresh Causal Delta")
    print(f"num_samples={args.num_samples}")
    print(f"pearson={pearson:.4f}")
    print(f"spearman={spearman:.4f}")
    print(f"top_{args.top_k}_overlap={len(top_shared)}/{args.top_k}")
    print(f"bottom_{args.top_k}_overlap={len(bottom_shared)}/{args.top_k}")

    print("\nShared top modules:")
    print(", ".join(top_shared) if top_shared else "none")

    print("\nShared bottom modules:")
    print(", ".join(bottom_shared) if bottom_shared else "none")

    print_rank_list("Top by learned gate:", by_gate, "gate", args.top_k)
    print_rank_list("Top by fresh causal delta:", by_delta, "delta", args.top_k)
    print_rank_list("Lowest by learned gate:", by_gate_low, "gate", args.top_k)
    print_rank_list("Lowest by fresh causal delta:", by_delta_low, "delta", args.top_k)

    rows = sorted(rows, key=lambda row: row["module"])
    write_csv(rows, args.output_csv)
    print(f"\nSaved per-module correlation table to {args.output_csv}")


if __name__ == "__main__":
    main()
