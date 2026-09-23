"""Compare learned gates with their own final training-time EMA targets."""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from safetensors import safe_open

from models.load_model import load_tinyllama_with_gates
from scripts.eval_all_methods import MethodRunner, evaluate_runner, prepare_datasets
from scripts.eval_common import set_seed
from scripts.eval_oracle_kl_baseline import (
    KEEP_LOGIT,
    SKIP_LOGIT,
    apply_binary_gate_mask,
    get_gate_values,
    get_module_names,
    iter_gate_modules,
    load_gate_checkpoint,
)
from utils.config import load_config


BUDGETS = {5: 2, 10: 4, 20: 9, 30: 13, 40: 18}
DATASETS = ("wikitext", "c4", "hellaswag", "piqa", "csqa", "winogrande")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/tinyllama_gated_ema_ablation"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--c4-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--csqa-samples", type=int, default=256)
    parser.add_argument("--winogrande-samples", type=int, default=500)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(a, b):
    a_mean = sum(a) / len(a)
    b_mean = sum(b) / len(b)
    left = [value - a_mean for value in a]
    right = [value - b_mean for value in b]
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else None


def average_descending_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def ordered_indices(values):
    return sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)


def validate_artifacts(directory, seed):
    if directory.resolve() == (PROJECT_ROOT / "outputs/tinyllama_gated").resolve():
        raise ValueError("EMA ablation must not load the old Table 1 checkpoint")
    metadata_path = directory / "training_metadata.json"
    ema_path = directory / "final_ema_targets.pt"
    checkpoint_path = directory / "model.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["seed"] != seed:
        raise ValueError(f"Evaluation seed {seed} differs from training seed {metadata['seed']}")
    for path, key in ((ema_path, "final_ema_targets_sha256"), (checkpoint_path, "checkpoint_sha256")):
        actual = sha256_file(path)
        if actual != metadata[key]:
            raise ValueError(f"Artifact hash mismatch for {path}: {actual} != {metadata[key]}")
    gate_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.(attn|mlp)_gate\.gate_logit$")
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        gate_names = []
        for key in handle.keys():
            match = gate_pattern.search(key)
            if match:
                gate_names.append(f"L{int(match[1]):02d}.{match[2]}")
    payload = torch.load(ema_path, map_location="cpu", weights_only=True)
    expected_names = [f"L{layer:02d}.{kind}" for layer in range(22) for kind in ("attn", "mlp")]
    if len(gate_names) != 44 or set(gate_names) != set(expected_names):
        raise ValueError("Checkpoint does not contain exactly 44 canonical gate logits")
    if payload["module_names"] != expected_names or metadata["module_names"] != expected_names:
        raise ValueError("EMA targets must use the canonical 44-module attention/MLP order")
    if payload["seed"] != seed or payload["final_step"] != metadata["final_step"]:
        raise ValueError("EMA artifact does not match training metadata")
    targets = payload["ema_targets"].detach().float().cpu()
    if targets.shape != (44,) or not torch.isfinite(targets).all():
        raise ValueError("Expected 44 finite final EMA target values")
    if metadata["model"] != "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        raise ValueError("Wrong backbone in training metadata")
    expected = {
        "training": {"learning_rate": 0.01, "grad_accum_steps": 8, "max_steps": 1000},
        "loss": {"lambda_lm": 1.0, "lambda_sparsity": 0.001},
        "causal": {"target_floor": 0.25, "use_ema_targets": True, "ema_beta": 0.9,
                   "use_rank_loss": True, "lambda_causal": 10.0, "lambda_rank": 2.0,
                   "rank_margin": 0.05, "rank_pairs": 128},
        "dataset": {"dataset_name": "wikitext", "dataset_config": "wikitext-2-raw-v1",
                    "split": "train", "max_length": 512},
    }
    for section, values in expected.items():
        for key, value in values.items():
            if metadata[section].get(key) != value:
                raise ValueError(f"Ablation training config mismatch: {section}.{key}")
    return metadata, targets.tolist(), expected_names


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_fingerprints(datasets):
    fingerprints = {}
    for name, examples in datasets.items():
        digest = hashlib.sha256()
        for example in examples:
            if "input_ids" in example:
                payload = example["input_ids"].reshape(-1).tolist()
            else:
                payload = example
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def main():
    args = parse_args()
    directory = args.checkpoint_dir.resolve()
    if args.seed != 123 or args.max_length != 512:
        raise ValueError("The controlled ablation uses seed=123 and max_length=512")
    sample_counts = {name: getattr(args, f"{name}_samples") for name in DATASETS}
    if sample_counts != {"wikitext": 128, "c4": 128, "hellaswag": 256,
                         "piqa": 256, "csqa": 256, "winogrande": 500}:
        raise ValueError("Sample counts must match the Table 1 protocol")
    metadata, ema_values, names = validate_artifacts(directory, args.seed)
    outputs = {name: directory / name for name in (
        "ema_vs_gate_ranking.csv", "ranking_diagnostics.json", "ema_ablation_results.csv"
    )}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing ablation result: {path}")

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    if config["data"]["max_length"] != args.max_length:
        raise ValueError("Evaluation max length differs from training configuration")
    config["model"]["name"] = str(directory)
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, directory)
    model.eval()
    if get_module_names(model) != names:
        raise ValueError("Checkpoint gate order differs from EMA target order")
    gate_values = get_gate_values(model)
    if len(gate_values) != 44 or not all(math.isfinite(value) for value in gate_values):
        raise ValueError("Expected 44 finite learned gates")
    if len(metadata["final_gate_values"]) != 44 or any(
        not math.isclose(actual, recorded, rel_tol=0, abs_tol=1e-6)
        for actual, recorded in zip(gate_values, metadata["final_gate_values"])
    ):
        raise ValueError("Loaded gates differ from the isolated training run")

    gate_order = ordered_indices(gate_values)
    ema_order = ordered_indices(ema_values)
    gate_ranks = {idx: rank for rank, idx in enumerate(gate_order, start=1)}
    ema_ranks = {idx: rank for rank, idx in enumerate(ema_order, start=1)}
    ranking_rows = [
        {"module": name, "gate": gate_values[idx], "ema_target": ema_values[idx],
         "gate_rank": gate_ranks[idx], "ema_rank": ema_ranks[idx]}
        for idx, name in enumerate(names)
    ]
    write_csv(outputs["ema_vs_gate_ranking.csv"], list(ranking_rows[0]), ranking_rows)

    gate_top10 = set(gate_order[:10])
    ema_top10 = set(ema_order[:10])
    gate_bottom10 = set(gate_order[-10:])
    ema_bottom10 = set(ema_order[-10:])
    per_budget = {}
    for pct, k in BUDGETS.items():
        gate_set = set(gate_order[-k:])
        ema_set = set(ema_order[-k:])
        overlap = len(gate_set & ema_set)
        per_budget[str(pct)] = {
            "k": k,
            "causalgate_removed": [names[idx] for idx in gate_order[-k:]],
            "ema_only_removed": [names[idx] for idx in ema_order[-k:]],
            "overlap": overlap,
            "jaccard": overlap / len(gate_set | ema_set),
            "differing_modules": len(gate_set ^ ema_set),
        }

    args.datasets = list(DATASETS)
    datasets = prepare_datasets(tokenizer, config, args)
    if tuple(datasets) != DATASETS:
        raise RuntimeError("Dataset order or selection differs from Table 1")
    fingerprints = dataset_fingerprints(datasets)
    results = []
    for pct, k in BUDGETS.items():
        for method, order in (("CausalGate", gate_order), ("EMA-Only", ema_order)):
            keep_indices = set(order[:-k])
            apply_binary_gate_mask(model, keep_indices)
            actual_removed = {names[idx] for idx in range(44) if idx not in keep_indices}
            expected_removed = {names[idx] for idx in order[-k:]}
            if actual_removed != expected_removed or len(actual_removed) != k:
                raise RuntimeError("Binary mask does not match ranked bottom-k modules")
            for idx, gate in enumerate(iter_gate_modules(model)):
                expected_logit = KEEP_LOGIT if idx in keep_indices else SKIP_LOGIT
                if not torch.all(gate.gate_logit.detach() == expected_logit):
                    raise RuntimeError(f"Unexpected gate mask at {names[idx]}")
            row = evaluate_runner(MethodRunner(method, model, tokenizer, saved_compute=k / 44), k / 44, datasets, args)
            results.append({
                "method": method,
                "target_removal_pct": pct,
                "modules_removed": k,
                "realized_removal_pct": 100 * k / 44,
                "wikitext_ppl": row["wikitext_ppl"],
                "c4_ppl": row["c4_ppl"],
                "hellaswag_acc": row["hellaswag_acc"],
                "piqa_acc": row["piqa_acc"],
                "commonsenseqa_acc": row["csqa_acc"],
                "winogrande_acc": row["winogrande_acc"],
            })
            print(f"{method} target={pct}% removed={k} WikiText PPL={row['wikitext_ppl']:.4f} C4 PPL={row['c4_ppl']:.4f}", flush=True)

    write_csv(outputs["ema_ablation_results.csv"], list(results[0]), results)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    diagnostics = {
        "training_checkpoint": str(directory / "model.safetensors"),
        "training_checkpoint_sha256": metadata["checkpoint_sha256"],
        "final_ema_targets_sha256": metadata["final_ema_targets_sha256"],
        "training_git_commit": metadata["git_commit"],
        "evaluation_git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "seed": args.seed,
        "datasets": {"wikitext": "test", "c4": "validation", "hellaswag": "validation",
                     "piqa": "validation", "csqa": "validation", "winogrande": "validation/winogrande_xl"},
        "sample_counts": sample_counts,
        "dataset_sha256": fingerprints,
        "max_length": args.max_length,
        "pearson_gate_vs_ema": pearson(gate_values, ema_values),
        "spearman_gate_vs_ema": pearson(average_descending_ranks(gate_values), average_descending_ranks(ema_values)),
        "top_10_overlap": len(gate_top10 & ema_top10),
        "bottom_10_overlap": len(gate_bottom10 & ema_bottom10),
        "by_budget": per_budget,
    }
    outputs["ranking_diagnostics.json"].write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results and ranking diagnostics in {directory}")


if __name__ == "__main__":
    main()
