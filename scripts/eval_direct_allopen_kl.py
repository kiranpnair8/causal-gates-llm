"""Evaluate static, independent all-open KL module ranking on Table 1 tasks."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from safetensors import safe_open

from models.load_model import load_tinyllama_with_gates
from scripts.canonical_kl import CANONICAL_CSV, load_canonical_kl, validate_rows
from scripts.eval_all_methods import MethodRunner, evaluate_runner, prepare_datasets
from scripts.eval_common import model_device, set_seed
from scripts.eval_joint_intervention_analysis import validate_checkpoint_gates, sha256_file
from scripts.eval_oracle_kl_baseline import (
    KEEP_LOGIT,
    SKIP_LOGIT,
    apply_all_modules_open,
    apply_binary_gate_mask,
    build_wikitext_loader,
    compute_oracle_kl_ranking,
    get_module_names,
    iter_gate_modules,
)
from scripts.statistical_tests import EMBEDDED_AGGREGATE_RESULTS
from utils.config import load_config


BUDGETS = {5: 2, 10: 4, 20: 9, 30: 13, 40: 18}
DATASETS = ("wikitext", "c4", "hellaswag", "piqa", "csqa", "winogrande")
COUNTS = {"wikitext": 128, "c4": 128, "hellaswag": 256,
          "piqa": 256, "csqa": 256, "winogrande": 500}
METRIC_FIELDS = {"wikitext": "wikitext_ppl", "c4": "c4_ppl",
                 "hellaswag": "hellaswag_acc", "piqa": "piqa_acc",
                 "csqa": "csqa_acc", "winogrande": "winogrande_acc"}
COMPARISON_FIELDS = ("removal", "k", "method", "wikitext_ppl", "c4_ppl",
                     "hellaswag_acc", "piqa_acc", "csqa_acc", "winogrande_acc")


def validate_comparison_schema(row=None):
    expected = {"removal", "k", "method", *METRIC_FIELDS.values()}
    if len(COMPARISON_FIELDS) != len(expected) or set(COMPARISON_FIELDS) != expected:
        raise ValueError("Comparison CSV columns do not match evaluated metric fields")
    if row is not None and set(row) != expected:
        raise ValueError(f"Comparison row has mismatched fields: {sorted(set(row) ^ expected)}")


def comparison_row(removal, k, method, metrics):
    if set(metrics) != set(METRIC_FIELDS):
        raise ValueError("Comparison metrics do not match the six Table 1 metrics")
    row = {"removal": removal, "k": k, "method": method,
           **{field: metrics[metric] for metric, field in METRIC_FIELDS.items()}}
    validate_comparison_schema(row)
    return row


def preflight_comparison_schema():
    validate_comparison_schema()
    empty_metrics = {metric: None for metric in METRIC_FIELDS}
    comparison_row(0, 0, "Full model", empty_metrics)
    for pct, k in BUDGETS.items():
        comparison_row(pct, k, "CausalGate (Table 1, rounded)", empty_metrics)
        comparison_row(pct, k, "Direct-KL", empty_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/tinyllama_gated"))
    parser.add_argument("--canonical-csv", type=Path, default=Path(CANONICAL_CSV),
                        help="Optional Figure-2 test-split CSV for reporting rank differences only")
    parser.add_argument("--output-dir", type=Path, default=Path("results/direct_allopen_kl"))
    parser.add_argument("--ranking-only", action="store_true",
                        help="Audit and print the ranking without downstream evaluation")
    return parser.parse_args()


def expected_names():
    return [f"L{layer:02d}.{kind}" for layer in range(22) for kind in ("attn", "mlp")]


def checkpoint_audit(path):
    checkpoint_file = validate_checkpoint_gates(path)
    if checkpoint_file.suffix == ".safetensors":
        with safe_open(str(checkpoint_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    else:
        keys = list(torch.load(checkpoint_file, map_location="cpu", weights_only=True))
    gate_keys = [key for key in keys if key.endswith(("attn_gate.gate_logit", "mlp_gate.gate_logit"))]
    if len(gate_keys) != 44:
        raise ValueError("Checkpoint must contain exactly 44 gate parameters")
    info = {"checkpoint_path": str(checkpoint_file.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_file),
            "gate_parameter_count": len(gate_keys),
            "other_checkpoint_key_count": len(keys) - len(gate_keys),
            "gate_parameters_loaded": False,
            "backbone_source": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
    return info, set(keys)


def assert_mask(model, removed):
    names = get_module_names(model)
    if names != expected_names():
        raise ValueError("Model gate order differs from canonical 44-module order")
    removed = set(removed)
    for name, gate in zip(names, iter_gate_modules(model)):
        expected = SKIP_LOGIT if name in removed else KEEP_LOGIT
        if not torch.all(gate.gate_logit.detach() == expected):
            raise AssertionError(f"Incorrect binary gate mask at {name}")


def train_calibration_scan(model, tokenizer, config):
    loader = build_wikitext_loader(config, tokenizer, "train", 32)
    if len(loader.dataset) != 32 or loader.batch_size != 1:
        raise ValueError("Direct-KL calibration requires 32 filtered WikiText-2 train examples, batch size 1")
    apply_all_modules_open(model)
    assert_mask(model, ())
    # The existing CSV has no scan metadata. Recompute rather than infer its provenance.
    raw = compute_oracle_kl_ranking(model, loader, expected_names())
    ranked = sorted(raw, key=lambda row: (-row["delta"], row["module"]))
    rows = []
    for rank, row in enumerate(ranked, 1):
        layer, kind = row["module"].split(".")
        rows.append({"module": row["module"], "layer": int(layer[1:]),
                     "module_type": kind, "mean_kl": row["delta"], "kl_rank": rank})
    validate_rows(rows)
    return rows


def compare_figure2_diagnostic(train_rows, canonical_path):
    diagnostic = {"artifact_path": str(canonical_path), "used_for_baseline": False}
    if not canonical_path.exists():
        diagnostic["status"] = "unavailable: Figure-2 CSV not present"
        print("Figure-2 test-based ranking comparison unavailable: CSV not present")
        return diagnostic
    try:
        test_rows = load_canonical_kl(canonical_path)
    except (ValueError, KeyError) as exc:
        diagnostic["status"] = f"unavailable: invalid Figure-2 CSV ({exc})"
        print(diagnostic["status"])
        return diagnostic
    train_names = [row["module"] for row in train_rows]
    test_names = [row["module"] for row in test_rows]
    differing = sum(a != b for a, b in zip(train_names, test_names))
    diagnostic.update({"status": "compared", "artifact_sha256": sha256_file(canonical_path),
                       "rankings_equal": differing == 0, "different_rank_positions": differing,
                       "bottom_k_overlap": {
                           str(pct): len(set(train_names[-k:]) & set(test_names[-k:]))
                           for pct, k in BUDGETS.items()}})
    print(f"Train-calibrated vs Figure-2 test-based ranking: "
          f"{'identical' if differing == 0 else 'different'} ({differing}/44 positions differ)")
    for pct, overlap in diagnostic["bottom_k_overlap"].items():
        print(f"  {pct}% bottom-k overlap: {overlap}/{BUDGETS[int(pct)]}")
    return diagnostic


def validate_full_model(row):
    reference = EMBEDDED_AGGREGATE_RESULTS[("full_model", 0.0)]
    for metric, field in METRIC_FIELDS.items():
        actual = row[field]
        expected = reference[metric]
        tolerance = 0.05 if metric in {"wikitext", "c4"} else 0.005
        if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0):
            raise RuntimeError(
                f"Full-model {metric}={actual:.6f} differs from Table 1 {expected:.6f}; "
                "stopping before Direct-KL evaluation"
            )


def compare_values(direct, gate, metric):
    if math.isclose(direct, gate, rel_tol=0, abs_tol=1e-12):
        return "tie"
    direct_better = direct < gate if metric in {"wikitext", "c4"} else direct > gate
    return "Direct-KL better" if direct_better else "CausalGate better"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    preflight_comparison_schema()
    outputs = {name: args.output_dir / name for name in
               ("ranking.json", "train_calibration.csv", "removed_modules.json",
                "metrics.json", "comparison.csv", "provenance.json")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Direct-KL output already exists; refusing to overwrite")

    config = load_config("utils/gate.yaml")
    if (config["model"]["name"] != "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            or config["model"]["torch_dtype"] != "float16"
            or config["data"]["max_length"] != 512
            or config["data"].get("eval_split", "test") != "test"
            or config["data"]["split"] != "train"
            or config["data"]["dataset_name"] != "wikitext"
            or config["data"]["dataset_config"] != "wikitext-2-raw-v1"):
        raise ValueError("TinyLlama backbone, data, precision, or max length differs from Table 1")
    seed = 123
    set_seed(seed)
    checkpoint_info, checkpoint_keys = checkpoint_audit(args.checkpoint_dir)
    model, tokenizer = load_tinyllama_with_gates(config)
    model.eval()
    if get_module_names(model) != expected_names():
        raise ValueError("Expected 44 alternating attention and MLP modules")
    model_keys = set(model.state_dict())
    checkpoint_info["backbone_missing_keys"] = len(model_keys - checkpoint_keys)
    checkpoint_info["unexpected_keys"] = len(checkpoint_keys - model_keys)
    print("Loading audit:")
    print(json.dumps(checkpoint_info, indent=2))
    print("Backbone is loaded from the pretrained model, as in Table 1; "
          "trained gate logits are inspected but never loaded or used.")
    print("Protocol: calibration=WikiText-2/train first 32 filtered examples, batch=1, "
          "seed=123, final-token KL(intervened || all-open), max_length=512")
    print(f"Evaluation: {COUNTS}; max_length=512; seed=123; fp16; budgets={BUDGETS}")
    ranking = train_calibration_scan(model, tokenizer, config)
    figure2_diagnostic = compare_figure2_diagnostic(ranking, args.canonical_csv)
    removed = {str(pct): {"target_removal_pct": pct, "k": k,
                          "modules": [row["module"] for row in ranking[-k:]]}
               for pct, k in BUDGETS.items()}
    print("\nDirect-KL TRAIN-calibrated ranking (high KL = important):")
    for row in ranking:
        print(f"{row['kl_rank']:2d} {row['module']:8s} {row['mean_kl']:.8g}")
    print("\nDirect-KL removed modules:")
    for pct, entry in removed.items():
        print(f"{pct}%: {entry['modules']}")

    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    provenance = {"git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                  **checkpoint_info, "model": config["model"]["name"],
                  "calibration_source": "fresh independent all-open averaged-KL scan of WikiText-2 train",
                  "calibration_artifact_path": str(outputs["train_calibration.csv"]),
                  "figure2_test_based_diagnostic": figure2_diagnostic,
                  "calibration_dataset": "WikiText-2", "calibration_dataset_config": "wikitext-2-raw-v1",
                  "calibration_split": "train",
                  "calibration_samples": 32, "calibration_batch_size": 1,
                  "calibration_filter": "len(text.strip()) > 20; first 32",
                  "calibration_seed": seed, "kl_orientation": "KL(intervened || all-open)",
                  "kl_token": "final token", "max_sequence_length": 512,
                  "evaluation_sample_counts": COUNTS,
                  "evaluation_splits": {"wikitext": "test", "c4": "validation",
                                        "hellaswag": "validation", "piqa": "validation",
                                        "csqa": "validation", "winogrande": "validation/winogrande_xl"},
                  "evaluation_seed": seed, "precision": "float16", "device": str(model_device(model)),
                  "removal_sets": removed,
                  "note": "Checkpoint gates are not loaded; pretrained backbone and forced-open binary gates only. "
                          "Binary masks suppress residual outputs after branch computation, so removal is theoretical compute savings."}
    if args.ranking_only:
        print("Ranking audit complete; downstream evaluation not started.")
        return

    class EvalArgs:
        datasets = list(DATASETS)
        max_length = 512
        wikitext_samples = 128
        c4_samples = 128
        hellaswag_samples = 256
        piqa_samples = 256
        csqa_samples = 256
        winogrande_samples = 500

    set_seed(seed)
    datasets = prepare_datasets(tokenizer, config, EvalArgs())
    if tuple(datasets) != DATASETS:
        raise RuntimeError("Dataset order differs from Table 1")
    if len(datasets["wikitext"]) != COUNTS["wikitext"] or any(
        len(datasets[name]) != COUNTS[name] for name in DATASETS if name != "wikitext"
    ):
        raise RuntimeError("Prepared dataset count differs from Table 1")

    apply_all_modules_open(model)
    assert_mask(model, ())
    full = evaluate_runner(MethodRunner("full_model", model, tokenizer), 0.0, datasets, EvalArgs())
    validate_full_model(full)
    print("Full-model sanity matches Table 1 rounded values.")
    results = {"full_model": {metric: full[field] for metric, field in METRIC_FIELDS.items()}}
    comparison = [comparison_row(0, 0, "Full model",
                 {metric: full[field] for metric, field in METRIC_FIELDS.items()})]
    for pct, k in BUDGETS.items():
        gate_reference = EMBEDDED_AGGREGATE_RESULTS[("causalgate", pct / 100)]
        comparison.append(comparison_row(pct, k, "CausalGate (Table 1, rounded)",
                                         {metric: gate_reference[metric] for metric in METRIC_FIELDS}))
        skip = set(removed[str(pct)]["modules"])
        keep = {idx for idx, name in enumerate(expected_names()) if name not in skip}
        apply_binary_gate_mask(model, keep)
        assert_mask(model, skip)
        row = evaluate_runner(MethodRunner("Direct-KL", model, tokenizer,
                                           saved_compute=k / 44), k / 44, datasets, EvalArgs())
        metrics = {metric: row[field] for metric, field in METRIC_FIELDS.items()}
        results[str(pct)] = {"k": k, "realized_removal_pct": 100 * k / 44,
                             "removed_modules": removed[str(pct)]["modules"], **metrics}
        comparison.append(comparison_row(pct, k, "Direct-KL", metrics))

    for row in comparison:
        validate_comparison_schema(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(outputs["ranking.json"], {"provenance": provenance, "modules": ranking})
    with outputs["train_calibration.csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("module", "mean_kl", "kl_rank"))
        writer.writeheader()
        writer.writerows({"module": row["module"], "mean_kl": format(row["mean_kl"], ".17g"),
                          "kl_rank": row["kl_rank"]} for row in ranking)
    write_json(outputs["removed_modules.json"], removed)
    write_json(outputs["provenance.json"], provenance)
    write_json(outputs["metrics.json"], results)
    with outputs["comparison.csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison)
    print("\nRemoval | k | WT2 PPL | C4 PPL | Hella | PIQA | CSQA | Wino")
    for pct, k in BUDGETS.items():
        row = results[str(pct)]
        print(f"{pct:>3}% | {k:>2} | {row['wikitext']:.4f} | {row['c4']:.4f} | "
              f"{row['hellaswag']:.4f} | {row['piqa']:.4f} | {row['csqa']:.4f} | "
              f"{row['winogrande']:.4f}")
        outcomes = {metric: compare_values(row[metric],
                    EMBEDDED_AGGREGATE_RESULTS[("causalgate", pct / 100)][metric], metric)
                    for metric in METRIC_FIELDS}
        for metric, outcome in outcomes.items():
            print(f"  {metric}: {outcome}")
        print(f"  Wins: CausalGate={list(outcomes.values()).count('CausalGate better')}/6; "
              f"Direct-KL={list(outcomes.values()).count('Direct-KL better')}/6; "
              f"ties={list(outcomes.values()).count('tie')}/6")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
