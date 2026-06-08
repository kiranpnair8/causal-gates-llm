import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.eval_oracle_kl_baseline import (
    build_wikitext_loader,
    compute_oracle_kl_ranking,
    get_gate_values,
    get_module_names,
    load_gate_checkpoint,
    load_saved_kl_ranking,
    write_kl_ranking_csv,
)
from models.load_model import load_tinyllama_with_gates
from utils.config import load_config


def rank_desc(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def pearson(x, y):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() <= 1e-8:
        return 0.0
    return float((x * y).sum() / denom)


def load_or_compute_kl(model, tokenizer, config, module_names, args):
    if args.kl_ranking_csv:
        ranking = load_saved_kl_ranking(args.kl_ranking_csv, module_names)
        if ranking is not None:
            return ranking
        raise FileNotFoundError(f"Could not load KL ranking from {args.kl_ranking_csv}")

    default_path = Path(args.oracle_output_csv)
    ranking = load_saved_kl_ranking(default_path, module_names)
    if ranking is not None and not args.recompute_kl:
        return ranking

    if not args.recompute_kl:
        alt_path = Path("outputs/gate_causal_correlation.csv")
        ranking = load_saved_kl_ranking(alt_path, module_names)
        if ranking is not None:
            return ranking

    split = config["data"].get("eval_split", "test")
    loader = build_wikitext_loader(config, tokenizer, split, args.calibration_samples)
    ranking = compute_oracle_kl_ranking(model, loader, module_names)
    write_kl_ranking_csv(ranking, args.oracle_output_csv)
    print(f"Saved recomputed KL ranking to {args.oracle_output_csv}")
    return ranking


def print_table(title, rows):
    print(f"\n{title}")
    print("| module | KL score | gate value | KL rank | gate rank |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['module']} | {row['kl_score']:.6f} | {row['gate_value']:.6f} | "
            f"{row['kl_rank']} | {row['gate_rank']} |"
        )


def write_audit_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module", "kl_score", "gate_value", "kl_rank", "gate_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit KL ranking vs learned CausalGate ranking.")
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--kl-ranking-csv", default=None)
    parser.add_argument("--oracle-output-csv", default="outputs/oracle_kl_module_ranking.csv")
    parser.add_argument("--output-csv", default="outputs/gate_ranking_audit.csv")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--recompute-kl", action="store_true")
    args = parser.parse_args()

    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    module_names = get_module_names(model)
    learned_gate_values = get_gate_values(model)

    kl_ranking = load_or_compute_kl(model, tokenizer, config, module_names, args)
    kl_by_module = {row["module"]: float(row["delta"]) for row in kl_ranking}
    kl_scores = [kl_by_module[name] for name in module_names]

    kl_ranks = rank_desc(kl_scores)
    gate_ranks = rank_desc(learned_gate_values)
    spearman = pearson(kl_ranks, gate_ranks)
    sign = "aligned" if spearman > 0 else "inverted" if spearman < 0 else "flat/unclear"

    rows = []
    for name, kl_score, gate_value, kl_rank, gate_rank in zip(
        module_names,
        kl_scores,
        learned_gate_values,
        kl_ranks,
        gate_ranks,
    ):
        rows.append({
            "module": name,
            "kl_score": kl_score,
            "gate_value": gate_value,
            "kl_rank": kl_rank,
            "gate_rank": gate_rank,
        })

    write_audit_csv(rows, args.output_csv)

    print("\nGate Ranking Audit")
    print(f"Spearman(KL rank, gate rank)={spearman:.4f}")
    print(f"Sign assessment: {sign}")
    print("Rank convention: rank 1 = highest KL / highest gate = most important.")

    print_table("Sorted by KL descending", sorted(rows, key=lambda row: row["kl_score"], reverse=True))
    print_table("Sorted by gate descending", sorted(rows, key=lambda row: row["gate_value"], reverse=True))
    print_table("Sorted by KL ascending", sorted(rows, key=lambda row: row["kl_score"]))
    print_table("Sorted by gate ascending", sorted(rows, key=lambda row: row["gate_value"]))

    low_kl = {row["module"] for row in sorted(rows, key=lambda row: row["kl_score"])[:4]}
    low_gate = {row["module"] for row in sorted(rows, key=lambda row: row["gate_value"])[:4]}
    print("\nBottom-4 overlap")
    print(f"overlap={len(low_kl & low_gate)}/4")
    print(", ".join(sorted(low_kl & low_gate)) if low_kl & low_gate else "none")
    print(f"\nSaved audit table to {args.output_csv}")


if __name__ == "__main__":
    main()
