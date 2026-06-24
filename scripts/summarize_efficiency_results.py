import argparse
import csv
from pathlib import Path


MODEL_ALIASES = {
    "tinyllama": "TinyLlama-1.1B",
    "tinyllama-1.1b": "TinyLlama-1.1B",
    "tinyllama/tinyllama-1.1b-chat-v1.0": "TinyLlama-1.1B",
    "qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct",
    "qwen/qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct",
}


def canonical_model(value):
    normalized = str(value).strip().lower()
    return MODEL_ALIASES.get(normalized, str(value).strip())


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_removal(row):
    for key in ("removal", "target_saved", "target_removal", "realized_saved"):
        if key in row and row[key] not in ("", None):
            return float(row[key])
    raise KeyError("No removal column found")


def index_rows(rows):
    indexed = {}
    for row in rows:
        model = canonical_model(row["model"])
        removal = row_removal(row)
        indexed[(model, round(removal, 4))] = row
    return indexed


def find_quality_row(rows, model, removal):
    candidates = [
        row for row in rows
        if canonical_model(row.get("model", "")) == model
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs(row_removal(row) - removal))
    return best if abs(row_removal(best) - removal) <= 0.03 else None


def optional_float(row, keys):
    if row is None:
        return ""
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return float(value)
    return ""


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Merge CausalGate quality, FLOP, latency, and throughput results."
    )
    parser.add_argument("--flops-csv", default="results/flops_profile.csv")
    parser.add_argument("--latency-csv", default="results/latency_throughput.csv")
    parser.add_argument(
        "--quality-csv",
        nargs="*",
        default=[
            "results/tinyllama_causalgate.csv",
            "results/qwen3b_causalgate.csv",
        ],
    )
    parser.add_argument("--output-csv", default="results/efficiency_summary.csv")
    args = parser.parse_args()

    flops_rows = read_csv(args.flops_csv)
    latency_rows = read_csv(args.latency_csv)
    if not flops_rows:
        raise FileNotFoundError(f"No FLOP results found at {args.flops_csv}")
    if not latency_rows:
        raise FileNotFoundError(f"No latency results found at {args.latency_csv}")

    quality_rows = []
    for path in args.quality_csv:
        quality_rows.extend(read_csv(path))

    latency_index = index_rows(latency_rows)
    rows = []
    for flop_row in flops_rows:
        model = canonical_model(flop_row["model"])
        removal = float(flop_row["removal"])
        key = (model, round(removal, 4))
        latency_row = latency_index.get(key)
        if latency_row is None:
            raise ValueError(f"Missing latency row for {model} at removal={removal}")
        quality_row = find_quality_row(quality_rows, model, removal)

        rows.append({
            "model": model,
            "removal": removal,
            "wikitext_ppl": optional_float(
                quality_row, ("wikitext_ppl", "WikiText PPL")
            ),
            "c4_ppl": optional_float(quality_row, ("c4_ppl", "C4 PPL")),
            "flop_reduction_percent": float(flop_row["flop_reduction_percent"]),
            "latency_speedup": float(latency_row["latency_speedup"]),
            "throughput_speedup": float(latency_row["throughput_speedup"]),
            "batch_size": int(latency_row["batch_size"]),
            "num_modules": int(flop_row["num_modules"]),
            "num_skipped": int(flop_row["num_skipped"]),
        })

    write_csv(rows, args.output_csv)
    print("\nCausalGate Efficiency Summary")
    print("| model | removal | WikiText PPL | C4 PPL | FLOP reduction | latency | throughput |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        wiki = "" if row["wikitext_ppl"] == "" else f"{row['wikitext_ppl']:.4f}"
        c4 = "" if row["c4_ppl"] == "" else f"{row['c4_ppl']:.4f}"
        print(
            f"| {row['model']} | {row['removal']:.0%} | {wiki} | {c4} | "
            f"{row['flop_reduction_percent']:.2f}% | "
            f"{row['latency_speedup']:.3f}x | "
            f"{row['throughput_speedup']:.3f}x |"
        )
    if not quality_rows:
        print(
            "\nWarning: no quality CSVs were found. WikiText and C4 columns "
            "were left blank; pass them with --quality-csv."
        )
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
