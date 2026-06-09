import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"module", "gate", "delta", "gate_rank", "delta_rank"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        for row in reader:
            rows.append({
                "module": row["module"],
                "gate": float(row["gate"]),
                "delta": float(row["delta"]),
                "gate_rank": float(row["gate_rank"]),
                "delta_rank": float(row["delta_rank"]),
            })

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def compute_metrics(rows, top_k=10):
    gate = [row["gate"] for row in rows]
    delta = [row["delta"] for row in rows]
    gate_rank = [row["gate_rank"] for row in rows]
    delta_rank = [row["delta_rank"] for row in rows]

    pearson_gate_delta = pearson(gate, delta)
    spearman_rank = pearson(gate_rank, delta_rank)

    top_by_gate = {row["module"] for row in sorted(rows, key=lambda row: row["gate_rank"])[:top_k]}
    top_by_delta = {row["module"] for row in sorted(rows, key=lambda row: row["delta_rank"])[:top_k]}
    bottom_by_gate = {row["module"] for row in sorted(rows, key=lambda row: row["gate_rank"], reverse=True)[:top_k]}
    bottom_by_delta = {row["module"] for row in sorted(rows, key=lambda row: row["delta_rank"], reverse=True)[:top_k]}

    return {
        "pearson": pearson_gate_delta,
        "spearman": spearman_rank,
        "top_overlap": len(top_by_gate & top_by_delta),
        "bottom_overlap": len(bottom_by_gate & bottom_by_delta),
        "top_shared": sorted(top_by_gate & top_by_delta),
        "bottom_shared": sorted(bottom_by_gate & bottom_by_delta),
    }


def annotate_modules(ax, rows):
    top5 = sorted(rows, key=lambda row: row["delta_rank"])[:5]
    bottom5 = sorted(rows, key=lambda row: row["delta_rank"], reverse=True)[:5]

    for row in top5:
        ax.annotate(
            row["module"],
            xy=(row["delta_rank"], row["gate_rank"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="black",
            arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.6},
        )

    for row in bottom5:
        ax.annotate(
            row["module"],
            xy=(row["delta_rank"], row["gate_rank"]),
            xytext=(5, -9),
            textcoords="offset points",
            fontsize=8,
            color="#b00000",
            arrowprops={"arrowstyle": "-", "color": "#b00000", "linewidth": 0.6},
        )


def plot(rows, metrics, output_base):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    x = np.asarray([row["delta_rank"] for row in rows], dtype=float)
    y = np.asarray([row["gate_rank"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x, y, s=42, color="#2f6fbb", edgecolor="black", linewidth=0.4, alpha=0.9)

    max_rank = int(max(x.max(), y.max()))
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="gray", linewidth=1.2, label="Perfect rank agreement")

    annotate_modules(ax, rows)

    ax.set_xlabel("Causal KL Rank")
    ax.set_ylabel("Learned Gate Rank")
    ax.set_title("Learned Gate Ranking vs. Causal Importance Ranking")
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    text = (
        f"Pearson(gate, KL) = {metrics['pearson']:.3f}\n"
        f"Spearman(ranks) = {metrics['spearman']:.3f}\n"
        f"Top-10 overlap = {metrics['top_overlap']}/10\n"
        f"Bottom-10 overlap = {metrics['bottom_overlap']}/10"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot learned gate rank against causal KL rank.")
    parser.add_argument("--input-csv", default="outputs/gate_causal_correlation.csv")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    metrics = compute_metrics(rows)

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_base = figure_dir / "gate_vs_causal_rank"
    plot(rows, metrics, output_base)

    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Top-10 overlap: {metrics['top_overlap']}/10")
    print(f"Bottom-10 overlap: {metrics['bottom_overlap']}/10")
    print(f"Saved {output_base}.png")
    print(f"Saved {output_base}.pdf")


if __name__ == "__main__":
    main()
