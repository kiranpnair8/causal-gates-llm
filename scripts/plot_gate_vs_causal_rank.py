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
    }


def plot_rank_heatmap(rows, metrics, output_base):
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 7,
    })

    sorted_rows = sorted(rows, key=lambda row: row["delta_rank"])
    matrix = np.asarray(
        [[row["delta_rank"], row["gate_rank"]] for row in sorted_rows],
        dtype=float,
    )
    modules = [row["module"] for row in sorted_rows]
    max_rank = int(max(matrix.max(), 1))

    fig, ax = plt.subplots(figsize=(6, 10))
    image = ax.imshow(matrix, cmap="viridis_r", vmin=1, vmax=max_rank, aspect="auto")

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Rank (1 = highest importance)", fontsize=9)

    ax.set_title("Learned Gate Ranking vs. Causal KL Ranking", pad=52)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["KL Rank", "Gate Rank"], fontweight="bold")
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels(modules)

    for idx, label in enumerate(ax.get_yticklabels()):
        if idx < 10:
            label.set_fontweight("bold")
            label.set_color("#006400")
        elif idx >= len(modules) - 10:
            label.set_fontweight("bold")
            label.set_color("#8b0000")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(modules), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(round(matrix[row_idx, col_idx]))
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < max_rank * 0.55 else "black",
                fontweight="bold" if row_idx < 10 or row_idx >= len(modules) - 10 else "normal",
            )

    summary = (
        f"Pearson(gate, KL) = {metrics['pearson']:.3f}    "
        f"Spearman(rank) = {metrics['spearman']:.3f}    "
        f"Top-10 overlap = {metrics['top_overlap']}/10    "
        f"Bottom-10 overlap = {metrics['bottom_overlap']}/10"
    )
    ax.text(
        0.5,
        1.035,
        summary,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.96},
    )

    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot learned gate rank against causal KL rank as a heatmap.")
    parser.add_argument("--input-csv", default="outputs/gate_causal_correlation.csv")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    metrics = compute_metrics(rows)

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_base = figure_dir / "gate_rank_heatmap"
    plot_rank_heatmap(rows, metrics, output_base)

    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Top-10 overlap: {metrics['top_overlap']}/10")
    print(f"Bottom-10 overlap: {metrics['bottom_overlap']}/10")
    print(f"Saved {output_base}.png")
    print(f"Saved {output_base}.pdf")


if __name__ == "__main__":
    main()
