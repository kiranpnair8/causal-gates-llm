import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from canonical_kl import CANONICAL_CSV, load_canonical_kl


GATE_KEY_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.(attn|mlp)_gate\.gate_logit$")


def load_gate_values(checkpoint_dir):
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            gate_tensors = {key: handle.get_tensor(key) for key in handle.keys() if GATE_KEY_RE.search(key)}
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        gate_tensors = {key: value for key, value in state.items() if GATE_KEY_RE.search(key)}
    else:
        raise FileNotFoundError(f"No trained gate checkpoint in {checkpoint_dir}")

    values = {}
    for key, tensor in gate_tensors.items():
        match = GATE_KEY_RE.search(key)
        module = f"L{int(match[1]):02d}.{match[2]}"
        if module in values or tensor.numel() != 1:
            raise ValueError(f"Duplicate or non-scalar gate for {module}")
        values[module] = float(torch.sigmoid(tensor.float()).item())
        if not math.isfinite(values[module]):
            raise ValueError(f"Non-finite gate for {module}")
    return values


def load_rows(path, checkpoint_dir):
    # Figure 2 uses independent all-open KL, not a trained-soft-gated scan.
    canonical = load_canonical_kl(path)
    gates = load_gate_values(checkpoint_dir)
    expected = {row["module"] for row in canonical}
    if set(gates) != expected:
        raise ValueError(f"Checkpoint gate modules do not match canonical KL: missing={expected - set(gates)}, extra={set(gates) - expected}")
    gate_ranks = {
        module: rank for rank, module in enumerate(sorted(gates, key=lambda name: (-gates[name], name)), 1)
    }
    rows = [
        {"module": row["module"], "gate": gates[row["module"]],
         "delta": row["mean_kl"], "gate_rank": gate_ranks[row["module"]],
         "delta_rank": row["kl_rank"]}
        for row in canonical
    ]
    ranks_from_values = {
        row["module"]: rank for rank, row in enumerate(sorted(canonical, key=lambda row: (-row["mean_kl"], row["module"])), 1)
    }
    if any(row["delta_rank"] != ranks_from_values[row["module"]] for row in rows):
        raise AssertionError("Figure 2 KL ranks disagree with Figure 7 mean KL values")
    print("All 44 Figure 2 KL ranks match Figure 7 mean KL values")
    l19 = next(row for row in rows if row["module"] == "L19.mlp")
    print(f"L19.mlp mean_kl={l19['delta']:.17g} kl_rank={l19['delta_rank']}")
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


def plot_split_rank_heatmap(rows, output_path):
    sorted_rows = sorted(rows, key=lambda row: row["delta_rank"])
    if len(sorted_rows) != 44:
        raise ValueError(f"Expected 44 modules for the split heatmap, found {len(sorted_rows)}")

    matrix = np.asarray(
        [[row["delta_rank"], row["gate_rank"]] for row in sorted_rows],
        dtype=float,
    )
    norm = Normalize(vmin=1, vmax=max(1, matrix.max()))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.9))
    fig.subplots_adjust(left=0.115, right=0.875, bottom=0.09, top=0.91, wspace=0.30)

    for panel_idx, ax in enumerate(axes):
        start = panel_idx * 22
        panel_rows = sorted_rows[start:start + 22]
        panel_matrix = matrix[start:start + 22]
        image = ax.imshow(panel_matrix, cmap="viridis_r", norm=norm, aspect="auto")
        ax.set_title(
            "(a) KL ranks 1--22" if panel_idx == 0 else "(b) KL ranks 23--44",
            fontsize=10,
            pad=7,
        )
        ax.set_xticks([0, 1], labels=["KL Rank", "Gate Rank"])
        ax.tick_params(axis="x", labelsize=8, length=0, pad=5)
        ax.set_yticks(np.arange(22), labels=[row["module"] for row in panel_rows])
        ax.tick_params(axis="y", labelsize=8, length=0, pad=5)

        for local_idx, label in enumerate(ax.get_yticklabels()):
            global_idx = start + local_idx
            if global_idx < 10:
                label.set_fontweight("bold")
                label.set_color("#006400")
            elif global_idx >= 34:
                label.set_fontweight("bold")
                label.set_color("#8b0000")

        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 22, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row_idx in range(22):
            for col_idx in range(2):
                value = int(round(panel_matrix[row_idx, col_idx]))
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if luminance > 0.5 else "white",
                    fontweight="bold" if start + row_idx < 10 or start + row_idx >= 34 else "normal",
                )

    colorbar_ax = fig.add_axes([0.91, 0.12, 0.018, 0.75])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Rank (1 = highest importance)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot learned gate rank against causal KL rank as a heatmap.")
    parser.add_argument("--input-csv", default=CANONICAL_CSV)
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--split-output", default="figures/gate_rank_heatmap_split.png")
    parser.add_argument("--original", action="store_true", help="Also regenerate the original single-panel figure.")
    args = parser.parse_args()

    rows = load_rows(args.input_csv, args.checkpoint_dir)
    metrics = compute_metrics(rows)

    if args.original:
        figure_dir = Path(args.figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        output_base = figure_dir / "gate_rank_heatmap"
        plot_rank_heatmap(rows, metrics, output_base)

    plot_split_rank_heatmap(rows, args.split_output)

    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Top-10 overlap: {metrics['top_overlap']}/10")
    print(f"Bottom-10 overlap: {metrics['bottom_overlap']}/10")
    if args.original:
        print(f"Saved {output_base}.png")
        print(f"Saved {output_base}.pdf")
    print(f"Saved {args.split_output}")


if __name__ == "__main__":
    main()
