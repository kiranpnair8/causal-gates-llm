import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


MODULE_RE = re.compile(r"^L(?P<layer>\d+)\.(?P<kind>attn|mlp)$")
ROWS = {"attn": 0, "mlp": 1}
ROW_LABELS = ["Attention", "MLP"]


def load_kl_scores(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "module" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'module' column")

        value_col = None
        for candidate in ("kl_delta", "delta", "kl", "kl_score", "importance"):
            if candidate in reader.fieldnames:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError(
                f"{path} must contain one KL column: kl_delta, delta, kl, kl_score, or importance"
            )

        for row in reader:
            module = row["module"].strip()
            match = MODULE_RE.match(module)
            if match is None:
                continue
            rows.append({
                "module": module,
                "layer": int(match.group("layer")),
                "kind": match.group("kind"),
                "kl": float(row[value_col]),
            })

    if not rows:
        raise ValueError(f"No TinyLlama module rows found in {path}")
    return rows


def build_heatmap(rows):
    max_layer = max(row["layer"] for row in rows)
    heatmap = np.full((2, max_layer + 1), np.nan, dtype=float)

    for row in rows:
        heatmap[ROWS[row["kind"]], row["layer"]] = row["kl"]

    missing = np.argwhere(np.isnan(heatmap))
    if len(missing) > 0:
        missing_labels = [f"L{col:02d}.{'attn' if r == 0 else 'mlp'}" for r, col in missing]
        raise ValueError("Missing KL values for modules: " + ", ".join(missing_labels))

    return heatmap


def cell_for_module(module):
    match = MODULE_RE.match(module)
    if match is None:
        raise ValueError(f"Invalid module name: {module}")
    return ROWS[match.group("kind")], int(match.group("layer"))


def annotate_cells(ax, matrix, image):
    norm = image.norm
    cmap = image.cmap
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if luminance > 0.55 else "white"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )


def add_highlights(ax, rows, matrix, use_log_values=False):
    sorted_rows = sorted(rows, key=lambda row: row["kl"], reverse=True)
    top5 = sorted_rows[:5]
    bottom5 = sorted_rows[-5:]

    for row in top5:
        r, c = cell_for_module(row["module"])
        ax.add_patch(
            patches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
        )

    for row in bottom5:
        r, c = cell_for_module(row["module"])
        ax.add_patch(
            patches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
                linestyle="--",
            )
        )

    top_patch = patches.Patch(
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
        label="Top-5 Causally Important Modules",
    )
    bottom_patch = patches.Patch(
        facecolor="none",
        edgecolor="black",
        linewidth=1.8,
        linestyle="--",
        label="Bottom-5 Causally Important Modules",
    )
    ax.legend(
        handles=[top_patch, bottom_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        fontsize=10,
    )


def plot_heatmap(rows, matrix, output_base, logscale=False):
    if logscale:
        plot_matrix = np.log10(matrix + 1e-6)
        colorbar_label = "Causal Importance log10(KL + 1e-6)"
        title = "Module-Level Causal Importance in TinyLlama-1.1B (Log Scale)"
    else:
        plot_matrix = matrix
        colorbar_label = "Causal Importance (KL Divergence)"
        title = "Module-Level Causal Importance in TinyLlama-1.1B"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(plot_matrix, cmap="coolwarm", aspect="auto")

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=10)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Layer Index")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(idx) for idx in range(matrix.shape[1])])
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(ROW_LABELS)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_cells(ax, plot_matrix, image)
    add_highlights(ax, rows, matrix, use_log_values=logscale)

    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(rows):
    sorted_rows = sorted(rows, key=lambda row: row["kl"], reverse=True)
    top10 = sorted_rows[:10]
    bottom10 = sorted_rows[-10:]

    print("\nTop 10 modules by KL")
    for row in top10:
        print(f"{row['module']:8s} KL={row['kl']:.6f}")

    print("\nBottom 10 modules by KL")
    for row in reversed(bottom10):
        print(f"{row['module']:8s} KL={row['kl']:.6f}")

    attn_values = [row["kl"] for row in rows if row["kind"] == "attn"]
    mlp_values = [row["kl"] for row in rows if row["kind"] == "mlp"]
    print(f"\nMean KL for attention modules: {np.mean(attn_values):.6f}")
    print(f"Mean KL for MLP modules: {np.mean(mlp_values):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Plot TinyLlama module-level causal importance heatmap.")
    parser.add_argument("--input-csv", default="outputs/oracle_kl_module_ranking.csv")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--no-logscale", action="store_true")
    args = parser.parse_args()

    rows = load_kl_scores(args.input_csv)
    matrix = build_heatmap(rows)

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_base = figure_dir / "causal_importance_heatmap"
    plot_heatmap(rows, matrix, output_base, logscale=False)
    print(f"Saved {output_base}.png")
    print(f"Saved {output_base}.pdf")

    if not args.no_logscale:
        log_output_base = figure_dir / "causal_importance_heatmap_logscale"
        plot_heatmap(rows, matrix, log_output_base, logscale=True)
        print(f"Saved {log_output_base}.png")
        print(f"Saved {log_output_base}.pdf")

    print_summary(rows)


if __name__ == "__main__":
    main()
