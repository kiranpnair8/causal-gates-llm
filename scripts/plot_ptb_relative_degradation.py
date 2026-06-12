from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["CausalGate", "GateSkip", "MoD", "AdaSkip", "CALM-HS", "CALM-SM"]
DEGRADATION_5 = [10.3, 22.7, 23.3, 64.1, 103.7, 137.8]
DEGRADATION_10 = [13.1, 51.7, 78.4, 95.4, 293.5, 368.6]


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if height > 100 else 0,
        )


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.arange(len(METHODS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bars_5 = ax.bar(
        x - width / 2,
        DEGRADATION_5,
        width,
        label="5% compute savings",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.7,
    )
    bars_10 = ax.bar(
        x + width / 2,
        DEGRADATION_10,
        width,
        label="10% compute savings",
        color="#F58518",
        edgecolor="black",
        linewidth=0.7,
    )

    annotate_bars(ax, bars_5)
    annotate_bars(ax, bars_10)

    ax.set_title("Relative PTB Perplexity Degradation", pad=12)
    ax.set_ylabel("PTB PPL Increase (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, max(DEGRADATION_10) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    png_path = figures_dir / "ptb_relative_degradation.png"
    pdf_path = figures_dir / "ptb_relative_degradation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
