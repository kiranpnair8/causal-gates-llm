from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ["Qwen2.5-3B", "Llama-3.1-8B"]

WIKITEXT_PPL = {
    "Full model": [12.864708627112647, 12.876924709440688],
    "5% module removal": [14.24895212452269, 15.313481160506582],
    "10% module removal": [16.45518324098659, 15.981638765388565],
}

C4_PPL = {
    "Full model": [13.548308489189528, 10.947976091027526],
    "5% module removal": [14.84084015654677, 12.735831111167647],
    "10% module removal": [16.774172458480137, 13.4328467976935],
}

STYLES = {
    "Full model": {
        "color": "#6B7280",
        "marker": "o",
        "linestyle": ":",
        "linewidth": 1.7,
    },
    "5% module removal": {
        "color": "#277DA1",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.2,
    },
    "10% module removal": {
        "color": "#F3722C",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.2,
    },
}


def plot_panel(ax, values, panel_title):
    x = np.arange(len(MODELS))
    for label, ppl in values.items():
        style = STYLES[label]
        ax.plot(
            x,
            ppl,
            label=label,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            **style,
        )

    ax.set_title(panel_title, pad=7, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("Model")
    ax.set_ylabel("Perplexity (PPL)")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    plot_panel(axes[0], WIKITEXT_PPL, "(a) WikiText-2")
    plot_panel(axes[1], C4_PPL, "(b) C4")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "ppl_3B8B.png"
    pdf_path = output_dir / "ppl_3B8B.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
