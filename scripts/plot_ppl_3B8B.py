from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


MODEL_RESULTS = {
    "Qwen2.5-3B-Instruct": {
        "WikiText-2": [12.8647086271, 14.2489521245, 16.4551832410, 20.7643413074, 25.6129328318],
        "C4": [13.5483084892, 14.8408401565, 16.7741724585, 20.0804266215, 23.3878558872],
    },
    "Llama-3.1-8B-Instruct": {
        "WikiText-2": [12.8769247094, 15.3134811605, 15.9816387654, 21.4861012971, 29.4471135473],
        "C4": [10.9479760910, 12.7358311112, 13.4328467977, 17.6909408676, 22.4697410870],
    },
}

TARGET_REMOVAL = [0.0, 0.05, 0.10, 0.15, 0.20]

DATASET_STYLES = {
    "WikiText-2": {
        "color": "#2166AC",
        "marker": "o",
    },
    "C4": {
        "color": "#B2182B",
        "marker": "s",
    },
}


def plot_panel(ax, panel_label, model_name, results):
    x = TARGET_REMOVAL

    for dataset_name, style in DATASET_STYLES.items():
        ax.plot(
            x,
            results[dataset_name],
            color=style["color"],
            marker=style["marker"],
            linestyle=":",
            linewidth=2.0,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label=dataset_name,
        )

    ax.set_title(f"{panel_label} {model_name}", pad=7, fontweight="semibold")
    ax.set_xlabel("Target Module Removal (%)")
    ax.set_ylabel("Perplexity (PPL)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    ax.margins(x=0.04, y=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    panels = ["(a)", "(b)"]

    for ax, panel_label, (model_name, results) in zip(
        axes,
        panels,
        MODEL_RESULTS.items(),
    ):
        plot_panel(ax, panel_label, model_name, results)

    fig.tight_layout()

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
