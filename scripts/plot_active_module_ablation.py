from pathlib import Path

import matplotlib.pyplot as plt


ACTIVE_PERCENT = [100, 75, 50, 25]
HELLASWAG = [0.6045, 0.2851, 0.2615, 0.2626]
COMMONSENSEQA = [0.1916, 0.1960, 0.2015, 0.1968]
PIQA = [0.7427, 0.5401, 0.5285, 0.5274]
PPL = [14.9, 431, 2554, 26722]


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax_acc = plt.subplots(figsize=(8.2, 4.8))

    acc_lines = [
        ax_acc.plot(
            ACTIVE_PERCENT,
            HELLASWAG,
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            color="#4C78A8",
            label="HellaSwag",
        )[0],
        ax_acc.plot(
            ACTIVE_PERCENT,
            COMMONSENSEQA,
            marker="s",
            linewidth=2.2,
            markersize=6.5,
            color="#54A24B",
            label="CommonsenseQA",
        )[0],
        ax_acc.plot(
            ACTIVE_PERCENT,
            PIQA,
            marker="^",
            linewidth=2.2,
            markersize=7.0,
            color="#F58518",
            label="PIQA",
        )[0],
    ]

    ax_ppl = ax_acc.twinx()
    ppl_line = ax_ppl.plot(
        ACTIVE_PERCENT,
        PPL,
        marker="D",
        linewidth=2.3,
        markersize=6.5,
        color="#B279A2",
        linestyle="--",
        label="PPL",
    )[0]

    ax_acc.set_title("Effect of Active Module Ratio on Model Performance", pad=12)
    ax_acc.set_xlabel("Active Modules (%)")
    ax_acc.set_ylabel("Accuracy")
    ax_ppl.set_ylabel("Perplexity (PPL, log scale)")
    ax_ppl.set_yscale("log")

    ax_acc.set_xticks(ACTIVE_PERCENT)
    ax_acc.set_xlim(103, 22)
    ax_acc.set_ylim(0.15, 0.78)
    ax_acc.invert_xaxis()

    ax_acc.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax_ppl.grid(False)
    ax_acc.set_axisbelow(True)

    lines = acc_lines + [ppl_line]
    labels = [line.get_label() for line in lines]
    ax_acc.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)

    for spine in ["top"]:
        ax_acc.spines[spine].set_visible(False)
        ax_ppl.spines[spine].set_visible(False)

    fig.tight_layout()

    png_path = figures_dir / "active_module_ablation.png"
    pdf_path = figures_dir / "active_module_ablation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
