"""Plot the measured EMA-Only versus CausalGate ablation without modifying results."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter


BUDGETS = (5, 10, 20, 30, 40)
METHODS = ("CausalGate", "EMA-Only")
MODULE_COUNTS = {5: 2, 10: 4, 20: 9, 30: 13, 40: 18}
METRICS = {
    "wikitext_ppl": ("WikiText-2", "#2764A5"),
    "c4_ppl": ("C4", "#D55E00"),
    "hellaswag_acc": ("HellaSwag", "#008B73"),
    "piqa_acc": ("PIQA", "#7856A6"),
    "commonsenseqa_acc": ("CommonsenseQA", "#B54850"),
    "winogrande_acc": ("WinoGrande", "#A37713"),
}
METHOD_STYLES = {
    "CausalGate": ("-", "o"),
    "EMA-Only": ("--", "s"),
}
REQUIRED_COLUMNS = {"method", "target_removal_pct", "modules_removed", "realized_removal_pct", *METRICS}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/tinyllama_gated_ema_ablation/ema_ablation_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    return parser.parse_args()


def load_results(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        raw_rows = list(reader)
    if len(raw_rows) != 10:
        raise ValueError(f"Expected exactly 10 rows (2 methods x 5 budgets), found {len(raw_rows)}")

    results = {}
    for row in raw_rows:
        method = row["method"].strip()
        if method not in METHODS:
            raise ValueError(f"Unexpected method: {method!r}")
        budget_value = float(row["target_removal_pct"])
        if not math.isfinite(budget_value) or not budget_value.is_integer():
            raise ValueError(f"Invalid removal budget: {row['target_removal_pct']!r}")
        budget = int(budget_value)
        if budget not in BUDGETS or (method, budget) in results:
            raise ValueError(f"Unexpected or duplicate method/budget: {method}, {budget}")
        if int(row["modules_removed"]) != MODULE_COUNTS[budget]:
            raise ValueError(f"Unexpected module count at {budget}% for {method}")
        realized = float(row["realized_removal_pct"])
        if not math.isfinite(realized) or not math.isclose(
            realized, 100 * MODULE_COUNTS[budget] / 44, rel_tol=0, abs_tol=1e-3
        ):
            raise ValueError(f"Inconsistent realized removal at {budget}% for {method}")
        for key in METRICS:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} at {budget}% for {method}")
            if key.endswith("_ppl") and value <= 0:
                raise ValueError(f"Non-positive perplexity at {budget}% for {method}")
            if key.endswith("_acc") and not 0 <= value <= 1:
                raise ValueError(f"Accuracy outside [0, 1] at {budget}% for {method}")
        results[method, budget] = row

    expected = {(method, budget) for method in METHODS for budget in BUDGETS}
    if set(results) != expected:
        raise ValueError(f"Missing method/budget pairs: {sorted(expected - set(results))}")
    return results


def curve_values(results, method, metric):
    rows = [results[method, budget] for budget in BUDGETS]
    raw_values = [row[metric] for row in rows]
    print(f"{METRICS[metric][0]} | {method} | " + ", ".join(
        f"{budget}%={value}" for budget, value in zip(BUDGETS, raw_values)
    ))
    return [float(value) for value in raw_values]


def plot_panel(ax, results, metrics):
    plotted = []
    for metric in metrics:
        _, color = METRICS[metric]
        for method in METHODS:
            linestyle, marker = METHOD_STYLES[method]
            values = curve_values(results, method, metric)
            plotted.extend(values)
            ax.plot(
                BUDGETS,
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.65,
                markersize=4.0,
                markerfacecolor="white" if method == "EMA-Only" else color,
                markeredgewidth=1.0,
            )
    ax.set_xticks(BUDGETS)
    ax.set_xlim(3, 42)
    ax.set_xlabel("Module Removal (%)")
    ax.grid(axis="y", color="#C8CDD2", linewidth=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plotted


def main():
    args = parse_args()
    results = load_results(args.input)
    print(f"Validated 10 rows from {args.input}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, (ax_lm, ax_reason) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    lm_values = plot_panel(ax_lm, results, ("wikitext_ppl", "c4_ppl"))
    reason_values = plot_panel(ax_reason, results, (
        "hellaswag_acc", "piqa_acc", "commonsenseqa_acc", "winogrande_acc"
    ))
    ax_lm.set_title("(a) Language Modeling", pad=6)
    ax_reason.set_title("(b) Commonsense Reasoning", pad=6)
    if max(lm_values) / min(lm_values) >= 10:
        ax_lm.set_yscale("log")
        ax_lm.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax_lm.yaxis.set_major_formatter(ScalarFormatter())
        ax_lm.set_ylabel("Perplexity ↓ (log scale)")
        print("Perplexity axis: logarithmic (raw values plotted)")
    else:
        ax_lm.set_ylabel("Perplexity ↓")
        print("Perplexity axis: linear")
    ax_reason.set_ylabel("Accuracy ↑")
    reason_min, reason_max = min(reason_values), max(reason_values)
    reason_pad = max(0.03, 0.07 * (reason_max - reason_min))
    ax_reason.set_ylim(max(0, reason_min - reason_pad), min(1, reason_max + reason_pad))

    dataset_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label)
        for label, color in METRICS.values()
    ]
    method_handles = [
        Line2D([0], [0], color="#333333", linestyle=style[0], marker=style[1],
               markersize=4.5, linewidth=1.7, label=method)
        for method, style in METHOD_STYLES.items()
    ]
    fig.legend(handles=dataset_handles, loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=6, frameon=False, columnspacing=1.0, handlelength=1.8)
    fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=2, frameon=False, columnspacing=2.0, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.21, 1, 1), w_pad=1.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "ema_vs_gates.pdf"
    png_path = args.output_dir / "ema_vs_gates.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
