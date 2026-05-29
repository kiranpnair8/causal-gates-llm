import argparse
import csv
import re
from pathlib import Path


METRIC_RE = re.compile(r"(\w+)=([^\s]+)")
GATE_RE = re.compile(r"^(L\d{2}\.(?:attn|mlp))\s+gate=([0-9.]+)")
SEED_RE = re.compile(r"Using seed=(\d+)")


def expand_log_paths(paths):
    expanded = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_dir():
            expanded.extend(sorted(path.glob("*.out")))
        elif path.exists():
            expanded.append(path)
        else:
            raise FileNotFoundError(
                f"Could not find {path}. Pass the real .out filename, a glob expanded by the shell, or a directory like logs/."
            )

    if not expanded:
        raise FileNotFoundError("No .out logs found to compare.")

    return expanded


def parse_step_metrics(line):
    metrics = {}

    for key, value in METRIC_RE.findall(line):
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    if "step" in metrics:
        metrics["step"] = int(metrics["step"])

    return metrics


def parse_log(path):
    path = Path(path)
    seed = None
    final_metrics = None
    top_gates = []
    low_gates = []
    section = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            seed_match = SEED_RE.search(line)
            if seed_match:
                seed = int(seed_match.group(1))
                continue

            if line.startswith("step="):
                final_metrics = parse_step_metrics(line)
                continue

            if line == "Top learned gates:":
                section = "top"
                continue

            if line == "Lowest learned gates:":
                section = "low"
                continue

            match = GATE_RE.match(line)
            if match and section:
                module, value = match.groups()
                row = (module, float(value))

                if section == "top":
                    top_gates.append(row)
                elif section == "low":
                    low_gates.append(row)

    if final_metrics is None:
        raise ValueError(f"No step metrics found in {path}")

    name = f"seed{seed}" if seed is not None else path.stem

    return {
        "name": name,
        "path": str(path),
        "seed": seed,
        "metrics": final_metrics,
        "top_gates": top_gates,
        "low_gates": low_gates,
        "all_ranked_gates": top_gates + low_gates,
    }


def format_float(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_metric_table(runs):
    metric_keys = [
        "step",
        "lm_loss",
        "causal_loss",
        "gate_mean",
        "gate_min",
        "gate_max",
        "gate_range",
        "gate_target_corr",
    ]

    print("# Seed Run Comparison\n")
    print("## Final Metrics\n")
    print("| run | " + " | ".join(metric_keys) + " |")
    print("|" + "---|" * (len(metric_keys) + 1))

    for run in runs:
        values = [run["name"]]
        values.extend(format_float(run["metrics"].get(key, "")) for key in metric_keys)
        print("| " + " | ".join(values) + " |")

    print()


def print_overlap_table(runs, gate_key, title):
    module_sets = [set(module for module, _ in run[gate_key]) for run in runs]
    shared = set.intersection(*module_sets) if module_sets else set()

    print(f"## {title}\n")
    print(f"Shared modules across all runs: {len(shared)}")

    if shared:
        print("\n" + ", ".join(sorted(shared)))

    print()
    print("| run | modules |")
    print("|---|---|")

    for run in runs:
        modules = [module for module, _ in run[gate_key]]
        print(f"| {run['name']} | {', '.join(modules)} |")

    print()


def write_gate_csv(runs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "seed", "rank_group", "rank", "module", "gate"],
        )
        writer.writeheader()

        for run in runs:
            for group_name, gate_key in [("top", "top_gates"), ("lowest", "low_gates")]:
                for rank, (module, gate) in enumerate(run[gate_key], start=1):
                    writer.writerow(
                        {
                            "run": run["name"],
                            "seed": "" if run["seed"] is None else run["seed"],
                            "rank_group": group_name,
                            "rank": rank,
                            "module": module,
                            "gate": f"{gate:.6f}",
                        }
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare learned causal gate rankings across seed run logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to SLURM .out logs, shell-expanded globs, or directories containing .out logs",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write extracted top/lowest gate rankings as CSV",
    )
    args = parser.parse_args()

    log_paths = expand_log_paths(args.logs)
    runs = [parse_log(path) for path in log_paths]

    print_metric_table(runs)
    print_overlap_table(runs, "top_gates", "Top Gate Overlap")
    print_overlap_table(runs, "low_gates", "Lowest Gate Overlap")

    if args.csv:
        write_gate_csv(runs, args.csv)
        print(f"Wrote gate rankings to {args.csv}")


if __name__ == "__main__":
    main()
