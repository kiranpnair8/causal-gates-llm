import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


CLASSIFICATION_BENCHMARKS = ("hellaswag", "piqa", "csqa", "winogrande")
LM_BENCHMARKS = ("wikitext", "c4")
DEFAULT_BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40)
BASELINE_METHODS = (
    "calm_softmax",
    "calm_hidden_state",
    "mod",
    "gateskip",
    "adaskip",
    "act_norm",
)
CAUSALGATE_ALIASES = {"causalgate", "causalgate_ours", "causalgate_topk_binary", "CausalGate (Ours)"}
METHOD_ALIASES = {
    "causalgate": "causalgate",
    "causalgate_ours": "causalgate",
    "causalgate_topk_binary": "causalgate",
    "CausalGate (Ours)": "causalgate",
    "calm_softmax": "calm_softmax",
    "CALM (Softmax)": "calm_softmax",
    "calm_hidden_state": "calm_hidden_state",
    "calm_hidden_state_saturation": "calm_hidden_state",
    "CALM (Hidden-State)": "calm_hidden_state",
    "mod": "mod",
    "mod_random_router": "mod",
    "MoD": "mod",
    "gateskip": "gateskip",
    "gateskip_style": "gateskip",
    "GateSkip": "gateskip",
    "adaskip": "adaskip",
    "adaskip_style": "adaskip",
    "AdaSkip": "adaskip",
    "act_norm": "act_norm",
    "activation_l2_pruning": "act_norm",
    "Act-Norm (Zero-Shot)": "act_norm",
}
BENCHMARK_ALIASES = {
    "hellaswag": "hellaswag",
    "HellaSwag": "hellaswag",
    "piqa": "piqa",
    "PIQA": "piqa",
    "csqa": "csqa",
    "commonsenseqa": "csqa",
    "commonsense_qa": "csqa",
    "CSQA": "csqa",
    "winogrande": "winogrande",
    "WinoGrande": "winogrande",
    "wikitext": "wikitext",
    "wikitext2": "wikitext",
    "WikiText": "wikitext",
    "WikiText-2": "wikitext",
    "c4": "c4",
    "C4": "c4",
}

# Aggregate benchmark values from the paper table. These are useful for
# audit/context only; the script must not run significance tests from them.
EMBEDDED_AGGREGATE_RESULTS = {
    ("full_model", 0.00): {
        "wikitext": 13.58,
        "c4": 9.89,
        "hellaswag": 0.484,
        "piqa": 0.707,
        "csqa": 0.367,
        "winogrande": 0.540,
    },
    ("causalgate", 0.05): {
        "wikitext": 16.00,
        "c4": 10.84,
        "hellaswag": 0.445,
        "piqa": 0.723,
        "csqa": 0.359,
        "winogrande": 0.540,
    },
    ("calm_softmax", 0.05): {
        "wikitext": 32.33,
        "c4": 20.85,
        "hellaswag": 0.383,
        "piqa": 0.664,
        "csqa": 0.320,
        "winogrande": 0.524,
    },
    ("calm_hidden_state", 0.05): {
        "wikitext": 23.11,
        "c4": 28.97,
        "hellaswag": 0.453,
        "piqa": 0.734,
        "csqa": 0.313,
        "winogrande": 0.540,
    },
    ("mod", 0.05): {
        "wikitext": 25.04,
        "c4": 16.00,
        "hellaswag": 0.484,
        "piqa": 0.648,
        "csqa": 0.359,
        "winogrande": 0.516,
    },
    ("gateskip", 0.05): {
        "wikitext": 14.71,
        "c4": 16.12,
        "hellaswag": 0.461,
        "piqa": 0.684,
        "csqa": 0.313,
        "winogrande": 0.540,
    },
    ("adaskip", 0.05): {
        "wikitext": 27.08,
        "c4": 26.42,
        "hellaswag": 0.406,
        "piqa": 0.633,
        "csqa": 0.273,
        "winogrande": 0.510,
    },
    ("act_norm", 0.05): {
        "wikitext": 25.26,
        "c4": 19.59,
        "hellaswag": 0.359,
        "piqa": 0.606,
        "csqa": 0.301,
        "winogrande": 0.520,
    },
    ("causalgate", 0.10): {
        "wikitext": 18.46,
        "c4": 12.84,
        "hellaswag": 0.410,
        "piqa": 0.715,
        "csqa": 0.332,
        "winogrande": 0.506,
    },
    ("calm_softmax", 0.10): {
        "wikitext": 71.96,
        "c4": 48.35,
        "hellaswag": 0.383,
        "piqa": 0.648,
        "csqa": 0.297,
        "winogrande": 0.522,
    },
    ("calm_hidden_state", 0.10): {
        "wikitext": 46.43,
        "c4": 70.59,
        "hellaswag": 0.391,
        "piqa": 0.641,
        "csqa": 0.281,
        "winogrande": 0.538,
    },
    ("mod", 0.10): {
        "wikitext": 60.29,
        "c4": 40.75,
        "hellaswag": 0.406,
        "piqa": 0.605,
        "csqa": 0.336,
        "winogrande": 0.492,
    },
    ("gateskip", 0.10): {
        "wikitext": 22.43,
        "c4": 27.34,
        "hellaswag": 0.441,
        "piqa": 0.613,
        "csqa": 0.293,
        "winogrande": 0.518,
    },
    ("adaskip", 0.10): {
        "wikitext": 40.40,
        "c4": 43.75,
        "hellaswag": 0.410,
        "piqa": 0.602,
        "csqa": 0.246,
        "winogrande": 0.514,
    },
    ("act_norm", 0.10): {
        "wikitext": 81.11,
        "c4": 161.86,
        "hellaswag": 0.262,
        "piqa": 0.512,
        "csqa": 0.234,
        "winogrande": 0.490,
    },
    ("causalgate", 0.20): {
        "wikitext": 67.0,
        "c4": 40.0,
        "hellaswag": 0.340,
        "piqa": 0.610,
        "csqa": 0.260,
        "winogrande": 0.490,
    },
    ("calm_softmax", 0.20): {
        "wikitext": 198.0,
        "c4": 141.0,
        "hellaswag": 0.350,
        "piqa": 0.610,
        "csqa": 0.300,
        "winogrande": 0.510,
    },
    ("calm_hidden_state", 0.20): {
        "wikitext": 80.0,
        "c4": 130.0,
        "hellaswag": 0.370,
        "piqa": 0.590,
        "csqa": 0.250,
        "winogrande": 0.520,
    },
    ("mod", 0.20): {
        "wikitext": 407.0,
        "c4": 269.0,
        "hellaswag": 0.310,
        "piqa": 0.510,
        "csqa": 0.260,
        "winogrande": 0.520,
    },
    ("gateskip", 0.20): {
        "wikitext": 81.0,
        "c4": 100.0,
        "hellaswag": 0.320,
        "piqa": 0.550,
        "csqa": 0.260,
        "winogrande": 0.480,
    },
    ("adaskip", 0.20): {
        "wikitext": 1816.0,
        "c4": 531.0,
        "hellaswag": 0.246,
        "piqa": 0.554,
        "csqa": 0.200,
        "winogrande": 0.490,
    },
    ("act_norm", 0.20): {
        "wikitext": 2781.6,
        "c4": 1085.0,
        "hellaswag": 0.273,
        "piqa": 0.535,
        "csqa": 0.231,
        "winogrande": 0.510,
    },
    ("causalgate", 0.30): {
        "wikitext": 116.0,
        "c4": 69.0,
        "hellaswag": 0.340,
        "piqa": 0.540,
        "csqa": 0.250,
        "winogrande": 0.490,
    },
    ("calm_softmax", 0.30): {
        "wikitext": 601.0,
        "c4": 422.0,
        "hellaswag": 0.290,
        "piqa": 0.600,
        "csqa": 0.290,
        "winogrande": 0.500,
    },
    ("calm_hidden_state", 0.30): {
        "wikitext": 185.0,
        "c4": 299.0,
        "hellaswag": 0.290,
        "piqa": 0.600,
        "csqa": 0.230,
        "winogrande": 0.520,
    },
    ("mod", 0.30): {
        "wikitext": 1313.0,
        "c4": 990.0,
        "hellaswag": 0.250,
        "piqa": 0.530,
        "csqa": 0.240,
        "winogrande": 0.510,
    },
    ("gateskip", 0.30): {
        "wikitext": 365.0,
        "c4": 491.0,
        "hellaswag": 0.257,
        "piqa": 0.520,
        "csqa": 0.250,
        "winogrande": 0.470,
    },
    ("adaskip", 0.30): {
        "wikitext": 2064.0,
        "c4": 623.0,
        "hellaswag": 0.238,
        "piqa": 0.539,
        "csqa": 0.170,
        "winogrande": 0.490,
    },
    ("act_norm", 0.30): {
        "wikitext": 3563.2,
        "c4": 3385.3,
        "hellaswag": 0.238,
        "piqa": 0.535,
        "csqa": 0.176,
        "winogrande": 0.500,
    },
    ("causalgate", 0.40): {
        "wikitext": 1365.0,
        "c4": 441.0,
        "hellaswag": 0.280,
        "piqa": 0.500,
        "csqa": 0.200,
        "winogrande": 0.480,
    },
    ("calm_softmax", 0.40): {
        "wikitext": 1574.0,
        "c4": 1047.0,
        "hellaswag": 0.280,
        "piqa": 0.530,
        "csqa": 0.250,
        "winogrande": 0.500,
    },
    ("calm_hidden_state", 0.40): {
        "wikitext": 1472.0,
        "c4": 712.0,
        "hellaswag": 0.270,
        "piqa": 0.570,
        "csqa": 0.220,
        "winogrande": 0.490,
    },
    ("mod", 0.40): {
        "wikitext": 4433.0,
        "c4": 3403.0,
        "hellaswag": 0.260,
        "piqa": 0.510,
        "csqa": 0.180,
        "winogrande": 0.500,
    },
    ("gateskip", 0.40): {
        "wikitext": 1698.0,
        "c4": 2046.0,
        "hellaswag": 0.250,
        "piqa": 0.490,
        "csqa": 0.200,
        "winogrande": 0.460,
    },
    ("adaskip", 0.40): {
        "wikitext": 4260.0,
        "c4": 4509.0,
        "hellaswag": 0.235,
        "piqa": 0.527,
        "csqa": 0.160,
        "winogrande": 0.460,
    },
    ("act_norm", 0.40): {
        "wikitext": 20825.4,
        "c4": 31094.0,
        "hellaswag": 0.254,
        "piqa": 0.527,
        "csqa": 0.207,
        "winogrande": 0.488,
    },
}


def normalize_method(value):
    return METHOD_ALIASES.get(str(value).strip(), str(value).strip())


def normalize_benchmark(value):
    return BENCHMARK_ALIASES.get(str(value).strip(), str(value).strip().lower())


def normalize_budget(value):
    text = str(value).strip().replace("%", "")
    numeric = float(text)
    if numeric > 1.0:
        numeric /= 100.0
    return round(numeric, 6)


def parse_boolish(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "correct"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "incorrect"}:
        return 0
    return int(float(text))


def read_csv_if_exists(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_existing_files(root, preferred_names, fallback_patterns):
    root = Path(root)
    found = []
    for directory_name in ("results", "outputs"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for name in preferred_names:
            path = directory / name
            if path.exists():
                found.append(path)
        for pattern in fallback_patterns:
            found.extend(sorted(directory.glob(pattern)))
    seen = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_classification_predictions(root):
    paths = find_existing_files(
        root,
        preferred_names=("per_example_predictions.csv", "classification_predictions.csv"),
        fallback_patterns=("*prediction*.csv", "*per_example*.csv", "*example*.csv"),
    )
    records = defaultdict(dict)
    loaded_files = []
    required = {"method", "budget", "benchmark", "example_id", "correct"}
    for path in paths:
        rows = read_csv_if_exists(path)
        if not rows or not required.issubset(rows[0].keys()):
            continue
        loaded_files.append(str(path))
        for row in rows:
            benchmark = normalize_benchmark(row["benchmark"])
            if benchmark not in CLASSIFICATION_BENCHMARKS:
                continue
            key = (
                normalize_method(row["method"]),
                normalize_budget(row["budget"]),
                benchmark,
            )
            records[key][str(row["example_id"])] = parse_boolish(row["correct"])
    return records, loaded_files


def load_lm_nll(root):
    paths = find_existing_files(
        root,
        preferred_names=("per_sequence_nll.csv", "per_document_nll.csv", "lm_nll_by_sequence.csv"),
        fallback_patterns=("*nll*.csv", "*sequence*.csv", "*document*.csv"),
    )
    records = defaultdict(dict)
    loaded_files = []
    required = {"method", "budget", "benchmark", "example_id", "nll"}
    for path in paths:
        rows = read_csv_if_exists(path)
        if not rows or not required.issubset(rows[0].keys()):
            continue
        loaded_files.append(str(path))
        for row in rows:
            benchmark = normalize_benchmark(row["benchmark"])
            if benchmark not in LM_BENCHMARKS:
                continue
            key = (
                normalize_method(row["method"]),
                normalize_budget(row["budget"]),
                benchmark,
            )
            records[key][str(row["example_id"])] = float(row["nll"])
    return records, loaded_files


def load_seed_results(root):
    paths = find_existing_files(
        root,
        preferred_names=("seed_results.csv", "multi_seed_results.csv"),
        fallback_patterns=("*seed*.csv",),
    )
    records = defaultdict(dict)
    loaded_files = []
    required = {"method", "budget", "benchmark", "seed", "value"}
    for path in paths:
        rows = read_csv_if_exists(path)
        if not rows or not required.issubset(rows[0].keys()):
            continue
        loaded_files.append(str(path))
        for row in rows:
            key = (
                normalize_method(row["method"]),
                normalize_budget(row["budget"]),
                normalize_benchmark(row["benchmark"]),
            )
            records[key][str(row["seed"])] = float(row["value"])
    return records, loaded_files


def has_aggregate_outputs(root):
    root = Path(root)
    aggregate_patterns = (
        "outputs/*tradeoff*.csv",
        "outputs/*all_methods*.csv",
        "outputs/*baseline*.csv",
        "outputs/*baselines*.csv",
        "results/*tradeoff*.csv",
        "results/*all_methods*.csv",
        "results/*baseline*.csv",
        "results/*baselines*.csv",
    )
    files = []
    for pattern in aggregate_patterns:
        files.extend(sorted(root.glob(pattern)))
    return [str(path) for path in files]


def embedded_aggregate_value(method, budget, benchmark):
    values = EMBEDDED_AGGREGATE_RESULTS.get((normalize_method(method), normalize_budget(budget)))
    if not values:
        return "NA"
    return values.get(normalize_benchmark(benchmark), "NA")


def embedded_aggregate_difference(budget, benchmark, baseline):
    cg_value = embedded_aggregate_value("causalgate", budget, benchmark)
    base_value = embedded_aggregate_value(baseline, budget, benchmark)
    if cg_value == "NA" or base_value == "NA":
        return "NA"
    return cg_value - base_value


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def paired_bootstrap_diff(a_values, b_values, samples, seed):
    rng = random.Random(seed)
    n = len(a_values)
    observed = sum(a - b for a, b in zip(a_values, b_values)) / max(n, 1)
    boot = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            total += a_values[idx] - b_values[idx]
        boot.append(total / max(n, 1))
    return observed, percentile(boot, 0.025), percentile(boot, 0.975)


def exact_mcnemar_pvalue(cg_values, baseline_values):
    b = 0
    c = 0
    for cg, base in zip(cg_values, baseline_values):
        if cg == 1 and base == 0:
            b += 1
        elif cg == 0 and base == 1:
            c += 1
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail), b, c


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank_pvalue(diffs):
    diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0, 0.0
    abs_with_sign = sorted((abs(d), 1 if d > 0 else -1) for d in diffs)
    ranks = [0.0] * n
    idx = 0
    while idx < n:
        j = idx + 1
        while j < n and abs(abs_with_sign[j][0] - abs_with_sign[idx][0]) <= 1e-12:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[k] = avg_rank
        idx = j
    w_pos = sum(rank for rank, (_, sign) in zip(ranks, abs_with_sign) if sign > 0)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var == 0:
        return 1.0, w_pos
    z = (abs(w_pos - mean) - 0.5) / math.sqrt(var)
    pvalue = 2.0 * (1.0 - normal_cdf(abs(z)))
    return max(0.0, min(1.0, pvalue)), w_pos


def paired_seed_wilcoxon(cg_seed_values, base_seed_values):
    seeds = sorted(set(cg_seed_values) & set(base_seed_values))
    cg = [cg_seed_values[s] for s in seeds]
    base = [base_seed_values[s] for s in seeds]
    diffs = [a - b for a, b in zip(cg, base)]
    pvalue, statistic = wilcoxon_signed_rank_pvalue(diffs)
    return seeds, cg, base, diffs, pvalue, statistic


def std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def cohens_d_paired(diffs):
    sd = std(diffs)
    if sd == 0.0:
        return 0.0
    return (sum(diffs) / len(diffs)) / sd


def holm_bonferroni(rows):
    indexed = []
    for idx, row in enumerate(rows):
        try:
            pvalue = float(row["raw_p_value"])
        except (ValueError, TypeError):
            continue
        if math.isnan(pvalue):
            continue
        indexed.append((pvalue, idx))
    indexed.sort()
    m = len(indexed)
    adjusted = [None] * len(rows)
    running_max = 0.0
    for rank, (pvalue, idx) in enumerate(indexed, start=1):
        corrected = min(1.0, (m - rank + 1) * pvalue)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    for idx, row in enumerate(rows):
        if adjusted[idx] is not None:
            row["holm_p_value"] = adjusted[idx]
        elif row["holm_p_value"] == "":
            row["holm_p_value"] = "NA"
    return rows


def empty_row(budget, benchmark, baseline, test_type, reason, available_data):
    cg_value = embedded_aggregate_value("causalgate", budget, benchmark)
    base_value = embedded_aggregate_value(baseline, budget, benchmark)
    aggregate_diff = embedded_aggregate_difference(budget, benchmark, baseline)
    ppl_difference = aggregate_diff if benchmark in LM_BENCHMARKS else "NA"
    mean_difference = aggregate_diff if benchmark in CLASSIFICATION_BENCHMARKS else "NA"
    return {
        "budget": budget,
        "benchmark": benchmark,
        "comparison": f"causalgate_vs_{baseline}",
        "test_type": test_type,
        "available_data": available_data,
        "n_pairs": 0,
        "causalgate_mean": cg_value,
        "baseline_mean": base_value,
        "mean_difference": mean_difference,
        "ppl_difference": ppl_difference,
        "mean_nll_difference": "NA",
        "ci95_low": "NA",
        "ci95_high": "NA",
        "raw_p_value": "NA",
        "holm_p_value": "NA",
        "effect_size": "NA",
        "status": "not_testable",
        "notes": reason,
    }


def classification_row(budget, benchmark, baseline, cg_records, base_records, bootstrap_samples, seed):
    ids = sorted(set(cg_records) & set(base_records))
    cg_values = [cg_records[i] for i in ids]
    base_values = [base_records[i] for i in ids]
    diff, ci_low, ci_high = paired_bootstrap_diff(cg_values, base_values, bootstrap_samples, seed)
    pvalue, b, c = exact_mcnemar_pvalue(cg_values, base_values)
    return {
        "budget": budget,
        "benchmark": benchmark,
        "comparison": f"causalgate_vs_{baseline}",
        "test_type": "paired_bootstrap_accuracy_and_exact_mcnemar",
        "available_data": "per_example_predictions",
        "n_pairs": len(ids),
        "causalgate_mean": sum(cg_values) / len(cg_values),
        "baseline_mean": sum(base_values) / len(base_values),
        "mean_difference": diff,
        "ppl_difference": "NA",
        "mean_nll_difference": "NA",
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "raw_p_value": pvalue,
        "holm_p_value": "",
        "effect_size": "NA",
        "status": "ok",
        "notes": f"McNemar discordant counts: causalgate_correct_only={b}; baseline_correct_only={c}",
    }


def lm_row(budget, benchmark, baseline, cg_records, base_records, bootstrap_samples, seed):
    ids = sorted(set(cg_records) & set(base_records))
    cg_values = [cg_records[i] for i in ids]
    base_values = [base_records[i] for i in ids]
    diffs = [a - b for a, b in zip(cg_values, base_values)]
    mean_diff, ci_low, ci_high = paired_bootstrap_diff(cg_values, base_values, bootstrap_samples, seed)
    pvalue, statistic = wilcoxon_signed_rank_pvalue(diffs)
    cg_nll = sum(cg_values) / len(cg_values)
    base_nll = sum(base_values) / len(base_values)
    return {
        "budget": budget,
        "benchmark": benchmark,
        "comparison": f"causalgate_vs_{baseline}",
        "test_type": "paired_bootstrap_nll_and_wilcoxon_signed_rank",
        "available_data": "per_sequence_nll",
        "n_pairs": len(ids),
        "causalgate_mean": cg_nll,
        "baseline_mean": base_nll,
        "mean_difference": "NA",
        "ppl_difference": math.exp(cg_nll) - math.exp(base_nll),
        "mean_nll_difference": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "raw_p_value": pvalue,
        "holm_p_value": "",
        "effect_size": "NA",
        "status": "ok",
        "notes": f"Wilcoxon signed-rank statistic={statistic}",
    }


def seed_row(budget, benchmark, baseline, cg_records, base_records):
    seeds, cg, base, diffs, pvalue, statistic = paired_seed_wilcoxon(cg_records, base_records)
    return {
        "budget": budget,
        "benchmark": benchmark,
        "comparison": f"causalgate_vs_{baseline}",
        "test_type": "matched_seed_wilcoxon_signed_rank",
        "available_data": "matched_seed_results",
        "n_pairs": len(seeds),
        "causalgate_mean": f"{sum(cg) / len(cg):.8g} ± {std(cg):.8g}",
        "baseline_mean": f"{sum(base) / len(base):.8g} ± {std(base):.8g}",
        "mean_difference": sum(diffs) / len(diffs),
        "ppl_difference": "NA",
        "mean_nll_difference": "NA",
        "ci95_low": "NA",
        "ci95_high": "NA",
        "raw_p_value": pvalue,
        "holm_p_value": "",
        "effect_size": cohens_d_paired(diffs),
        "status": "ok",
        "notes": f"Wilcoxon signed-rank statistic={statistic}; seeds={','.join(seeds)}",
    }


def build_rows(classification, lm_nll, seeds, aggregate_files, args):
    rows = []
    embedded_aggregates_available = bool(EMBEDDED_AGGREGATE_RESULTS)
    available_data = "aggregate_only" if aggregate_files or embedded_aggregates_available else "no_result_files_found"
    aggregate_note = (
        "Only aggregate result values are available; significance tests require paired per-example predictions, "
        "per-sequence NLL values, or repeated matched seeds. Aggregate values are reported for audit only."
        if aggregate_files or embedded_aggregates_available
        else "No per-example, per-sequence, matched-seed, or aggregate result files were found in outputs/ or results/."
    )
    for budget in args.budgets:
        budget = normalize_budget(budget)
        for baseline in args.baselines:
            baseline = normalize_method(baseline)
            for benchmark in CLASSIFICATION_BENCHMARKS:
                cg_key = ("causalgate", budget, benchmark)
                base_key = (baseline, budget, benchmark)
                if cg_key in classification and base_key in classification:
                    ids = set(classification[cg_key]) & set(classification[base_key])
                    if ids:
                        rows.append(
                            classification_row(
                                budget,
                                benchmark,
                                baseline,
                                classification[cg_key],
                                classification[base_key],
                                args.bootstrap_samples,
                                args.seed,
                            )
                        )
                        continue
                if cg_key in seeds and base_key in seeds:
                    common_seeds = set(seeds[cg_key]) & set(seeds[base_key])
                    if common_seeds:
                        rows.append(seed_row(budget, benchmark, baseline, seeds[cg_key], seeds[base_key]))
                        continue
                rows.append(
                    empty_row(
                        budget,
                        benchmark,
                        baseline,
                        "paired_bootstrap_accuracy_and_exact_mcnemar",
                        aggregate_note,
                        available_data,
                    )
                )
            for benchmark in LM_BENCHMARKS:
                cg_key = ("causalgate", budget, benchmark)
                base_key = (baseline, budget, benchmark)
                if cg_key in lm_nll and base_key in lm_nll:
                    ids = set(lm_nll[cg_key]) & set(lm_nll[base_key])
                    if ids:
                        rows.append(
                            lm_row(
                                budget,
                                benchmark,
                                baseline,
                                lm_nll[cg_key],
                                lm_nll[base_key],
                                args.bootstrap_samples,
                                args.seed,
                            )
                        )
                        continue
                if cg_key in seeds and base_key in seeds:
                    common_seeds = set(seeds[cg_key]) & set(seeds[base_key])
                    if common_seeds:
                        rows.append(seed_row(budget, benchmark, baseline, seeds[cg_key], seeds[base_key]))
                        continue
                rows.append(
                    empty_row(
                        budget,
                        benchmark,
                        baseline,
                        "paired_bootstrap_nll_and_wilcoxon_signed_rank",
                        aggregate_note,
                        available_data,
                    )
                )
    return holm_bonferroni(rows)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "budget",
        "benchmark",
        "comparison",
        "test_type",
        "available_data",
        "n_pairs",
        "causalgate_mean",
        "baseline_mean",
        "mean_difference",
        "ppl_difference",
        "mean_nll_difference",
        "ci95_low",
        "ci95_high",
        "raw_p_value",
        "holm_p_value",
        "effect_size",
        "status",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def latex_escape(text):
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_tex(rows, output_path, loaded_files, aggregate_files, embedded_aggregates_available):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by scripts/statistical_tests.py\n")
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write("\\caption{Statistical significance audit for CausalGate comparisons.}\n")
        handle.write("\\label{tab:statistical-tests}\n")
        if not ok_rows:
            handle.write("\\begin{tabular}{p{0.94\\linewidth}}\n")
            handle.write("\\toprule\n")
            handle.write(
                "No valid paired significance test could be performed from the available artifacts. "
                "The benchmark table contains one aggregate score per method, budget, and benchmark; "
                "paired tests require per-example predictions, per-sequence negative log-likelihoods, "
                "or matched repeated-seed results. No samples were fabricated from aggregate means.\\\\\n"
            )
            if loaded_files:
                handle.write(f"Recognized paired input files: {latex_escape(', '.join(loaded_files))}.\\\\\n")
            if aggregate_files:
                handle.write(f"Aggregate files found: {latex_escape(', '.join(aggregate_files))}.\\\\\n")
            elif embedded_aggregates_available:
                handle.write(
                    "Aggregate benchmark values are embedded in \\texttt{scripts/statistical\\_tests.py} "
                    "for audit/context only.\\\\\n"
                )
            else:
                handle.write("No aggregate result CSV files were found under \\texttt{outputs/} or \\texttt{results/}.\\\\\n")
            handle.write("\\bottomrule\n")
            handle.write("\\end{tabular}\n")
        else:
            handle.write("\\begin{tabular}{llllrrrr}\n")
            handle.write("\\toprule\n")
            handle.write("Budget & Benchmark & Comparison & Test & $n$ & Diff. & 95\\% CI & $p_{\\mathrm{Holm}}$\\\\\n")
            handle.write("\\midrule\n")
            for row in ok_rows:
                diff = row["mean_difference"]
                if diff == "NA":
                    diff = row["mean_nll_difference"]
                if diff == "NA":
                    diff = row["ppl_difference"]
                ci = f"[{format_value(row['ci95_low'])}, {format_value(row['ci95_high'])}]"
                handle.write(
                    f"{float(row['budget']) * 100:.0f}\\% & "
                    f"{latex_escape(row['benchmark'])} & "
                    f"{latex_escape(row['comparison'])} & "
                    f"{latex_escape(row['test_type'])} & "
                    f"{row['n_pairs']} & "
                    f"{format_value(diff)} & "
                    f"{latex_escape(ci)} & "
                    f"{format_value(row['holm_p_value'])}\\\\\n"
                )
            handle.write("\\bottomrule\n")
            handle.write("\\end{tabular}\n")
        handle.write("\\end{table*}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired significance tests only when per-example predictions, per-sequence NLLs, "
            "or matched-seed results are available. Aggregate-only tables are marked not testable."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--baselines", nargs="+", default=list(BASELINE_METHODS))
    parser.add_argument("--output-csv", default="results/statistical_tests.csv")
    parser.add_argument("--output-tex", default="results/statistical_tests_summary.tex")
    args = parser.parse_args()

    root = Path(args.root)
    classification, classification_files = load_classification_predictions(root)
    lm_nll, lm_files = load_lm_nll(root)
    seeds, seed_files = load_seed_results(root)
    aggregate_files = has_aggregate_outputs(root)
    embedded_aggregates_available = bool(EMBEDDED_AGGREGATE_RESULTS)
    loaded_files = classification_files + lm_files + seed_files

    rows = build_rows(classification, lm_nll, seeds, aggregate_files, args)
    write_csv(rows, args.output_csv)
    write_tex(rows, args.output_tex, loaded_files, aggregate_files, embedded_aggregates_available)

    ok_count = sum(row["status"] == "ok" for row in rows)
    print(f"Saved statistical test table to {args.output_csv}")
    print(f"Saved LaTeX summary to {args.output_tex}")
    if ok_count == 0:
        print(
            "No valid significance tests were run: only aggregate/no result files were available. "
            "Per-example predictions, per-sequence NLLs, or matched-seed results are required."
        )
    else:
        print(f"Completed {ok_count} valid paired significance tests.")


if __name__ == "__main__":
    main()
