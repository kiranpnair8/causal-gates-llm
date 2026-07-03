import argparse
import copy
import csv
import gc
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from models.intervention import clear_intervention, set_intervention
from models.load_model import load_tinyllama_with_gates
from training.losses import gate_sparsity_loss
from utils.config import load_config


DEFAULTS = {
    "lr": 0.01,
    "ema_beta": 0.9,
    "target_floor": 0.25,
    "rank_margin": 0.05,
    "rank_pairs": 128,
    "lambda_causal": 10.0,
    "lambda_rank": 2.0,
    "grad_accum": 8,
    "max_steps": 1000,
}

SWEEPS = {
    "lr": [0.001, 0.005, 0.01, 0.02],
    "ema_beta": [0.7, 0.8, 0.9, 0.95],
    "target_floor": [0.10, 0.25, 0.40],
    "rank_margin": [0.01, 0.05, 0.10],
    "rank_pairs": [32, 64, 128, 256],
    "lambda_causal": [1.0, 5.0, 10.0, 20.0],
    "lambda_rank": [0.0, 1.0, 2.0, 3.0],
    "grad_accum": [4, 8, 16],
    "max_steps": [500, 1000, 1500],
}

CONFIG_PATHS = {
    "lr": ("training", "learning_rate"),
    "ema_beta": ("causal", "ema_beta"),
    "target_floor": ("causal", "target_floor"),
    "rank_margin": ("causal", "rank_margin"),
    "rank_pairs": ("causal", "rank_pairs"),
    "lambda_causal": ("causal", "lambda_causal"),
    "lambda_rank": ("causal", "lambda_rank"),
    "grad_accum": ("training", "grad_accum_steps"),
    "max_steps": ("training", "max_steps"),
}

DISPLAY_NAMES = {
    "lr": "Learning Rate",
    "ema_beta": "EMA Beta",
    "target_floor": "Target Floor",
    "rank_margin": "Rank Margin",
    "rank_pairs": "Ranking Pairs",
    "lambda_causal": r"$\lambda_{\mathrm{causal}}$",
    "lambda_rank": r"$\lambda_{\mathrm{rank}}$",
    "grad_accum": "Gradient Accumulation",
    "max_steps": "Training Steps",
}

PLOT_FILENAMES = {
    "lr": "hp_lr",
    "ema_beta": "hp_ema",
    "target_floor": "hp_target_floor",
    "rank_margin": "hp_rank_margin",
    "rank_pairs": "hp_rank_pairs",
    "lambda_causal": "hp_lambda_causal",
    "lambda_rank": "hp_lambda_rank",
    "grad_accum": "hp_grad_accum",
    "max_steps": "hp_max_steps",
}

CSV_FIELDS = ["hyperparameter", "value", "spearman", "wikitext_ppl_10pct"]
KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_names(model):
    return [
        f"L{layer_idx:02d}.{module_type}"
        for layer_idx in range(len(model.model.layers))
        for module_type in ("attn", "mlp")
    ]


def gate_modules(model):
    for layer in model.model.layers:
        yield layer.attn_gate
        yield layer.mlp_gate


def gate_values(model):
    return torch.stack([
        gate.gate_values_scalar()
        for gate in gate_modules(model)
    ])


def tokenize_dataset(config, tokenizer, split, num_samples=None):
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config"],
        split=split,
    )
    dataset = dataset.filter(lambda row: len(row["text"].strip()) > 20)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset.map(
        lambda row: tokenizer(
            row["text"],
            truncation=True,
            max_length=config["data"]["max_length"],
            padding=False,
        ),
        remove_columns=dataset.column_names,
    )


def make_loader(tokenized, tokenizer, shuffle):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(
        tokenized,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collator,
    )


def kl_delta(original_logits, intervened_logits):
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()
    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean").detach()


@torch.no_grad()
def all_module_deltas(model, batch, original_logits):
    deltas = []
    for layer_idx in range(len(model.model.layers)):
        for module_type in ("attn", "mlp"):
            set_intervention(
                layer_idx=layer_idx,
                module=module_type,
                mode="module",
            )
            outputs = model(**batch)
            clear_intervention()
            deltas.append(kl_delta(original_logits, outputs.logits.detach()))
    return torch.stack(deltas)


def ranking_loss(gates, targets, margin, num_pairs):
    gates = gates.float()
    targets = targets.detach().float()
    num_modules = gates.shape[0]
    idx_a = torch.randint(0, num_modules, (num_pairs,), device=gates.device)
    idx_b = torch.randint(0, num_modules, (num_pairs,), device=gates.device)
    target_diff = targets[idx_a] - targets[idx_b]
    gate_diff = gates[idx_a] - gates[idx_b]
    non_tied = target_diff.abs() > 1e-6
    if non_tied.sum().item() == 0:
        return gates.new_tensor(0.0)
    signs = target_diff[non_tied].sign()
    return F.relu(margin - signs * gate_diff[non_tied]).mean()


def train_one_configuration(model, loader, config):
    model.train()
    parameters = [
        parameter
        for gate in gate_modules(model)
        for parameter in gate.parameters()
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        eps=float(config["training"].get("adam_eps", 1e-6)),
    )

    lambda_lm = float(config["loss"]["lambda_lm"])
    lambda_sparse = float(config["loss"]["lambda_sparsity"])
    lambda_causal = float(config["causal"]["lambda_causal"])
    lambda_rank = float(config["causal"]["lambda_rank"])
    target_floor = float(config["causal"]["target_floor"])
    ema_beta = float(config["causal"]["ema_beta"])
    rank_margin = float(config["causal"]["rank_margin"])
    rank_pairs = int(config["causal"]["rank_pairs"])
    grad_accum = int(config["training"]["grad_accum_steps"])
    max_steps = int(config["training"]["max_steps"])

    optimizer.zero_grad()
    ema_targets = None
    progress = tqdm(loader, desc="Training gates", leave=False)
    for step, batch in enumerate(progress):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        clear_intervention()
        outputs = model(**batch)
        original_logits = outputs.logits.detach()
        deltas = all_module_deltas(model, batch, original_logits)

        normalized = deltas / (deltas.max() + 1e-8)
        current_targets = target_floor + (1.0 - target_floor) * normalized
        current_targets = current_targets.to(model.device)
        if ema_targets is None:
            ema_targets = current_targets.detach().clone()
        else:
            ema_targets.mul_(ema_beta).add_(
                current_targets.detach(),
                alpha=1.0 - ema_beta,
            )
        targets = ema_targets.detach()
        gates = gate_values(model)

        causal_loss = F.mse_loss(gates.float(), targets.float())
        rank_loss = ranking_loss(gates, targets, rank_margin, rank_pairs)
        sparse_loss = gate_sparsity_loss(model)
        loss = (
            lambda_lm * outputs.loss
            + lambda_sparse * sparse_loss
            + lambda_causal * causal_loss
            + lambda_rank * rank_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(config["training"]["max_grad_norm"]),
            )
            optimizer.step()
            optimizer.zero_grad()

        if step >= max_steps:
            break

    clear_intervention()
    return [float(value) for value in gate_values(model).detach().float().cpu()]


def average_ranks(values, descending=True):
    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
        reverse=descending,
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def spearman_correlation(gates, oracle_scores):
    return pearson(
        average_ranks(gates, descending=True),
        average_ranks(oracle_scores, descending=True),
    )


def load_oracle_scores(path, expected_names):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Oracle-KL ranking not found at {path}. Run the Oracle-KL scan first."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "module" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a module column")
        score_field = next(
            (
                field
                for field in ("kl_delta", "delta", "kl", "causal_delta", "importance")
                if field in reader.fieldnames
            ),
            None,
        )
        if score_field is None:
            raise ValueError(f"{path} does not contain an Oracle-KL score column")
        scores = {
            row["module"]: float(row[score_field])
            for row in reader
            if row["module"] in expected_names
        }
    missing = [name for name in expected_names if name not in scores]
    if missing:
        raise ValueError(f"Oracle-KL ranking is missing modules: {missing}")
    return [scores[name] for name in expected_names]


def apply_binary_mask(model, learned_gates, removal=0.10):
    num_modules = len(learned_gates)
    num_skipped = max(0, min(num_modules - 1, round(num_modules * removal)))
    ranked = sorted(
        range(num_modules),
        key=lambda index: learned_gates[index],
        reverse=True,
    )
    kept = set(ranked[: num_modules - num_skipped])
    for index, gate in enumerate(gate_modules(model)):
        gate.gate_logit.data.fill_(KEEP_LOGIT if index in kept else SKIP_LOGIT)


@torch.no_grad()
def evaluate_wikitext_ppl(model, loader):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
    mean_nll = total_nll / max(total_tokens, 1)
    return math.exp(mean_nll)


def apply_sweep_value(base_config, hyperparameter, value):
    config = copy.deepcopy(base_config)
    section, key = CONFIG_PATHS[hyperparameter]
    config[section][key] = value
    return config


def config_signature(config):
    return (
        float(config["training"]["learning_rate"]),
        int(config["training"]["grad_accum_steps"]),
        int(config["training"]["max_steps"]),
        float(config["causal"]["ema_beta"]),
        float(config["causal"]["target_floor"]),
        float(config["causal"]["rank_margin"]),
        int(config["causal"]["rank_pairs"]),
        float(config["loss"]["lambda_lm"]),
        float(config["loss"]["lambda_sparsity"]),
        float(config["causal"]["lambda_causal"]),
        float(config["causal"]["lambda_rank"]),
    )


def load_existing_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(hyperparameter, value):
    return hyperparameter, f"{float(value):.12g}"


def ordered_rows(rows):
    by_key = {
        row_key(row["hyperparameter"], row["value"]): row
        for row in rows
    }
    return [
        by_key[row_key(hyperparameter, value)]
        for hyperparameter, values in SWEEPS.items()
        for value in values
        if row_key(hyperparameter, value) in by_key
    ]


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows(rows))


def format_value(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def create_plots(rows, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    indexed = {
        row_key(row["hyperparameter"], row["value"]): row
        for row in rows
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    for hyperparameter, values in SWEEPS.items():
        sweep_rows = [
            indexed[row_key(hyperparameter, value)]
            for value in values
            if row_key(hyperparameter, value) in indexed
        ]
        if len(sweep_rows) != len(values):
            print(f"Skipping incomplete plot for {hyperparameter}")
            continue

        x_positions = list(range(len(values)))
        spearman = [float(row["spearman"]) for row in sweep_rows]
        perplexity = [float(row["wikitext_ppl_10pct"]) for row in sweep_rows]
        fig, left_axis = plt.subplots(figsize=(4.6, 3.1))
        right_axis = left_axis.twinx()

        left_line = left_axis.plot(
            x_positions,
            spearman,
            color="#2166AC",
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="Spearman",
        )
        right_line = right_axis.plot(
            x_positions,
            perplexity,
            color="#B2182B",
            marker="s",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            label="WikiText-2 PPL",
        )

        default_index = values.index(DEFAULTS[hyperparameter])
        default_line = left_axis.axvline(
            default_index,
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            label="Default",
        )
        left_axis.set_xticks(x_positions)
        left_axis.set_xticklabels([format_value(value) for value in values])
        left_axis.set_xlabel(DISPLAY_NAMES[hyperparameter])
        left_axis.set_ylabel("Spearman Correlation", color="#2166AC")
        right_axis.set_ylabel("WikiText-2 PPL (10% Removal)", color="#B2182B")
        left_axis.tick_params(axis="y", colors="#2166AC")
        right_axis.tick_params(axis="y", colors="#B2182B")
        left_axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        left_axis.set_axisbelow(True)
        left_axis.spines["top"].set_visible(False)
        right_axis.spines["top"].set_visible(False)
        left_axis.set_title(f"Sensitivity to {DISPLAY_NAMES[hyperparameter]}")

        lines = left_line + right_line + [default_line]
        labels = [line.get_label() for line in lines]
        left_axis.legend(lines, labels, loc="best", frameon=False)
        fig.tight_layout()

        stem = figure_dir / PLOT_FILENAMES[hyperparameter]
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def run_study(args):
    base_config = load_config(args.config)
    seed = int(base_config.get("system", {}).get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(base_config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = tokenize_dataset(
        base_config,
        tokenizer,
        split=base_config["data"]["split"],
    )
    eval_data = tokenize_dataset(
        base_config,
        tokenizer,
        split=args.eval_split,
        num_samples=args.wikitext_samples,
    )

    rows = load_existing_rows(args.output_csv) if args.resume else []
    completed = {
        row_key(row["hyperparameter"], row["value"])
        for row in rows
    }
    result_cache = {}
    for row in rows:
        hyperparameter = row["hyperparameter"]
        value = float(row["value"])
        if hyperparameter not in SWEEPS:
            continue
        config = apply_sweep_value(base_config, hyperparameter, value)
        result_cache[config_signature(config)] = (
            float(row["spearman"]),
            float(row["wikitext_ppl_10pct"]),
        )

    for hyperparameter, values in SWEEPS.items():
        for value in values:
            key = row_key(hyperparameter, value)
            if key in completed:
                print(f"Skipping completed {hyperparameter}={value}")
                continue

            config = apply_sweep_value(base_config, hyperparameter, value)
            signature = config_signature(config)
            if signature in result_cache:
                spearman, ppl = result_cache[signature]
                print(f"Reusing default result for {hyperparameter}={value}")
            else:
                print(f"\nRunning {hyperparameter}={value}")
                set_seed(seed)
                model, _ = load_tinyllama_with_gates(config)
                names = module_names(model)
                oracle_scores = load_oracle_scores(args.oracle_kl_csv, names)
                train_loader = make_loader(train_data, tokenizer, shuffle=True)
                eval_loader = make_loader(eval_data, tokenizer, shuffle=False)

                learned_gates = train_one_configuration(model, train_loader, config)
                spearman = spearman_correlation(learned_gates, oracle_scores)
                apply_binary_mask(model, learned_gates, removal=0.10)
                ppl = evaluate_wikitext_ppl(model, eval_loader)
                result_cache[signature] = (spearman, ppl)

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            row = {
                "hyperparameter": hyperparameter,
                "value": value,
                "spearman": spearman,
                "wikitext_ppl_10pct": ppl,
            }
            rows.append(row)
            completed.add(key)
            write_rows(rows, args.output_csv)
            print(
                f"{hyperparameter}={value}: "
                f"spearman={spearman:.4f}, ppl={ppl:.4f}"
            )

    create_plots(ordered_rows(rows), args.figure_dir)
    print(f"\nSaved sensitivity results to {args.output_csv}")
    print(f"Saved sensitivity figures to {args.figure_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time CausalGate hyperparameter sensitivity sweeps."
    )
    parser.add_argument("--config", default="utils/gate.yaml")
    parser.add_argument(
        "--oracle-kl-csv",
        default="outputs/oracle_kl_module_ranking.csv",
    )
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--output-csv",
        default="results/hyperparameter_sensitivity.csv",
    )
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        rows = load_existing_rows(args.output_csv)
        if not rows:
            raise FileNotFoundError(f"No sensitivity results at {args.output_csv}")
        create_plots(rows, args.figure_dir)
        return

    run_study(args)


if __name__ == "__main__":
    main()
