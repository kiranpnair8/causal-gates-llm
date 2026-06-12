import argparse
import csv
import gc
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

from models.load_model import load_tinyllama_with_gates
from scripts.eval_adaskip_style import add_adaskip_wrappers
from scripts.eval_calm_vs_causalgate import (
    apply_all_modules_open,
    apply_binary_gate_mask,
    calibrate_threshold,
    get_gate_values,
    load_gate_checkpoint,
)
from scripts.eval_gateskip_style import add_gateskip_wrappers, load_gates
from scripts.eval_mod_baselines import disable_mod, enable_mod
from scripts.eval_new_datasets_all_methods import (
    CALM_POLICIES,
    MOD_POLICIES,
    MethodRunner,
    cleanup_model,
    evaluate_lm_ppl,
    load_adaskip_skip_set,
    load_raw_tinyllama,
    load_tokenizer,
)
from utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ptb_texts(split, num_samples):
    try:
        dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
    except Exception:
        dataset = load_dataset("penn_treebank", split=split)

    texts = []
    for example in dataset:
        text = example.get("sentence", example.get("text", "")).strip()
        if len(text) > 0:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


def build_ptb_batches(tokenizer, texts, max_length):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized = [
        tokenizer(text, truncation=True, max_length=max_length, padding=False)
        for text in texts
        if len(text.strip()) > 0
    ]
    return [collator([item]) for item in tokenized]


def make_row(policy, target_saved, realized_saved, mean_nll, ppl):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "ptb_mean_nll": mean_nll,
        "ptb_ppl": ppl,
    }


def evaluate_runner(runner, target_saved, ptb_batches):
    mean_nll, ppl, observed_saved = evaluate_lm_ppl(runner, ptb_batches)
    realized = observed_saved if runner.mode in {"mod", "gateskip"} else target_saved
    return make_row(runner.policy, target_saved, realized, mean_nll, ppl)


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nPenn Treebank Evaluation")
    print("| policy | target_saved | realized_saved | PTB mean NLL | PTB PPL |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['target_saved']:.4f} | {row['realized_saved']:.4f} | "
            f"{row['ptb_mean_nll']:.4f} | {row['ptb_ppl']:.4f} |"
        )


def evaluate_gated_family(rows, ptb_batches, args, config):
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    num_modules = len(model.model.layers) * 2
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)

    split = config["data"].get("eval_split", "test")
    from scripts.eval_calm_vs_causalgate import build_wikitext_loader
    calibration_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.calibration_samples)

    apply_all_modules_open(model)
    rows.append(evaluate_runner(MethodRunner("full_model", model, tokenizer, saved_compute=0.0), 0.0, ptb_batches))

    for target_saved in args.target_saved:
        skip_count = max(0, min(num_modules - 1, round(num_modules * target_saved)))
        keep_count = num_modules - skip_count
        apply_binary_gate_mask(model, set(gate_ranked_indices[:keep_count]))
        realized = skip_count / num_modules
        rows.append(evaluate_runner(
            MethodRunner("causalgate_topk_binary", model, tokenizer, saved_compute=realized),
            realized,
            ptb_batches,
        ))

    apply_all_modules_open(model)
    for calm_policy in CALM_POLICIES:
        for target_saved in args.target_saved:
            threshold, realized = calibrate_threshold(model, calibration_loader, calm_policy, target_saved)
            rows.append(evaluate_runner(
                MethodRunner(calm_policy, model, tokenizer, mode="calm", calm_policy=calm_policy, threshold=threshold, saved_compute=realized),
                realized,
                ptb_batches,
            ))

    cleanup_model(model)


def evaluate_mod_family(rows, ptb_batches, args, config):
    model, tokenizer = load_raw_tinyllama(config)
    model.eval()
    for policy in MOD_POLICIES:
        for target_saved in args.target_saved:
            enable_mod(model, policy, skip_ratio=target_saved)
            rows.append(evaluate_runner(
                MethodRunner(policy, model, tokenizer, mode="mod", saved_compute=target_saved),
                target_saved,
                ptb_batches,
            ))
            disable_mod(model)
    cleanup_model(model)


def evaluate_gateskip_family(rows, ptb_batches, args, config):
    if not Path(args.gateskip_checkpoint).exists():
        print(f"Skipping GateSkip: checkpoint not found at {args.gateskip_checkpoint}")
        return
    model, tokenizer = load_raw_tinyllama(config)

    class GateArgs:
        gate_init_bias = 5.0
        gate_init_std = 0.01

    add_gateskip_wrappers(model, GateArgs())
    load_gates(model, args.gateskip_checkpoint)
    model.eval()
    for target_saved in args.target_saved:
        model._gateskip_skip_ratio = target_saved
        rows.append(evaluate_runner(
            MethodRunner("gateskip_style", model, tokenizer, mode="gateskip", saved_compute=target_saved),
            target_saved,
            ptb_batches,
        ))
    cleanup_model(model)


def evaluate_adaskip_family(rows, ptb_batches, args, config):
    if not Path(args.adaskip_ranking).exists():
        print(f"Skipping AdaSkip: ranking not found at {args.adaskip_ranking}")
        return
    model, tokenizer = load_raw_tinyllama(config)
    add_adaskip_wrappers(model)
    model.eval()
    num_modules = len(model.model.layers) * 2
    for target_saved in args.target_saved:
        skip_modules = load_adaskip_skip_set(args.adaskip_ranking, target_saved, num_modules)
        model._adaskip_skip_modules = skip_modules
        realized = len(skip_modules) / num_modules
        rows.append(evaluate_runner(
            MethodRunner("adaskip_style", model, tokenizer, mode="adaskip", saved_compute=realized),
            realized,
            ptb_batches,
        ))
    cleanup_model(model)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods on Penn Treebank only.")
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--gateskip-checkpoint", default="outputs/gateskip_style_gates.pt")
    parser.add_argument("--adaskip-ranking", default="outputs/adaskip_style_io_similarity_ranking.csv")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--ptb-split", default="test")
    parser.add_argument("--ptb-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/ptb_all_methods.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    tokenizer = load_tokenizer(config)
    ptb_texts = load_ptb_texts(args.ptb_split, args.ptb_samples)
    ptb_batches = build_ptb_batches(tokenizer, ptb_texts, args.max_length)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    evaluate_gated_family(rows, ptb_batches, args, config)
    evaluate_mod_family(rows, ptb_batches, args, config)
    evaluate_gateskip_family(rows, ptb_batches, args, config)
    evaluate_adaskip_family(rows, ptb_batches, args, config)

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved Penn Treebank evaluation table to {args.output_csv}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
