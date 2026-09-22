"""Evaluate ShortGPT-style Block Influence at attention/MLP granularity.

This is an observational submodule adaptation, not an exact reproduction of
vanilla ShortGPT's full-block pruning method.
"""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scripts.eval_common import (
    build_lm_batches,
    build_wikitext_loader,
    load_c4_texts,
    load_dataset_examples,
    load_raw_tinyllama,
    make_choice_tensors,
    model_device,
    nll_from_logits,
    set_seed,
    write_csv,
)
from utils.config import load_config


POLICY_NAME = "block_influence_submodule"
DISPLAY_NAME = "Block Influence (Submodule)"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXPECTED_MODULES = 44
DEFAULT_BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40)
EXPECTED_REMOVALS = {0.05: 2, 0.10: 4, 0.20: 9, 0.30: 13, 0.40: 18}
CANONICAL_COUNTS = {
    "calibration_samples": 128,
    "wikitext_samples": 128,
    "c4_samples": 128,
    "hellaswag_samples": 256,
    "piqa_samples": 256,
    "commonsenseqa_samples": 256,
    "winogrande_samples": 500,
    "max_length": 512,
    "seed": 123,
}


class BlockInfluenceWrapper(nn.Module):
    """Wrap one residual branch for calibration and static branch zeroing."""

    def __init__(self, original_module, parent_model, module_idx, module_name, layer_idx, branch_type):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "parent_model", parent_model)
        self.module_idx = module_idx
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.branch_type = branch_type

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_module(hidden_states, *args, **kwargs)
        is_tuple = isinstance(outputs, tuple)
        branch_output = outputs[0] if is_tuple else outputs

        if getattr(self.parent_model, "_bi_calibrating", False):
            residual = self.parent_model._bi_residuals.get(self.layer_idx)
            if residual is None:
                raise RuntimeError(f"Missing residual-stream state for {self.module_name}")
            post_residual = residual + branch_output
            self._record_block_influence(residual, post_residual)
            if self.branch_type == "attn":
                self.parent_model._bi_residuals[self.layer_idx] = post_residual

        if self.module_idx in getattr(self.parent_model, "_bi_skip_modules", set()):
            branch_output = torch.zeros_like(branch_output)

        if is_tuple:
            return (branch_output,) + outputs[1:]
        return branch_output

    def _record_block_influence(self, pre_residual, post_residual):
        mask = getattr(self.parent_model, "_bi_current_attention_mask", None)
        if mask is None or tuple(mask.shape) != tuple(pre_residual.shape[:2]):
            valid_mask = torch.ones(pre_residual.shape[:2], dtype=torch.bool, device=pre_residual.device)
        else:
            valid_mask = mask.to(device=pre_residual.device, dtype=torch.bool)

        # ShortGPT-style influence adapted to one residual branch:
        # BI_m = 1 - mean_t cosine(h_pre_m,t, h_post_m,t).
        similarities = F.cosine_similarity(
            pre_residual.float(),
            post_residual.float(),
            dim=-1,
            eps=1e-8,
        )
        valid_influences = (1.0 - similarities)[valid_mask]
        if valid_influences.numel() == 0:
            return
        stats = self.parent_model._bi_stats[self.module_idx]
        stats["sum"] += float(valid_influences.sum().item())
        stats["count"] += int(valid_influences.numel())


def iter_block_influence_wrappers(model):
    for layer in model.model.layers:
        if isinstance(layer.self_attn, BlockInfluenceWrapper):
            yield layer.self_attn
        if isinstance(layer.mlp, BlockInfluenceWrapper):
            yield layer.mlp


def _capture_layer_residual(parent_model, layer_idx):
    def hook(_module, inputs):
        if getattr(parent_model, "_bi_calibrating", False):
            parent_model._bi_residuals[layer_idx] = inputs[0]

    return hook


def add_block_influence_wrappers(model):
    handles = []
    module_idx = 0
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(_capture_layer_residual(model, layer_idx)))
        layer.self_attn = BlockInfluenceWrapper(
            layer.self_attn, model, module_idx, f"L{layer_idx:02d}.attn", layer_idx, "attn"
        )
        module_idx += 1
        layer.mlp = BlockInfluenceWrapper(
            layer.mlp, model, module_idx, f"L{layer_idx:02d}.mlp", layer_idx, "mlp"
        )
        module_idx += 1

    model._bi_hook_handles = handles
    model._bi_calibrating = False
    model._bi_current_attention_mask = None
    model._bi_residuals = {}
    model._bi_skip_modules = set()
    model._bi_stats = {idx: {"sum": 0.0, "count": 0} for idx in range(module_idx)}
    return model


def set_batch_mask(model, attention_mask):
    model._bi_current_attention_mask = attention_mask.detach() if attention_mask is not None else None


def clear_forward_state(model):
    model._bi_residuals.clear()


@torch.no_grad()
def calibrate_block_influence(model, loader):
    model.eval()
    model._bi_skip_modules = set()
    model._bi_calibrating = True
    model._bi_stats = {
        wrapper.module_idx: {"sum": 0.0, "count": 0}
        for wrapper in iter_block_influence_wrappers(model)
    }

    for batch in tqdm(loader, desc="Calibrating Block Influence (Submodule)"):
        batch = {key: value.to(model_device(model)) for key, value in batch.items()}
        set_batch_mask(model, batch.get("attention_mask"))
        model(**batch, use_cache=False)
        clear_forward_state(model)

    model._bi_calibrating = False
    ranked = []
    for wrapper in iter_block_influence_wrappers(model):
        stats = model._bi_stats[wrapper.module_idx]
        ranked.append(
            {
                "module_idx": wrapper.module_idx,
                "module": wrapper.module_name,
                "block_influence": stats["sum"] / max(stats["count"], 1),
                "valid_tokens": stats["count"],
            }
        )
    ranked.sort(key=lambda row: (row["block_influence"], row["module_idx"]))
    return ranked


def select_skip_modules(ranked_modules, target_saved):
    skip_count = max(0, min(len(ranked_modules), round(len(ranked_modules) * target_saved)))
    selected = ranked_modules[:skip_count]
    return {row["module_idx"] for row in selected}, selected


@torch.no_grad()
def evaluate_ppl(model, batches):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        batch = {key: value.to(model_device(model)) for key, value in batch.items()}
        set_batch_mask(model, batch.get("attention_mask"))
        outputs = model(**batch, use_cache=False)
        nll, token_count = nll_from_logits(outputs.logits, batch["labels"])
        total_nll += float(nll.item())
        total_tokens += token_count
        clear_forward_state(model)
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(mean_nll)


def choice_fields(dataset_name, example):
    if dataset_name == "hellaswag":
        context = example.get("ctx", "") or f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
        return context, example["endings"], int(example["label"])
    if dataset_name == "piqa":
        return example["goal"], [example["sol1"], example["sol2"]], int(example["label"])
    if dataset_name == "csqa":
        return example["question"], example["choices"]["text"], example["choices"]["label"].index(example["answerKey"])
    if dataset_name == "winogrande":
        prefix, suffix = example["sentence"].split("_", 1)
        return prefix, [example["option1"] + suffix, example["option2"] + suffix], int(example["answer"]) - 1
    raise ValueError(dataset_name)


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, dataset_name, examples, max_length):
    model.eval()
    correct = 0
    for example in examples:
        context, endings, label = choice_fields(dataset_name, example)
        scores = []
        for ending in endings:
            tensors = make_choice_tensors(tokenizer, context, ending, max_length, model_device(model))
            set_batch_mask(model, tensors.get("attention_mask"))
            outputs = model(**tensors, use_cache=False)
            nll, token_count = nll_from_logits(outputs.logits, tensors["labels"])
            scores.append(float(nll.item()) / max(token_count, 1))
            clear_forward_state(model)
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
    total = len(examples)
    return correct / max(total, 1), correct, total


def evaluate_suite(model, tokenizer, datasets, max_length):
    w_nll, w_ppl = evaluate_ppl(model, datasets["wikitext"])
    c4_nll, c4_ppl = evaluate_ppl(model, datasets["c4"])
    hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", datasets["hellaswag"], max_length)
    piqa = evaluate_multiple_choice(model, tokenizer, "piqa", datasets["piqa"], max_length)
    csqa = evaluate_multiple_choice(model, tokenizer, "csqa", datasets["csqa"], max_length)
    wino = evaluate_multiple_choice(model, tokenizer, "winogrande", datasets["winogrande"], max_length)
    return w_nll, w_ppl, c4_nll, c4_ppl, hs, piqa, csqa, wino


def make_result_row(target_saved, realized_saved, selected, metrics, args):
    w_nll, w_ppl, c4_nll, c4_ppl, hs, piqa, csqa, wino = metrics
    return {
        "policy": "full_model" if target_saved == 0.0 else POLICY_NAME,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "num_modules": EXPECTED_MODULES,
        "num_skipped": len(selected),
        "skipped_modules": ";".join(row["module"] for row in selected),
        "wikitext_mean_nll": w_nll,
        "wikitext_ppl": w_ppl,
        "c4_mean_nll": c4_nll,
        "c4_ppl": c4_ppl,
        "hellaswag_acc": hs[0],
        "hellaswag_correct": hs[1],
        "hellaswag_total": hs[2],
        "piqa_acc": piqa[0],
        "piqa_correct": piqa[1],
        "piqa_total": piqa[2],
        "commonsenseqa_acc": csqa[0],
        "commonsenseqa_correct": csqa[1],
        "commonsenseqa_total": csqa[2],
        "winogrande_acc": wino[0],
        "winogrande_correct": wino[1],
        "winogrande_total": wino[2],
        "calibration_dataset": "wikitext/wikitext-2-raw-v1",
        "calibration_split": "test",
        "calibration_samples": args.calibration_samples,
        "max_length": args.max_length,
        "seed": args.seed,
    }


def write_ranking_csv(ranked_modules, output_path, args):
    rows = []
    for rank, row in enumerate(ranked_modules, start=1):
        rows.append(
            {
                "prune_rank": rank,
                **row,
                "calibration_dataset": "wikitext/wikitext-2-raw-v1",
                "calibration_split": "test",
                "calibration_samples": args.calibration_samples,
                "batch_size": 1,
                "max_length": args.max_length,
                "seed": args.seed,
                "score_definition": "1 - mean cosine(h_pre, h_post)",
            }
        )
    write_csv(rows, output_path)


def write_metadata(path, args, model_name, num_modules):
    metadata = {
        "method": DISPLAY_NAME,
        "method_note": "ShortGPT-style submodule adaptation; not exact vanilla ShortGPT",
        "model": model_name,
        "model_mode": "eval",
        "gates": "none",
        "use_cache": False,
        "num_modules": num_modules,
        "score": "BI_m = 1 - mean cosine(h_pre_m, h_post_m)",
        "cosine_dtype": "float32",
        "padding_tokens_excluded": True,
        "ranking_direction": "ascending; low Block Influence pruned first",
        "ranking_recomputed_after_pruning": False,
        "calibration": {
            "dataset": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "samples": args.calibration_samples,
            "filter": "len(text.strip()) > 20",
            "batch_size": 1,
            "max_length": args.max_length,
            "shuffle": False,
            "seed": args.seed,
        },
        "evaluation": {
            "wikitext": {"split": "test", "samples": args.wikitext_samples, "filter": "len(text.strip()) > 20"},
            "c4": {"split": "validation", "samples": args.c4_samples, "filter": "len(text.strip()) > 50"},
            "hellaswag": {"split": "validation", "samples": args.hellaswag_samples},
            "piqa": {"split": "validation", "samples": args.piqa_samples},
            "commonsenseqa": {"split": "validation", "samples": args.commonsenseqa_samples},
            "winogrande": {"config": "winogrande_xl", "split": "validation", "samples": args.winogrande_samples},
            "ppl": "token-weighted next-token NLL followed by exp(mean_nll)",
            "multiple_choice": "lowest mean continuation-token NLL",
        },
        "removal_counts": {str(target): EXPECTED_REMOVALS[target] for target in DEFAULT_BUDGETS},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def print_protocol(args):
    print("Block Influence (Submodule) protocol")
    print("Calibration:")
    print("dataset = wikitext/wikitext-2-raw-v1")
    print("split = test")
    print(f"samples = {args.calibration_samples}")
    print("batch size = 1")
    print(f"max length = {args.max_length}")
    print(f"seed = {args.seed}")
    print("model = raw dense TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("model mode = eval(); gates = none; use_cache = False")
    print("\nEvaluation:")
    print(f"WikiText-2 samples = {args.wikitext_samples}")
    print(f"C4 samples = {args.c4_samples}")
    print(f"HellaSwag samples = {args.hellaswag_samples}")
    print(f"PIQA samples = {args.piqa_samples}")
    print(f"CommonsenseQA samples = {args.commonsenseqa_samples}")
    print(f"WinoGrande samples = {args.winogrande_samples}")
    print("\nRemoval:")
    for target in DEFAULT_BUDGETS:
        print(f"{int(target * 100)}% = {EXPECTED_REMOVALS[target]} modules")
    print("\nConfirmed: settings match the specified Table 1 evaluation protocol and the static Activation-Norm calibration protocol.\n")


def validate_protocol(args, config):
    mismatches = []
    if config["model"]["name"] != MODEL_NAME:
        mismatches.append(f"model={config['model']['name']!r}, expected {MODEL_NAME!r}")
    for field, expected in CANONICAL_COUNTS.items():
        actual = getattr(args, field)
        if actual != expected:
            mismatches.append(f"{field}={actual}, expected {expected}")
    if tuple(args.target_saved) != DEFAULT_BUDGETS:
        mismatches.append(f"target_saved={tuple(args.target_saved)}, expected {DEFAULT_BUDGETS}")
    if mismatches:
        raise ValueError("Noncanonical Block Influence protocol:\n- " + "\n- ".join(mismatches))


def print_results(rows):
    print("\nBlock Influence (Submodule) Tradeoff")
    print("| policy | target | realized | skipped | WikiText PPL | C4 PPL | HellaSwag | PIQA | CSQA | WinoGrande |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{row['num_skipped']} | {row['wikitext_ppl']:.4f} | {row['c4_ppl']:.4f} | "
            f"{row['hellaswag_acc']:.4f} | {row['piqa_acc']:.4f} | "
            f"{row['commonsenseqa_acc']:.4f} | {row['winogrande_acc']:.4f} |"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Block Influence at TinyLlama attention/MLP submodule granularity.")
    parser.add_argument("--target-saved", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--c4-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--winogrande-samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ranking-csv", default="outputs/block_influence_submodule_ranking.csv")
    parser.add_argument("--output-csv", default="outputs/block_influence_submodule_tradeoff.csv")
    parser.add_argument("--metadata-json", default="outputs/block_influence_submodule_metadata.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("utils/gate.yaml")
    validate_protocol(args, config)
    print_protocol(args)
    set_seed(args.seed)
    config["data"]["max_length"] = args.max_length
    model, tokenizer = load_raw_tinyllama(config)
    add_block_influence_wrappers(model)
    model.eval()

    wrappers = list(iter_block_influence_wrappers(model))
    if len(wrappers) != EXPECTED_MODULES:
        raise RuntimeError(f"Expected {EXPECTED_MODULES} attention/MLP modules, found {len(wrappers)}")
    for target in args.target_saved:
        expected = EXPECTED_REMOVALS.get(round(target, 2))
        actual = round(len(wrappers) * target)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Removal-count mismatch at target {target}: expected {expected}, got {actual}")

    calibration_loader = build_wikitext_loader(
        config, tokenizer, split="test", num_samples=args.calibration_samples, batch_size=1, shuffle=False
    )
    wikitext_loader = list(
        build_wikitext_loader(
            config, tokenizer, split="test", num_samples=args.wikitext_samples, batch_size=1, shuffle=False
        )
    )
    datasets = {
        "wikitext": wikitext_loader,
        "c4": build_lm_batches(tokenizer, load_c4_texts(args.c4_samples), args.max_length),
        "hellaswag": load_dataset_examples("hellaswag", args.hellaswag_samples),
        "piqa": load_dataset_examples("piqa", args.piqa_samples),
        "csqa": load_dataset_examples("csqa", args.commonsenseqa_samples),
        "winogrande": load_dataset_examples("winogrande", args.winogrande_samples),
    }

    ranked_modules = calibrate_block_influence(model, calibration_loader)
    write_ranking_csv(ranked_modules, args.ranking_csv, args)
    write_metadata(args.metadata_json, args, config["model"]["name"], len(wrappers))
    print(f"Saved static ranking to {args.ranking_csv}")
    print(f"Saved provenance metadata to {args.metadata_json}")
    print("Lowest Block Influence modules (pruned first):")
    for row in ranked_modules[:10]:
        print(f"{row['module']} BI={row['block_influence']:.8f}")

    rows = []
    model._bi_skip_modules = set()
    rows.append(make_result_row(0.0, 0.0, [], evaluate_suite(model, tokenizer, datasets, args.max_length), args))
    for target_saved in args.target_saved:
        skip_set, selected = select_skip_modules(ranked_modules, target_saved)
        model._bi_skip_modules = skip_set
        metrics = evaluate_suite(model, tokenizer, datasets, args.max_length)
        rows.append(make_result_row(target_saved, len(skip_set) / len(wrappers), selected, metrics, args))

    write_csv(rows, args.output_csv)
    print_results(rows)
    print(f"\nSaved tradeoff results to {args.output_csv}")


if __name__ == "__main__":
    main()

