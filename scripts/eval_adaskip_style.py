import argparse
import csv
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model):
    return next(model.parameters()).device


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)


def load_tinyllama(config):
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = config.get("model", {}).get("torch_dtype", "float16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.get("system", {}).get("device_map", "auto"),
    )
    model.config.use_cache = False
    return model, tokenizer


class AdaSkipSublayerWrapper(nn.Module):
    def __init__(self, original_module, parent_model, module_idx, module_name):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "parent_model", parent_model)
        self.module_idx = module_idx
        self.module_name = module_name

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_module(hidden_states, *args, **kwargs)
        is_tuple = isinstance(outputs, tuple)
        branch_output = outputs[0] if is_tuple else outputs

        if getattr(self.parent_model, "_adaskip_calibrating", False):
            self._record_similarity(hidden_states, branch_output)

        if self.module_idx in getattr(self.parent_model, "_adaskip_skip_modules", set()):
            branch_output = torch.zeros_like(branch_output)

        if is_tuple:
            return (branch_output,) + outputs[1:]
        return branch_output

    def _valid_mask(self, shape, device):
        current_mask = getattr(self.parent_model, "_adaskip_current_attention_mask", None)
        if current_mask is not None and tuple(current_mask.shape) == tuple(shape):
            return current_mask.to(device=device, dtype=torch.bool)
        return torch.ones(shape, device=device, dtype=torch.bool)

    def _record_similarity(self, input_hidden, output_hidden):
        valid_mask = self._valid_mask(input_hidden.shape[:2], input_hidden.device)
        similarity = F.cosine_similarity(input_hidden.float(), output_hidden.float(), dim=-1)
        valid_similarity = similarity[valid_mask]
        if valid_similarity.numel() == 0:
            return
        stats = self.parent_model._adaskip_similarity_stats[self.module_idx]
        stats["sum"] += float(valid_similarity.sum().item())
        stats["count"] += int(valid_similarity.numel())


def iter_adaskip_wrappers(model):
    for layer in model.model.layers:
        if isinstance(layer.self_attn, AdaSkipSublayerWrapper):
            yield layer.self_attn
        if isinstance(layer.mlp, AdaSkipSublayerWrapper):
            yield layer.mlp


def add_adaskip_wrappers(model):
    module_idx = 0
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn = AdaSkipSublayerWrapper(
            layer.self_attn,
            model,
            module_idx,
            f"L{layer_idx:02d}.attn",
        )
        module_idx += 1
        layer.mlp = AdaSkipSublayerWrapper(
            layer.mlp,
            model,
            module_idx,
            f"L{layer_idx:02d}.mlp",
        )
        module_idx += 1
    reset_adaskip_state(model)
    return model


def reset_adaskip_state(model):
    model._adaskip_calibrating = False
    model._adaskip_current_attention_mask = None
    model._adaskip_skip_modules = set()
    model._adaskip_similarity_stats = {idx: {"sum": 0.0, "count": 0} for idx, _ in enumerate(iter_adaskip_wrappers(model))}


def set_adaskip_batch_mask(model, attention_mask):
    model._adaskip_current_attention_mask = attention_mask.detach() if attention_mask is not None else None


def build_wikitext_loader(config, tokenizer, split, num_samples, batch_size, shuffle):
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, config["data"]["max_length"]),
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


@torch.no_grad()
def calibrate_io_similarity(model, loader):
    model.eval()
    model._adaskip_skip_modules = set()
    model._adaskip_calibrating = True
    model._adaskip_similarity_stats = {idx: {"sum": 0.0, "count": 0} for idx, _ in enumerate(iter_adaskip_wrappers(model))}

    for batch in tqdm(loader, desc="Calibrating AdaSkip IO similarity"):
        batch = {k: v.to(model_device(model)) for k, v in batch.items()}
        set_adaskip_batch_mask(model, batch.get("attention_mask"))
        model(**batch, use_cache=False)

    model._adaskip_calibrating = False
    ranked = []
    for wrapper in iter_adaskip_wrappers(model):
        stats = model._adaskip_similarity_stats[wrapper.module_idx]
        mean_similarity = stats["sum"] / max(stats["count"], 1)
        ranked.append({
            "module_idx": wrapper.module_idx,
            "module_name": wrapper.module_name,
            "io_similarity": mean_similarity,
            "count": stats["count"],
        })
    ranked.sort(key=lambda row: row["io_similarity"], reverse=True)
    return ranked


def select_skip_modules(ranked_modules, target_saved):
    num_modules = len(ranked_modules)
    skip_count = max(0, min(num_modules, round(num_modules * target_saved)))
    selected = ranked_modules[:skip_count]
    return {row["module_idx"] for row in selected}, selected


@torch.no_grad()
def evaluate_wikitext_ppl(model, loader):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(model_device(model)) for k, v in batch.items()}
        set_adaskip_batch_mask(model, batch.get("attention_mask"))
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(mean_nll)


def get_hellaswag_context(example):
    ctx = example.get("ctx", "")
    return ctx if ctx else f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()


def make_choice_tensors(tokenizer, context, ending, max_length, device):
    context = context.strip()
    ending = ending.strip()
    if not ending.startswith(" "):
        ending = " " + ending
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(context + ending, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    context_len = min(len(context_ids), max(0, len(full_ids) - 1))
    labels = torch.tensor(full_ids, dtype=torch.long)
    labels[:context_len] = -100
    if int((labels != -100).sum().item()) == 0 and len(labels) > 0:
        labels[-1] = full_ids[-1]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels.unsqueeze(0).to(device),
    }


@torch.no_grad()
def score_choice(model, tensors):
    set_adaskip_batch_mask(model, tensors.get("attention_mask"))
    outputs = model(**tensors, use_cache=False)
    token_count = int((tensors["labels"][:, 1:] != -100).sum().item())
    return float(outputs.loss.item()) * token_count / max(token_count, 1)


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, dataset_name, num_samples, max_length):
    model.eval()
    if dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    elif dataset_name == "commonsense_qa":
        dataset = load_dataset("commonsense_qa", split="validation")
    else:
        raise ValueError(dataset_name)

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    correct = 0
    total = 0

    for example in dataset:
        if dataset_name == "hellaswag":
            context = get_hellaswag_context(example)
            endings = example["endings"]
            label = int(example["label"])
        elif dataset_name == "piqa":
            context = example["goal"]
            endings = [example["sol1"], example["sol2"]]
            label = int(example["label"])
        else:
            context = example["question"]
            endings = example["choices"]["text"]
            label = example["choices"]["label"].index(example["answerKey"])

        scores = []
        for ending in endings:
            tensors = make_choice_tensors(tokenizer, context, ending, max_length, model_device(model))
            scores.append(score_choice(model, tensors))
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
        total += 1

    return correct / max(total, 1), correct, total


def evaluate_suite(model, tokenizer, wikitext_loader, args):
    w_nll, w_ppl = evaluate_wikitext_ppl(model, wikitext_loader)
    hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", args.hellaswag_samples, args.max_length)
    piqa = evaluate_multiple_choice(model, tokenizer, "piqa", args.piqa_samples, args.max_length)
    csqa = evaluate_multiple_choice(model, tokenizer, "commonsense_qa", args.commonsenseqa_samples, args.max_length)
    return w_nll, w_ppl, hs, piqa, csqa


def make_row(policy, target_saved, realized_saved, skipped_modules, skipped_names, wikitext_nll, wikitext_ppl, hs, piqa, csqa):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "skipped_modules": skipped_modules,
        "skipped_module_names": ";".join(skipped_names),
        "wikitext_mean_nll": wikitext_nll,
        "wikitext_ppl": wikitext_ppl,
        "hellaswag_acc": hs[0],
        "hellaswag_correct": hs[1],
        "hellaswag_total": hs[2],
        "piqa_acc": piqa[0],
        "piqa_correct": piqa[1],
        "piqa_total": piqa[2],
        "commonsenseqa_acc": csqa[0],
        "commonsenseqa_correct": csqa[1],
        "commonsenseqa_total": csqa[2],
    }


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ranking_csv(ranked_modules, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "module_idx", "module_name", "io_similarity", "count"])
        writer.writeheader()
        for rank, row in enumerate(ranked_modules, start=1):
            writer.writerow({"rank": rank, **row})


def print_summary(rows):
    print("\nAdaSkip-Style Tradeoff")
    print("| policy | target_saved | realized_saved | skipped | WikiText PPL | HellaSwag | PIQA | CSQA |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{row['skipped_modules']} | {row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} | "
            f"{row['piqa_acc']:.4f} | {row['commonsenseqa_acc']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaSkip-style IO-similarity global sublayer skipping.")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/adaskip_style_tradeoff.csv")
    parser.add_argument("--ranking-csv", default="outputs/adaskip_style_io_similarity_ranking.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama(config)
    add_adaskip_wrappers(model)

    split = config["data"].get("eval_split", "test")
    calibration_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=split,
        num_samples=args.calibration_samples,
        batch_size=1,
        shuffle=False,
    )
    wikitext_loader = build_wikitext_loader(
        config,
        tokenizer,
        split=split,
        num_samples=args.wikitext_samples,
        batch_size=1,
        shuffle=False,
    )

    ranked_modules = calibrate_io_similarity(model, calibration_loader)
    write_ranking_csv(ranked_modules, args.ranking_csv)
    print(f"Saved AdaSkip IO-similarity ranking to {args.ranking_csv}")
    print("Top AdaSkip skip candidates:")
    for row in ranked_modules[:10]:
        print(f"{row['module_name']} io_similarity={row['io_similarity']:.6f}")

    rows = []
    model._adaskip_skip_modules = set()
    w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
    rows.append(make_row("full_model", 0.0, 0.0, 0, [], w_nll, w_ppl, hs, piqa, csqa))

    num_modules = len(ranked_modules)
    for target_saved in args.target_saved:
        skip_set, selected = select_skip_modules(ranked_modules, target_saved)
        model._adaskip_skip_modules = skip_set
        skipped_names = [row["module_name"] for row in selected]
        realized_saved = len(skip_set) / max(num_modules, 1)
        w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
        rows.append(make_row(
            "adaskip_style",
            target_saved,
            realized_saved,
            len(skip_set),
            skipped_names,
            w_nll,
            w_ppl,
            hs,
            piqa,
            csqa,
        ))

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved AdaSkip-style tradeoff table to {args.output_csv}")


if __name__ == "__main__":
    main()
