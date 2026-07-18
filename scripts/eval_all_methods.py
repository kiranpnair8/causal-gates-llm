"""Unified all-method evaluator for supplementary experiments.

This replaces the dataset-specific all-method scripts while keeping those
filenames as compatibility wrappers. It evaluates CausalGate, CALM, MoD,
GateSkip, and AdaSkip with shared scoring code.
"""

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.load_model import load_tinyllama_with_gates
from scripts.eval_adaskip_style import add_adaskip_wrappers
from scripts.eval_calm_vs_causalgate import (
    apply_all_modules_open,
    apply_binary_gate_mask,
    calibrate_threshold,
    get_calm_confidences_and_logits,
    get_gate_values,
    load_gate_checkpoint,
    select_calm_logits,
)
from scripts.eval_common import (
    build_lm_batches,
    build_wikitext_loader,
    cleanup_model,
    load_c4_texts,
    load_dataset_examples,
    load_lambada_texts,
    load_ptb_texts,
    load_raw_tinyllama,
    load_tokenizer,
    make_choice_tensors,
    model_device,
    nll_from_logits,
    set_seed,
    write_csv,
)
from scripts.eval_gateskip_style import (
    add_gateskip_wrappers,
    get_gateskip_saved,
    load_gates,
    reset_gateskip_stats,
    set_gateskip_batch_mask,
)
from scripts.eval_mod_baselines import disable_mod, enable_mod, get_mod_saved, set_mod_context
from utils.config import load_config


CALM_POLICIES = ("calm_softmax", "calm_hidden_state_saturation")
MOD_POLICIES = ("mod_random_router",)


class MethodRunner:
    def __init__(self, policy, model, tokenizer, mode="standard", calm_policy=None, threshold=None, saved_compute=0.0):
        self.policy = policy
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.calm_policy = calm_policy
        self.threshold = threshold
        self.saved_compute = saved_compute

    def prepare_batch(self, batch):
        if self.mode == "mod":
            set_mod_context(self.model, attention_mask=batch.get("attention_mask"), labels=batch.get("labels"))
        elif self.mode == "gateskip":
            set_gateskip_batch_mask(self.model, batch.get("attention_mask"))
        elif self.mode == "adaskip":
            mask = batch.get("attention_mask")
            self.model._adaskip_current_attention_mask = mask.detach() if mask is not None else None

    def reset_stats(self):
        if self.mode == "mod":
            self.model._mod_total_tokens = 0.0
            self.model._mod_skipped_tokens = 0.0
        elif self.mode == "gateskip":
            reset_gateskip_stats(self.model)

    def observed_saved(self):
        if self.mode == "mod":
            return get_mod_saved(self.model)
        if self.mode == "gateskip":
            return get_gateskip_saved(self.model)
        return self.saved_compute

    @torch.no_grad()
    def logits(self, batch):
        self.prepare_batch(batch)
        if self.mode == "calm":
            confidences, layer_logits = get_calm_confidences_and_logits(self.model, batch, self.calm_policy)
            selected_logits, _ = select_calm_logits(confidences, layer_logits, self.threshold)
            return selected_logits
        return self.model(**batch, use_cache=False).logits

    @torch.no_grad()
    def next_token_logits(self, input_ids, attention_mask):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        self.prepare_batch(batch)
        if self.mode == "calm":
            confidences, layer_logits = get_calm_confidences_and_logits(self.model, batch, self.calm_policy)
            selected_logits, _ = select_calm_logits(confidences, layer_logits, self.threshold)
            return selected_logits[:, -1, :]
        return self.model(**batch, use_cache=False).logits[:, -1, :]


@torch.no_grad()
def evaluate_lm_ppl(runner, batches):
    runner.model.eval()
    runner.reset_stats()
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        batch = {k: v.to(model_device(runner.model)) for k, v in batch.items()}
        nll, count = nll_from_logits(runner.logits(batch), batch["labels"])
        total_nll += float(nll.item())
        total_tokens += count
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, torch.exp(torch.tensor(mean_nll)).item(), runner.observed_saved()


def score_choice(runner, context, ending, max_length):
    tensors = make_choice_tensors(runner.tokenizer, context, ending, max_length, model_device(runner.model))
    nll, count = nll_from_logits(runner.logits(tensors), tensors["labels"])
    return float(nll.item()) / max(count, 1)


def arc_label(example):
    labels = example["choices"]["label"]
    answer = str(example["answerKey"])
    if answer in labels:
        return labels.index(answer)
    if answer.isdigit():
        return int(answer) - 1
    raise ValueError(f"Could not parse ARC answer key {answer!r}")


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
    if dataset_name == "openbookqa":
        return example["question_stem"], example["choices"]["text"], example["choices"]["label"].index(example["answerKey"])
    if dataset_name == "arcc":
        return example["question"], example["choices"]["text"], arc_label(example)
    raise ValueError(dataset_name)


def evaluate_choice_dataset(runner, dataset_name, examples, max_length):
    runner.model.eval()
    runner.reset_stats()
    correct = 0
    for example in examples:
        context, endings, label = choice_fields(dataset_name, example)
        scores = [score_choice(runner, context, ending, max_length) for ending in endings]
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
    return correct / max(len(examples), 1), correct, len(examples), runner.observed_saved()


def split_lambada_text(text):
    pieces = text.strip().rsplit(" ", 1)
    if len(pieces) != 2:
        return None, None
    return pieces[0], " " + pieces[1]


def encode_lambada_context_and_target(tokenizer, context, target, max_length):
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return [], []
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    context_ids = context_ids[-max(1, max_length - len(target_ids)) :]
    return context_ids, target_ids


@torch.no_grad()
def evaluate_lambada(runner, texts, max_length):
    runner.model.eval()
    runner.reset_stats()
    correct = 0
    total = 0
    device = model_device(runner.model)
    for text in texts:
        context, target = split_lambada_text(text)
        if not context or not target:
            continue
        context_ids, target_ids = encode_lambada_context_and_target(runner.tokenizer, context, target, max_length)
        if not context_ids or not target_ids:
            continue
        generated = []
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        for _ in target_ids:
            logits = runner.next_token_logits(input_ids, attention_mask)
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            attention_mask = torch.ones_like(input_ids)
        correct += int(generated == target_ids)
        total += 1
    return correct / max(total, 1), correct, total, runner.observed_saved()


def evaluate_runner(runner, target_saved, datasets, args):
    row = {"policy": runner.policy, "target_saved": target_saved}
    observed = []
    for name, payload in datasets.items():
        if name in {"wikitext", "c4", "ptb"}:
            mean_nll, ppl, saved = evaluate_lm_ppl(runner, payload)
            row[f"{name}_mean_nll"] = mean_nll
            row[f"{name}_ppl"] = ppl
            observed.append(saved)
        elif name == "lambada":
            acc, correct, total, saved = evaluate_lambada(runner, payload, args.max_length)
            row["lambada_acc"] = acc
            row["lambada_correct"] = correct
            row["lambada_total"] = total
            observed.append(saved)
        else:
            acc, correct, total, saved = evaluate_choice_dataset(runner, name, payload, args.max_length)
            column = "arc_challenge" if name == "arcc" else name
            row[f"{column}_acc"] = acc
            row[f"{column}_correct"] = correct
            row[f"{column}_total"] = total
            observed.append(saved)
    row["realized_saved"] = sum(observed) / len(observed) if runner.mode in {"mod", "gateskip"} and observed else target_saved
    return row


def prepare_datasets(tokenizer, config, args):
    datasets = {}
    split = config["data"].get("eval_split", "test")
    if "wikitext" in args.datasets:
        loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.wikitext_samples)
        datasets["wikitext"] = list(loader)
    if "c4" in args.datasets:
        datasets["c4"] = build_lm_batches(tokenizer, load_c4_texts(args.c4_samples), args.max_length)
    if "ptb" in args.datasets:
        datasets["ptb"] = build_lm_batches(tokenizer, load_ptb_texts(args.ptb_split, args.ptb_samples), args.max_length)
    if "lambada" in args.datasets:
        datasets["lambada"] = load_lambada_texts(args.lambada_samples)
    for name in ("hellaswag", "piqa", "csqa", "winogrande", "openbookqa", "arcc"):
        if name in args.datasets:
            datasets[name] = load_dataset_examples(name, getattr(args, f"{name}_samples"))
    return datasets


def print_summary(rows):
    print("\nAll-Methods Evaluation")
    for row in rows:
        metrics = []
        for key, value in row.items():
            if key.endswith("_ppl") or key.endswith("_acc"):
                metrics.append(f"{key}={value:.4f}")
        print(f"{row['policy']} saved={row['realized_saved']:.4f} " + " ".join(metrics))


def load_adaskip_skip_set(path, target_saved, num_modules):
    import csv

    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if rows and "rank" in rows[0]:
        rows = sorted(rows, key=lambda row: int(row["rank"]))
    skip_count = max(0, min(num_modules, round(num_modules * target_saved)))
    return {int(row["module_idx"]) for row in rows[:skip_count]}


def evaluate_gated_family(rows, datasets, args, config):
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()
    num_modules = len(model.model.layers) * 2
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)
    split = config["data"].get("eval_split", "test")
    calibration_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.calibration_samples)

    apply_all_modules_open(model)
    rows.append(evaluate_runner(MethodRunner("full_model", model, tokenizer, saved_compute=0.0), 0.0, datasets, args))

    for target_saved in args.target_saved:
        skip_count = max(0, min(num_modules - 1, round(num_modules * target_saved)))
        keep_count = num_modules - skip_count
        apply_binary_gate_mask(model, set(gate_ranked_indices[:keep_count]))
        realized = skip_count / num_modules
        rows.append(evaluate_runner(MethodRunner("causalgate_topk_binary", model, tokenizer, saved_compute=realized), realized, datasets, args))

    apply_all_modules_open(model)
    for policy in CALM_POLICIES:
        for target_saved in args.target_saved:
            threshold, realized = calibrate_threshold(model, calibration_loader, policy, target_saved)
            rows.append(evaluate_runner(MethodRunner(policy, model, tokenizer, mode="calm", calm_policy=policy, threshold=threshold, saved_compute=realized), realized, datasets, args))
    cleanup_model(model)


def evaluate_mod_family(rows, datasets, args, config):
    model, tokenizer = load_raw_tinyllama(config)
    model.eval()
    for policy in MOD_POLICIES:
        for target_saved in args.target_saved:
            enable_mod(model, policy, skip_ratio=target_saved)
            rows.append(evaluate_runner(MethodRunner(policy, model, tokenizer, mode="mod", saved_compute=target_saved), target_saved, datasets, args))
            disable_mod(model)
    cleanup_model(model)


def evaluate_gateskip_family(rows, datasets, args, config):
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
        rows.append(evaluate_runner(MethodRunner("gateskip_style", model, tokenizer, mode="gateskip", saved_compute=target_saved), target_saved, datasets, args))
    cleanup_model(model)


def evaluate_adaskip_family(rows, datasets, args, config):
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
        rows.append(evaluate_runner(MethodRunner("adaskip_style", model, tokenizer, mode="adaskip", saved_compute=realized), realized, datasets, args))
    cleanup_model(model)


def build_parser(description, default_datasets, default_output_csv):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", nargs="+", default=list(default_datasets))
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--gateskip-checkpoint", default="outputs/gateskip_style_gates.pt")
    parser.add_argument("--adaskip-ranking", default="outputs/adaskip_style_io_similarity_ranking.csv")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--c4-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--csqa-samples", type=int, default=256)
    parser.add_argument("--winogrande-samples", type=int, default=500)
    parser.add_argument("--openbookqa-samples", type=int, default=500)
    parser.add_argument("--arcc-samples", type=int, default=500)
    parser.add_argument("--lambada-samples", type=int, default=500)
    parser.add_argument("--ptb-split", default="test")
    parser.add_argument("--ptb-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default=default_output_csv)
    return parser


def run(default_datasets=("wikitext", "c4", "hellaswag", "piqa", "csqa", "winogrande"), default_output_csv="outputs/all_methods.csv", description="Evaluate all methods on selected datasets."):
    args = build_parser(description, default_datasets, default_output_csv).parse_args()
    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    tokenizer = load_tokenizer(config)
    datasets = prepare_datasets(tokenizer, config, args)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    evaluate_gated_family(rows, datasets, args, config)
    evaluate_mod_family(rows, datasets, args, config)
    evaluate_gateskip_family(rows, datasets, args, config)
    evaluate_adaskip_family(rows, datasets, args, config)
    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved all-method evaluation table to {args.output_csv}")


def main():
    run()


if __name__ == "__main__":
    main()
