import argparse
import csv
import gc
import math
import random
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

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
from scripts.eval_gateskip_style import (
    add_gateskip_wrappers,
    get_gateskip_saved,
    load_gates,
    reset_gateskip_stats,
    set_gateskip_batch_mask,
)
from scripts.eval_mod_baselines import (
    disable_mod,
    enable_mod,
    get_mod_saved,
    set_mod_context,
)
from utils.config import load_config


KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0
MODULES = ("attn", "mlp")
CALM_POLICIES = ("calm_softmax", "calm_hidden_state_saturation")
MOD_POLICIES = ("mod_random_router",)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model):
    return next(model.parameters()).device


def load_raw_tinyllama(config):
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


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_texts(tokenizer, texts, max_length):
    return [
        tokenizer(text, truncation=True, max_length=max_length, padding=False)
        for text in texts
        if len(text.strip()) > 0
    ]


def build_lm_batches(tokenizer, texts, max_length):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized = tokenize_texts(tokenizer, texts, max_length)
    return [collator([item]) for item in tokenized]


def load_c4_texts(num_samples):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for example in dataset:
        text = example.get("text", "")
        if len(text.strip()) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


def load_openbookqa(num_samples):
    dataset = load_dataset("openbookqa", "main", split="validation")
    return list(islice(dataset, num_samples))


def load_winogrande(num_samples):
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    return list(islice(dataset, num_samples))


def load_lambada(num_samples):
    try:
        dataset = load_dataset("lambada", split="validation", trust_remote_code=True)
    except Exception:
        dataset = load_dataset("lambada", "plain_text", split="validation", trust_remote_code=True)
    examples = []
    for example in dataset:
        text = example.get("text", "").strip()
        if len(text.split()) >= 2:
            examples.append(text)
        if len(examples) >= num_samples:
            break
    return examples


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


def nll_from_logits(logits, labels):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels != -100
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    return losses[mask].sum(), int(mask.sum().item())


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
            self.model._adaskip_current_attention_mask = batch.get("attention_mask").detach() if batch.get("attention_mask") is not None else None

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
        if self.mode == "mod":
            set_mod_context(self.model, attention_mask=attention_mask, labels=None)
        elif self.mode == "gateskip":
            set_gateskip_batch_mask(self.model, attention_mask)
        elif self.mode == "adaskip":
            self.model._adaskip_current_attention_mask = attention_mask.detach()
        if self.mode == "calm":
            confidences, layer_logits = get_calm_confidences_and_logits(self.model, batch, self.calm_policy)
            selected_logits, _ = select_calm_logits(confidences, layer_logits, self.threshold)
            return selected_logits[:, -1, :]
        return self.model(**batch, use_cache=False).logits[:, -1, :]


def evaluate_lm_ppl(runner, batches):
    runner.model.eval()
    runner.reset_stats()
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        batch = {k: v.to(model_device(runner.model)) for k, v in batch.items()}
        logits = runner.logits(batch)
        nll, count = nll_from_logits(logits, batch["labels"])
        total_nll += float(nll.item())
        total_tokens += count
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(mean_nll), runner.observed_saved()


def score_choice(runner, context, ending, max_length):
    tensors = make_choice_tensors(runner.tokenizer, context, ending, max_length, model_device(runner.model))
    logits = runner.logits(tensors)
    nll, count = nll_from_logits(logits, tensors["labels"])
    return float(nll.item()) / max(count, 1)


def evaluate_openbookqa(runner, examples, max_length):
    runner.model.eval()
    runner.reset_stats()
    correct = 0
    for example in examples:
        context = example["question_stem"]
        endings = example["choices"]["text"]
        label = example["choices"]["label"].index(example["answerKey"])
        scores = [score_choice(runner, context, ending, max_length) for ending in endings]
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
    return correct / max(len(examples), 1), correct, len(examples), runner.observed_saved()


def evaluate_winogrande(runner, examples, max_length):
    runner.model.eval()
    runner.reset_stats()
    correct = 0
    for example in examples:
        sentence = example["sentence"]
        prefix, suffix = sentence.split("_", 1)
        endings = [example["option1"] + suffix, example["option2"] + suffix]
        label = int(example["answer"]) - 1
        scores = [score_choice(runner, prefix, ending, max_length) for ending in endings]
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
    max_context_length = max(1, max_length - len(target_ids))
    context_ids = context_ids[-max_context_length:]
    return context_ids, target_ids


@torch.no_grad()
def evaluate_lambada_greedy(runner, texts, max_length):
    runner.model.eval()
    runner.reset_stats()
    correct = 0
    total = 0
    tokenizer = runner.tokenizer
    device = model_device(runner.model)
    for text in texts:
        context, target = split_lambada_text(text)
        if not context or not target:
            continue
        context_ids, target_ids = encode_lambada_context_and_target(tokenizer, context, target, max_length)
        if not context_ids or not target_ids:
            continue
        generated = []
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        for _ in target_ids:
            logits = runner.next_token_logits(input_ids, attention_mask)
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            attention_mask = torch.ones_like(input_ids)
        correct += int(generated == target_ids)
        total += 1
    return correct / max(total, 1), correct, total, runner.observed_saved()


def make_row(policy, target_saved, realized_saved, obqa, wino, lambada, c4_nll, c4_ppl):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "openbookqa_acc": obqa[0],
        "openbookqa_correct": obqa[1],
        "openbookqa_total": obqa[2],
        "winogrande_acc": wino[0],
        "winogrande_correct": wino[1],
        "winogrande_total": wino[2],
        "lambada_acc": lambada[0],
        "lambada_correct": lambada[1],
        "lambada_total": lambada[2],
        "c4_mean_nll": c4_nll,
        "c4_ppl": c4_ppl,
    }


def evaluate_runner(runner, target_saved, datasets, args):
    obqa = evaluate_openbookqa(runner, datasets["openbookqa"], args.max_length)
    wino = evaluate_winogrande(runner, datasets["winogrande"], args.max_length)
    lambada = evaluate_lambada_greedy(runner, datasets["lambada"], args.max_length)
    c4_nll, c4_ppl, c4_saved = evaluate_lm_ppl(runner, datasets["c4_batches"])
    dynamic_saved = [obqa[3], wino[3], lambada[3], c4_saved]
    realized = sum(dynamic_saved) / len(dynamic_saved) if runner.mode in {"mod", "gateskip"} else target_saved
    return make_row(runner.policy, target_saved, realized, obqa, wino, lambada, c4_nll, c4_ppl)


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nNew Dataset Evaluation")
    print("| policy | saved | OpenBookQA | WinoGrande | LAMBADA | C4 PPL |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['realized_saved']:.4f} | {row['openbookqa_acc']:.4f} | "
            f"{row['winogrande_acc']:.4f} | {row['lambada_acc']:.4f} | {row['c4_ppl']:.4f} |"
        )


def prepare_datasets(tokenizer, args):
    print("Loading evaluation datasets...")
    c4_texts = load_c4_texts(args.c4_samples)
    return {
        "openbookqa": load_openbookqa(args.openbookqa_samples),
        "winogrande": load_winogrande(args.winogrande_samples),
        "lambada": load_lambada(args.lambada_samples),
        "c4_batches": build_lm_batches(tokenizer, c4_texts, args.max_length),
    }


def evaluate_gated_family(rows, datasets, args, config):
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    num_modules = sum(1 for _ in model.model.layers for _ in (0, 1))
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)

    split = config["data"].get("eval_split", "test")
    from scripts.eval_calm_vs_causalgate import build_wikitext_loader
    calibration_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.calibration_samples)

    apply_all_modules_open(model)
    rows.append(evaluate_runner(MethodRunner("full_model", model, tokenizer, saved_compute=0.0), 0.0, datasets, args))

    for target_saved in args.target_saved:
        skip_count = max(0, min(num_modules - 1, round(num_modules * target_saved)))
        keep_count = num_modules - skip_count
        apply_binary_gate_mask(model, set(gate_ranked_indices[:keep_count]))
        rows.append(evaluate_runner(
            MethodRunner("causalgate_topk_binary", model, tokenizer, saved_compute=skip_count / num_modules),
            skip_count / num_modules,
            datasets,
            args,
        ))

    apply_all_modules_open(model)
    for calm_policy in CALM_POLICIES:
        for target_saved in args.target_saved:
            threshold, realized = calibrate_threshold(model, calibration_loader, calm_policy, target_saved)
            rows.append(evaluate_runner(
                MethodRunner(calm_policy, model, tokenizer, mode="calm", calm_policy=calm_policy, threshold=threshold, saved_compute=realized),
                realized,
                datasets,
                args,
            ))

    cleanup_model(model)


def evaluate_mod_family(rows, datasets, args, config):
    model, tokenizer = load_raw_tinyllama(config)
    model.eval()
    for policy in MOD_POLICIES:
        for target_saved in args.target_saved:
            enable_mod(model, policy, skip_ratio=target_saved)
            rows.append(evaluate_runner(
                MethodRunner(policy, model, tokenizer, mode="mod", saved_compute=target_saved),
                target_saved,
                datasets,
                args,
            ))
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
        rows.append(evaluate_runner(
            MethodRunner("gateskip_style", model, tokenizer, mode="gateskip", saved_compute=target_saved),
            target_saved,
            datasets,
            args,
        ))
    cleanup_model(model)


def load_adaskip_skip_set(path, target_saved, num_modules):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if "rank" in rows[0]:
        rows = sorted(rows, key=lambda row: int(row["rank"]))
    skip_count = max(0, min(num_modules, round(num_modules * target_saved)))
    return {int(row["module_idx"]) for row in rows[:skip_count]}


def evaluate_adaskip_family(rows, datasets, args, config):
    if not Path(args.adaskip_ranking).exists():
        print(f"Skipping AdaSkip: ranking not found at {args.adaskip_ranking}")
        return
    model, tokenizer = load_raw_tinyllama(config)
    add_adaskip_wrappers(model)
    model.eval()
    num_modules = len(list(model.model.layers)) * 2
    for target_saved in args.target_saved:
        model._adaskip_skip_modules = load_adaskip_skip_set(args.adaskip_ranking, target_saved, num_modules)
        rows.append(evaluate_runner(
            MethodRunner("adaskip_style", model, tokenizer, mode="adaskip", saved_compute=len(model._adaskip_skip_modules) / num_modules),
            len(model._adaskip_skip_modules) / num_modules,
            datasets,
            args,
        ))
    cleanup_model(model)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods on OpenBookQA, WinoGrande, LAMBADA, and C4.")
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--gateskip-checkpoint", default="outputs/gateskip_style_gates.pt")
    parser.add_argument("--adaskip-ranking", default="outputs/adaskip_style_io_similarity_ranking.csv")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--openbookqa-samples", type=int, default=500)
    parser.add_argument("--winogrande-samples", type=int, default=500)
    parser.add_argument("--lambada-samples", type=int, default=500)
    parser.add_argument("--c4-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/new_dataset_all_methods.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    tokenizer_for_data = load_tokenizer(config)
    datasets = prepare_datasets(tokenizer_for_data, args)
    del tokenizer_for_data
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
    print(f"\nSaved new dataset evaluation table to {args.output_csv}")


if __name__ == "__main__":
    main()
