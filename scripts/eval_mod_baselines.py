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
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from utils.config import load_config


MOD_POLICIES = ("mod_magnitude_router", "mod_random_router")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def model_device(model):
    return next(model.parameters()).device


def build_wikitext_loader(config, tokenizer, split, num_samples):
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, config["data"]["max_length"]),
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=1, shuffle=False, collate_fn=collator)


def prediction_mask_from_labels(labels):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    return mask


def set_mod_context(model, attention_mask=None, labels=None):
    model._mod_attention_mask = attention_mask.detach() if attention_mask is not None else None
    model._mod_protected_mask = prediction_mask_from_labels(labels).detach() if labels is not None else None


def disable_mod(model):
    if not hasattr(model, "_mod_original_forwards"):
        return
    for layer, forward in zip(model.model.layers, model._mod_original_forwards):
        layer.forward = forward
    model._mod_enabled = False


def reset_mod_stats(model):
    model._mod_total_tokens = 0.0
    model._mod_skipped_tokens = 0.0


def get_mod_saved(model):
    total = getattr(model, "_mod_total_tokens", 0.0)
    skipped = getattr(model, "_mod_skipped_tokens", 0.0)
    return skipped / max(total, 1.0)


def make_skip_mask(model, hidden_states, attention_mask, router, skip_ratio):
    batch, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    if attention_mask is not None and attention_mask.dim() == 2:
        valid = attention_mask.to(device=device, dtype=torch.bool)
    else:
        valid = torch.ones(batch, seq_len, device=device, dtype=torch.bool)

    protected = getattr(model, "_mod_protected_mask", None)
    if protected is not None and tuple(protected.shape) == (batch, seq_len):
        protected = protected.to(device=device, dtype=torch.bool) & valid
    else:
        protected = torch.zeros_like(valid)

    if router == "mod_magnitude_router":
        scores = hidden_states.float().norm(dim=-1)
    elif router == "mod_random_router":
        scores = torch.rand(batch, seq_len, device=device)
    else:
        raise ValueError(f"Unknown MoD router: {router}")

    skip_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.bool)
    for row in range(batch):
        valid_idx = valid[row].nonzero(as_tuple=False).squeeze(-1)
        valid_count = int(valid_idx.numel())
        if valid_count == 0:
            continue

        skip_count = int(round(valid_count * skip_ratio))
        if skip_count <= 0:
            continue
        skip_count = min(skip_count, valid_count - 1) if valid_count > 1 else 0
        if skip_count <= 0:
            continue

        unprotected_idx = (valid[row] & ~protected[row]).nonzero(as_tuple=False).squeeze(-1)
        protected_idx = (valid[row] & protected[row]).nonzero(as_tuple=False).squeeze(-1)

        chosen = []
        if int(unprotected_idx.numel()) > 0:
            k = min(skip_count, int(unprotected_idx.numel()))
            local_scores = scores[row, unprotected_idx]
            chosen.append(unprotected_idx[torch.topk(local_scores, k=k, largest=False).indices])

        remaining = skip_count - sum(int(part.numel()) for part in chosen)
        if remaining > 0 and int(protected_idx.numel()) > 1:
            k = min(remaining, int(protected_idx.numel()) - 1)
            local_scores = scores[row, protected_idx]
            chosen.append(protected_idx[torch.topk(local_scores, k=k, largest=False).indices])

        if chosen:
            skip_mask[row, torch.cat(chosen)] = True

    return skip_mask, valid


def enable_mod(model, router, skip_ratio):
    disable_mod(model)
    if not hasattr(model, "_mod_original_forwards"):
        model._mod_original_forwards = [layer.forward for layer in model.model.layers]
    reset_mod_stats(model)
    model._mod_enabled = True
    model._mod_router = router
    model._mod_skip_ratio = skip_ratio

    for layer_idx, layer in enumerate(model.model.layers):
        original_forward = model._mod_original_forwards[layer_idx]

        def mod_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            _original_forward=original_forward,
            **kwargs,
        ):
            outputs = _original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            full_hidden = outputs[0]
            mask_source = getattr(model, "_mod_attention_mask", None)
            skip_mask, valid = make_skip_mask(
                model,
                hidden_states,
                mask_source,
                router=getattr(model, "_mod_router"),
                skip_ratio=getattr(model, "_mod_skip_ratio"),
            )
            mixed_hidden = torch.where(skip_mask.unsqueeze(-1), hidden_states, full_hidden)

            model._mod_skipped_tokens += float(skip_mask.sum().item())
            model._mod_total_tokens += float(valid.sum().item())
            return (mixed_hidden,) + outputs[1:]

        layer.forward = mod_forward


@torch.no_grad()
def evaluate_wikitext_ppl(model, loader):
    model.eval()
    reset_mod_stats(model)
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(model_device(model)) for k, v in batch.items()}
        set_mod_context(model, attention_mask=batch.get("attention_mask"), labels=batch.get("labels"))
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
    mean_nll = total_nll / max(total_tokens, 1)
    return mean_nll, math.exp(mean_nll), get_mod_saved(model)


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
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels.unsqueeze(0).to(device)}


@torch.no_grad()
def score_choice(model, tensors):
    set_mod_context(model, attention_mask=tensors.get("attention_mask"), labels=tensors.get("labels"))
    outputs = model(**tensors, use_cache=False)
    token_count = int((tensors["labels"][:, 1:] != -100).sum().item())
    return float(outputs.loss.item()) * token_count / max(token_count, 1)


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, dataset_name, num_samples, max_length):
    if dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    elif dataset_name == "commonsense_qa":
        dataset = load_dataset("commonsense_qa", split="validation")
    else:
        raise ValueError(dataset_name)

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    reset_mod_stats(model)
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
    return correct / max(total, 1), correct, total, get_mod_saved(model)


def evaluate_suite(model, tokenizer, wikitext_loader, args):
    w_nll, w_ppl, w_saved = evaluate_wikitext_ppl(model, wikitext_loader)
    hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", args.hellaswag_samples, args.max_length)
    piqa = evaluate_multiple_choice(model, tokenizer, "piqa", args.piqa_samples, args.max_length)
    csqa = evaluate_multiple_choice(model, tokenizer, "commonsense_qa", args.commonsenseqa_samples, args.max_length)
    realized_saved = (w_saved + hs[3] + piqa[3] + csqa[3]) / 4.0
    return realized_saved, w_nll, w_ppl, hs, piqa, csqa


def make_row(policy, target_saved, realized_saved, wikitext_nll, wikitext_ppl, hs, piqa, csqa):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
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


def print_summary(rows):
    print("\nMoD Router Baselines")
    print("| policy | target_saved | realized_saved | WikiText PPL | HellaSwag | PIQA | CSQA |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} | "
            f"{row['piqa_acc']:.4f} | {row['commonsenseqa_acc']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate standalone inference-only MoD router baselines.")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/mod_router_baselines.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama(config)
    model.eval()

    split = config["data"].get("eval_split", "test")
    wikitext_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.wikitext_samples)

    rows = []
    for policy in MOD_POLICIES:
        for target_saved in args.target_saved:
            enable_mod(model, policy, skip_ratio=target_saved)
            realized, w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
            rows.append(make_row(policy, target_saved, realized, w_nll, w_ppl, hs, piqa, csqa))
            disable_mod(model)

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved MoD baseline table to {args.output_csv}")
    print("Note: this is an inference-only quality/compute-accounting simulation; the full layer is still computed before token bypass is applied.")


if __name__ == "__main__":
    main()
