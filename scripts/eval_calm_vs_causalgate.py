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
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from models.load_model import load_tinyllama_with_gates
from utils.config import load_config


KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0
CALM_POLICIES = ("calm_softmax", "calm_hidden_state_saturation")
MOD_POLICIES = ("mod_magnitude_router", "mod_random_router")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)


def iter_gate_modules(model):
    for layer in model.model.layers:
        yield layer.attn_gate
        yield layer.mlp_gate


def get_gate_values(model):
    return [float(g.gate_values_scalar().detach().float().cpu()) for g in iter_gate_modules(model)]


def apply_binary_gate_mask(model, kept_indices):
    kept_indices = set(kept_indices)
    for idx, gate in enumerate(iter_gate_modules(model)):
        gate.gate_logit.data.fill_(KEEP_LOGIT if idx in kept_indices else SKIP_LOGIT)


def apply_all_modules_open(model):
    apply_binary_gate_mask(model, set(range(sum(1 for _ in iter_gate_modules(model)))))


def load_gate_checkpoint(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")

    gate_state = {
        k: v for k, v in state_dict.items()
        if k.endswith("attn_gate.gate_logit") or k.endswith("mlp_gate.gate_logit")
    }
    if not gate_state:
        raise ValueError(f"No gate_logit parameters found in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(gate_state, strict=False)
    print(f"Loaded {len(gate_state)} trained gate parameters from {checkpoint_dir}")
    print(f"Ignored missing backbone keys: {len(missing)}; unexpected keys: {len(unexpected)}")


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


@torch.no_grad()
def evaluate_standard_ppl(model, loader):
    total_nll = 0.0
    total_tokens = 0
    reset_mod_stats(model)
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"]
        token_count = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
    return total_nll / total_tokens, math.exp(total_nll / total_tokens), get_mod_saved(model)


def get_lm_head_logits(model, hidden_states):
    return model.lm_head(model.model.norm(hidden_states))


@torch.no_grad()
def get_calm_confidences_and_logits(model, batch, policy):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    num_layers = len(model.model.layers)
    layer_logits = []
    layer_confidences = []

    for layer_idx in range(1, num_layers + 1):
        h_l = hidden_states[layer_idx]
        logits_l = get_lm_head_logits(model, h_l)

        if policy == "calm_softmax":
            probs = torch.softmax(logits_l.float(), dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            confidence = top2[..., 0] - top2[..., 1]
        elif policy == "calm_hidden_state_saturation":
            h_prev = hidden_states[layer_idx - 1]
            confidence = F.cosine_similarity(
                F.normalize(h_l.float(), dim=-1),
                F.normalize(h_prev.float(), dim=-1),
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown CALM policy: {policy}")

        layer_logits.append(logits_l)
        layer_confidences.append(confidence)

    return torch.stack(layer_confidences, dim=0), torch.stack(layer_logits, dim=0)


def select_calm_logits(confidences, layer_logits, threshold):
    num_layers = confidences.shape[0]
    exits = confidences >= threshold
    any_exit = exits.any(dim=0)
    first_exit = exits.float().argmax(dim=0).long()
    final_layer = torch.full_like(first_exit, num_layers - 1)
    selected_layers = torch.where(any_exit, first_exit, final_layer)
    selected = torch.gather(
        layer_logits,
        dim=0,
        index=selected_layers.unsqueeze(0).unsqueeze(-1).expand(1, *selected_layers.shape, layer_logits.shape[-1]),
    ).squeeze(0)
    return selected, selected_layers


def shifted_token_mask(labels):
    return labels[:, 1:] != -100


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


@torch.no_grad()
def evaluate_calm_ppl(model, loader, policy, threshold):
    total_nll = 0.0
    total_tokens = 0
    total_saved = 0.0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"]
        confidences, layer_logits = get_calm_confidences_and_logits(model, batch, policy)
        selected_logits, selected_layers = select_calm_logits(confidences, layer_logits, threshold)
        nll, token_count = nll_from_logits(selected_logits, labels)
        pred_layers = selected_layers[:, :-1]
        mask = shifted_token_mask(labels)
        skipped = (len(model.model.layers) - (pred_layers[mask].float() + 1.0)) / len(model.model.layers)
        total_nll += float(nll.item())
        total_tokens += token_count
        total_saved += float(skipped.sum().item())
    return total_nll / total_tokens, math.exp(total_nll / total_tokens), total_saved / total_tokens


@torch.no_grad()
def estimate_calm_saved_compute(model, loader, policy, threshold):
    total_saved = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"]
        confidences, layer_logits = get_calm_confidences_and_logits(model, batch, policy)
        _, selected_layers = select_calm_logits(confidences, layer_logits, threshold)
        pred_layers = selected_layers[:, :-1]
        mask = shifted_token_mask(labels)
        skipped = (len(model.model.layers) - (pred_layers[mask].float() + 1.0)) / len(model.model.layers)
        total_saved += float(skipped.sum().item())
        total_tokens += int(mask.sum().item())
    return total_saved / total_tokens


def make_threshold_candidates(conf_values, policy):
    values = torch.cat(conf_values).float().cpu()
    candidates = torch.quantile(values, torch.linspace(0.0, 1.0, steps=101)).unique().tolist()
    candidates.extend([0.0, 1.0] if policy == "calm_softmax" else [-1.0, 1.0])
    return sorted(set(float(x) for x in candidates))


@torch.no_grad()
def collect_calibration_confidences(model, loader, policy):
    conf_values = []
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"]
        confidences, _ = get_calm_confidences_and_logits(model, batch, policy)
        mask = shifted_token_mask(labels)
        conf_values.append(confidences[:, :, :-1][:, mask].flatten().detach().cpu())
    return conf_values


def calibrate_threshold(model, loader, policy, target_saved):
    conf_values = collect_calibration_confidences(model, loader, policy)
    candidates = make_threshold_candidates(conf_values, policy)
    best = None
    for threshold in candidates:
        saved = estimate_calm_saved_compute(model, loader, policy, threshold)
        error = abs(saved - target_saved)
        if best is None or error < best[0]:
            best = (error, threshold, saved)
    _, threshold, realized_saved = best
    return threshold, realized_saved


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
    if not getattr(model, "_mod_enabled", False):
        return 0.0
    total = getattr(model, "_mod_total_tokens", 0.0)
    skipped = getattr(model, "_mod_skipped_tokens", 0.0)
    return skipped / max(total, 1.0)


def enable_mod(model, router, skip_ratio):
    disable_mod(model)
    if not hasattr(model, "_mod_original_forwards"):
        model._mod_original_forwards = [layer.forward for layer in model.model.layers]
    reset_mod_stats(model)
    model._mod_enabled = True

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
            batch, seq_len, _ = hidden_states.shape
            keep_count = max(1, min(seq_len, math.ceil(seq_len * (1.0 - skip_ratio))))

            if router == "mod_magnitude_router":
                scores = hidden_states.float().norm(dim=-1)
            elif router == "mod_random_router":
                scores = torch.rand(batch, seq_len, device=hidden_states.device)
            else:
                raise ValueError(f"Unknown MoD router: {router}")

            if attention_mask is not None and attention_mask.dim() == 2:
                scores = scores.masked_fill(attention_mask <= 0, float("-inf"))

            selected = torch.zeros(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
            topk = torch.topk(scores, k=keep_count, dim=-1).indices
            selected.scatter_(1, topk, True)
            mixed_hidden = torch.where(selected.unsqueeze(-1), full_hidden, hidden_states)

            valid_tokens = attention_mask.bool() if attention_mask is not None and attention_mask.dim() == 2 else torch.ones_like(selected)
            skipped = (~selected & valid_tokens).sum().item()
            total = valid_tokens.sum().item()
            model._mod_skipped_tokens += skipped
            model._mod_total_tokens += total

            return (mixed_hidden,) + outputs[1:]

        layer.forward = mod_forward


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
def score_standard_choice(model, tensors):
    outputs = model(**tensors)
    token_count = int((tensors["labels"][:, 1:] != -100).sum().item())
    return float(outputs.loss.item()) * token_count / max(token_count, 1)


@torch.no_grad()
def score_calm_choice(model, tensors, policy, threshold):
    labels = tensors["labels"]
    confidences, layer_logits = get_calm_confidences_and_logits(model, tensors, policy)
    selected_logits, _ = select_calm_logits(confidences, layer_logits, threshold)
    nll, token_count = nll_from_logits(selected_logits, labels)
    return float(nll.item()) / max(token_count, 1)


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, dataset_name, num_samples, max_length, mode, policy=None, threshold=None):
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
    reset_mod_stats(model)

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
            tensors = make_choice_tensors(tokenizer, context, ending, max_length, model.device)
            if mode == "standard":
                scores.append(score_standard_choice(model, tensors))
            elif mode == "calm":
                scores.append(score_calm_choice(model, tensors, policy, threshold))
            else:
                raise ValueError(mode)
        correct += int(min(range(len(scores)), key=lambda idx: scores[idx]) == label)
        total += 1

    return correct / total, correct, total, get_mod_saved(model)


def make_row(policy, target_saved, realized_saved, kept_modules, skipped_modules, threshold, wikitext_nll, wikitext_ppl, hs, piqa, csqa):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "kept_modules": kept_modules,
        "skipped_modules": skipped_modules,
        "threshold": threshold,
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
    fields = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nCALM / MoD / CausalGate Tradeoff")
    print("| policy | target_saved | realized_saved | kept | skipped | threshold | WikiText PPL | HellaSwag | PIQA | CSQA |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        t = row["threshold"]
        t = "" if t == "" else f"{float(t):.6f}"
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{row['kept_modules']} | {row['skipped_modules']} | {t} | "
            f"{row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} | {row['piqa_acc']:.4f} | {row['commonsenseqa_acc']:.4f} |"
        )


def eval_standard_suite(model, tokenizer, wikitext_loader, args):
    w_nll, w_ppl, w_saved = evaluate_standard_ppl(model, wikitext_loader)
    hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", args.hellaswag_samples, args.max_length, mode="standard")[:3]
    piqa = evaluate_multiple_choice(model, tokenizer, "piqa", args.piqa_samples, args.max_length, mode="standard")[:3]
    csqa = evaluate_multiple_choice(model, tokenizer, "commonsense_qa", args.commonsenseqa_samples, args.max_length, mode="standard")[:3]
    return w_nll, w_ppl, w_saved, hs, piqa, csqa


def main():
    parser = argparse.ArgumentParser(description="Compare CALM, MoD, and CausalGate at matched saved-compute budgets.")
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default="outputs/calm_causalgate_tradeoff_wikitext_hellaswag.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()

    num_modules = sum(1 for _ in iter_gate_modules(model))
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)

    split = config["data"].get("eval_split", "test")
    calibration_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.calibration_samples)
    wikitext_loader = build_wikitext_loader(config, tokenizer, split=split, num_samples=args.wikitext_samples)

    rows = []

    disable_mod(model)
    apply_all_modules_open(model)
    w_nll, w_ppl, _, hs, piqa, csqa = eval_standard_suite(model, tokenizer, wikitext_loader, args)
    rows.append(make_row("full_model", 0.0, 0.0, num_modules, 0, "", w_nll, w_ppl, hs, piqa, csqa))

    for target_saved in args.target_saved:
        disable_mod(model)
        skipped_modules = max(0, min(num_modules - 1, round(num_modules * target_saved)))
        keep_count = num_modules - skipped_modules
        kept_indices = set(gate_ranked_indices[:keep_count])
        apply_binary_gate_mask(model, kept_indices)
        w_nll, w_ppl, _, hs, piqa, csqa = eval_standard_suite(model, tokenizer, wikitext_loader, args)
        rows.append(make_row("causalgate_topk_binary", target_saved, skipped_modules / num_modules, keep_count, skipped_modules, "", w_nll, w_ppl, hs, piqa, csqa))

    disable_mod(model)
    apply_all_modules_open(model)
    for policy in CALM_POLICIES:
        for target_saved in args.target_saved:
            threshold, _ = calibrate_threshold(model, calibration_loader, policy, target_saved)
            w_nll, w_ppl, eval_saved = evaluate_calm_ppl(model, wikitext_loader, policy, threshold)
            hs = evaluate_multiple_choice(model, tokenizer, "hellaswag", args.hellaswag_samples, args.max_length, mode="calm", policy=policy, threshold=threshold)[:3]
            piqa = evaluate_multiple_choice(model, tokenizer, "piqa", args.piqa_samples, args.max_length, mode="calm", policy=policy, threshold=threshold)[:3]
            csqa = evaluate_multiple_choice(model, tokenizer, "commonsense_qa", args.commonsenseqa_samples, args.max_length, mode="calm", policy=policy, threshold=threshold)[:3]
            rows.append(make_row(policy, target_saved, eval_saved, "", "", threshold, w_nll, w_ppl, hs, piqa, csqa))

    for policy in MOD_POLICIES:
        for target_saved in args.target_saved:
            disable_mod(model)
            apply_all_modules_open(model)
            enable_mod(model, policy, skip_ratio=target_saved)
            w_nll, w_ppl, w_saved, hs, piqa, csqa = eval_standard_suite(model, tokenizer, wikitext_loader, args)
            rows.append(make_row(policy, target_saved, w_saved, "", "", "", w_nll, w_ppl, hs, piqa, csqa))

    disable_mod(model)
    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"\nSaved comparison table to {args.output_csv}")


if __name__ == "__main__":
    main()
