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
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from models.intervention import clear_intervention, set_intervention
from models.load_model import load_tinyllama_with_gates
from utils.config import load_config


MODULES = ("attn", "mlp")
KEEP_LOGIT = 20.0
SKIP_LOGIT = -20.0
RANKING_CANDIDATES = (
    "outputs/oracle_kl_module_ranking.csv",
    "outputs/gate_causal_correlation.csv",
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_fn(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)


def module_name(layer_idx, module):
    return f"L{layer_idx:02d}.{module}"


def get_module_names(model):
    names = []
    for layer_idx in range(len(model.model.layers)):
        for module in MODULES:
            names.append(module_name(layer_idx, module))
    return names


def iter_gate_modules(model):
    for layer in model.model.layers:
        yield layer.attn_gate
        yield layer.mlp_gate


def get_gate_values(model):
    return [float(gate.gate_values_scalar().detach().float().cpu()) for gate in iter_gate_modules(model)]


def get_gate_logits(model):
    return [gate.gate_logit.detach().clone() for gate in iter_gate_modules(model)]


def restore_gate_logits(model, logits):
    for gate, logit in zip(iter_gate_modules(model), logits):
        gate.gate_logit.data.copy_(logit.to(gate.gate_logit.device))


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
        key: value
        for key, value in state_dict.items()
        if key.endswith("attn_gate.gate_logit") or key.endswith("mlp_gate.gate_logit")
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


def compute_kl_delta(original_logits, intervened_logits):
    original_last = original_logits[:, -1, :].float()
    intervened_last = intervened_logits[:, -1, :].float()
    log_p = F.log_softmax(original_last, dim=-1)
    q = F.softmax(intervened_last, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


@torch.no_grad()
def compute_oracle_kl_ranking(model, loader, module_names):
    num_layers = len(model.model.layers)
    total_deltas = torch.zeros(len(module_names), dtype=torch.float32)
    num_batches = 0

    apply_all_modules_open(model)
    model.eval()
    for batch in tqdm(loader, desc="Computing Oracle-KL module ranking"):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        clear_intervention()
        original_outputs = model(**batch, use_cache=False)
        original_logits = original_outputs.logits.detach()

        deltas = []
        for layer_idx in range(num_layers):
            for module in MODULES:
                set_intervention(layer_idx=layer_idx, module=module, mode="module")
                intervened_outputs = model(**batch, use_cache=False)
                intervened_logits = intervened_outputs.logits.detach()
                clear_intervention()
                deltas.append(compute_kl_delta(original_logits, intervened_logits).detach().cpu().float())

        total_deltas += torch.stack(deltas)
        num_batches += 1

    clear_intervention()
    if num_batches == 0:
        raise ValueError("No calibration batches available for Oracle-KL scan")

    avg_deltas = total_deltas / num_batches
    rows = [
        {"module": name, "delta": float(delta), "source": "recomputed_oracle_kl"}
        for name, delta in zip(module_names, avg_deltas)
    ]
    return sorted(rows, key=lambda row: row["delta"], reverse=True)


def load_saved_kl_ranking(path, module_names):
    path = Path(path)
    if not path.exists():
        return None

    module_to_name = set(module_names)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "module" not in reader.fieldnames:
            return None

        delta_field = None
        for candidate in ("delta", "kl", "kl_delta", "causal_delta", "importance"):
            if candidate in reader.fieldnames:
                delta_field = candidate
                break
        if delta_field is None:
            return None

        for row in reader:
            module = row["module"]
            if module in module_to_name:
                rows.append({"module": module, "delta": float(row[delta_field]), "source": str(path)})

    if len(rows) != len(module_names):
        print(f"Ignoring {path}: expected {len(module_names)} modules, found {len(rows)}")
        return None

    print(f"Loaded Oracle-KL ranking from {path}")
    return sorted(rows, key=lambda row: row["delta"], reverse=True)


def find_or_compute_kl_ranking(model, calibration_loader, module_names, args):
    if args.recompute_kl:
        ranking = compute_oracle_kl_ranking(model, calibration_loader, module_names)
        write_kl_ranking_csv(ranking, args.oracle_output_csv)
        print(f"Saved recomputed Oracle-KL ranking to {args.oracle_output_csv}")
        return ranking

    if args.kl_ranking_csv:
        ranking = load_saved_kl_ranking(args.kl_ranking_csv, module_names)
        if ranking is not None:
            return ranking
        raise FileNotFoundError(f"Could not load KL ranking from {args.kl_ranking_csv}")

    for candidate in RANKING_CANDIDATES:
        ranking = load_saved_kl_ranking(candidate, module_names)
        if ranking is not None:
            return ranking

    raise FileNotFoundError(
        "No saved KL ranking found. Rerun with --recompute-kl to compute outputs/oracle_kl_module_ranking.csv"
    )


def write_kl_ranking_csv(ranking, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "module", "delta", "source"])
        writer.writeheader()
        for rank, row in enumerate(ranking, start=1):
            writer.writerow({"rank": rank, **row})


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
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels.unsqueeze(0).to(device)}


@torch.no_grad()
def score_choice(model, tensors):
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
            tensors = make_choice_tensors(tokenizer, context, ending, max_length, model.device)
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


def make_row(policy, target_saved, realized_saved, skipped_names, wikitext_nll, wikitext_ppl, hs, piqa, csqa, notes):
    return {
        "policy": policy,
        "target_saved": target_saved,
        "realized_saved": realized_saved,
        "skipped_modules": ";".join(skipped_names),
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
        "notes": notes,
    }


def write_results_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_full_kl_ranking(ranking):
    print("\nFull Oracle-KL ranking: high KL = keep, low KL = skip")
    print("| rank | module | KL delta |")
    print("|---:|---|---:|")
    for rank, row in enumerate(ranking, start=1):
        print(f"| {rank} | {row['module']} | {row['delta']:.6f} |")


def print_skipped(title, skipped_names):
    print(f"\n{title}")
    print(", ".join(skipped_names) if skipped_names else "none")


def print_results_table(rows):
    print("\nOracle-KL vs CausalGate")
    print("| policy | target_saved | realized_saved | skipped | WikiText PPL | HellaSwag | PIQA | CSQA |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        skipped_count = 0 if not row["skipped_modules"] else len(row["skipped_modules"].split(";"))
        print(
            f"| {row['policy']} | {row['target_saved']:.2f} | {row['realized_saved']:.4f} | "
            f"{skipped_count} | {row['wikitext_ppl']:.4f} | {row['hellaswag_acc']:.4f} | "
            f"{row['piqa_acc']:.4f} | {row['commonsenseqa_acc']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare raw Oracle-KL module skipping against learned CausalGate skipping.")
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--target-saved", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--wikitext-samples", type=int, default=128)
    parser.add_argument("--hellaswag-samples", type=int, default=256)
    parser.add_argument("--piqa-samples", type=int, default=256)
    parser.add_argument("--commonsenseqa-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--kl-ranking-csv", default=None)
    parser.add_argument("--oracle-output-csv", default="outputs/oracle_kl_module_ranking.csv")
    parser.add_argument("--output-csv", default="outputs/oracle_kl_vs_causalgate.csv")
    parser.add_argument("--recompute-kl", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    config["data"]["max_length"] = args.max_length

    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    trained_gate_logits = get_gate_logits(model)
    model.eval()

    split = config["data"].get("eval_split", "test")
    calibration_loader = build_wikitext_loader(config, tokenizer, split, args.calibration_samples)
    wikitext_loader = build_wikitext_loader(config, tokenizer, split, args.wikitext_samples)

    module_names = get_module_names(model)
    num_modules = len(module_names)
    module_to_idx = {name: idx for idx, name in enumerate(module_names)}

    kl_ranking = find_or_compute_kl_ranking(model, calibration_loader, module_names, args)
    restore_gate_logits(model, trained_gate_logits)
    write_kl_ranking_csv(kl_ranking, args.oracle_output_csv)
    print_full_kl_ranking(kl_ranking)

    oracle_ranked_indices = [module_to_idx[row["module"]] for row in kl_ranking]
    gate_values = get_gate_values(model)
    gate_ranked_indices = sorted(range(num_modules), key=lambda idx: gate_values[idx], reverse=True)

    rows = []
    apply_all_modules_open(model)
    w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
    rows.append(make_row("full_model", 0.0, 0.0, [], w_nll, w_ppl, hs, piqa, csqa, "all modules open"))

    skipped_sets = {}
    for target_saved in args.target_saved:
        skip_count = max(0, min(num_modules, round(num_modules * target_saved)))
        keep_count = num_modules - skip_count

        oracle_kept = set(oracle_ranked_indices[:keep_count])
        oracle_skipped = [module_names[idx] for idx in oracle_ranked_indices[keep_count:]]
        apply_binary_gate_mask(model, oracle_kept)
        w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
        rows.append(make_row(
            "oracle_kl_bottom_skip",
            target_saved,
            skip_count / num_modules,
            oracle_skipped,
            w_nll,
            w_ppl,
            hs,
            piqa,
            csqa,
            "skip lowest KL modules",
        ))
        skipped_sets[("oracle", target_saved)] = set(oracle_skipped)
        print_skipped(f"Bottom modules skipped by Oracle-KL at {target_saved:.0%}", oracle_skipped)

        gate_kept = set(gate_ranked_indices[:keep_count])
        gate_skipped = [module_names[idx] for idx in gate_ranked_indices[keep_count:]]
        apply_binary_gate_mask(model, gate_kept)
        w_nll, w_ppl, hs, piqa, csqa = evaluate_suite(model, tokenizer, wikitext_loader, args)
        rows.append(make_row(
            "learned_causalgate_bottom_skip",
            target_saved,
            skip_count / num_modules,
            gate_skipped,
            w_nll,
            w_ppl,
            hs,
            piqa,
            csqa,
            "skip lowest learned gate modules",
        ))
        skipped_sets[("gate", target_saved)] = set(gate_skipped)
        print_skipped(f"Bottom modules skipped by learned CausalGate at {target_saved:.0%}", gate_skipped)

        overlap = sorted(skipped_sets[("oracle", target_saved)] & skipped_sets[("gate", target_saved)])
        print(f"\nOverlap between Oracle-KL and CausalGate skipped sets at {target_saved:.0%}: {len(overlap)}/{skip_count}")
        print(", ".join(overlap) if overlap else "none")

    write_results_csv(rows, args.output_csv)
    print_results_table(rows)
    print(f"\nSaved Oracle-KL vs CausalGate table to {args.output_csv}")
    print(f"Saved Oracle-KL module ranking to {args.oracle_output_csv}")


if __name__ == "__main__":
    main()
