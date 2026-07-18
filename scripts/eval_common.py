"""Shared helpers for TinyLlama evaluation scripts."""

import csv
import gc
import math
import random
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


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
    return [collator([item]) for item in tokenize_texts(tokenizer, texts, max_length)]


def build_wikitext_loader(config, tokenizer, split, num_samples, batch_size=1, shuffle=False):
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=config["data"]["max_length"], padding=False),
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


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


def load_dataset_examples(dataset_name, num_samples):
    if dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    elif dataset_name == "csqa":
        dataset = load_dataset("commonsense_qa", split="validation")
    elif dataset_name == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    elif dataset_name == "openbookqa":
        dataset = load_dataset("openbookqa", "main", split="validation")
    elif dataset_name == "arcc":
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    else:
        raise ValueError(f"Unsupported example dataset: {dataset_name}")
    return list(islice(dataset, num_samples))


def load_lambada_texts(num_samples):
    try:
        dataset = load_dataset("lambada", split="validation", trust_remote_code=True)
    except Exception:
        dataset = load_dataset("lambada", "plain_text", split="validation", trust_remote_code=True)
    texts = []
    for example in dataset:
        text = example.get("text", "").strip()
        if len(text.split()) >= 2:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


def load_ptb_dataset(split):
    attempts = [
        ("ptb_text_only", "penn_treebank", {"trust_remote_code": True}),
        ("ptb_text_only", None, {"trust_remote_code": True}),
    ]
    errors = []
    for name, config_name, kwargs in attempts:
        try:
            if config_name is None:
                return load_dataset(name, split=split, **kwargs)
            return load_dataset(name, config_name, split=split, **kwargs)
        except Exception as exc:
            label = name if config_name is None else f"{name}/{config_name}"
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load Penn Treebank. Tried:\n" + "\n".join(errors))


def load_ptb_texts(split, num_samples):
    texts = []
    for example in load_ptb_dataset(split):
        text = example.get("sentence", example.get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts


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


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
