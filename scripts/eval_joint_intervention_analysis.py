"""Measure the joint impact of static TinyLlama module-removal sets."""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from scripts.canonical_kl import CANONICAL_CSV, load_canonical_kl
from scripts.eval_oracle_kl_baseline import (
    KEEP_LOGIT,
    SKIP_LOGIT,
    apply_all_modules_open,
    apply_binary_gate_mask,
    build_wikitext_loader,
    get_gate_values,
    iter_gate_modules,
    get_module_names,
    load_gate_checkpoint,
    set_seed,
)
from models.load_model import load_tinyllama_with_gates
from utils.config import load_config


BUDGETS = {2: 0.05, 4: 0.10, 9: 0.20, 13: 0.30, 18: 0.40}
EXPECTED_BOTTOM_FOUR = {"L17.attn", "L12.attn", "L18.attn", "L11.attn"}
EXPECTED_BOTTOM_NINE = EXPECTED_BOTTOM_FOUR | {
    "L08.attn", "L13.attn", "L15.attn", "L16.attn", "L10.attn"
}
PERCENTILE_DEFINITION = (
    "100 * fraction of random subsets at the same k with joint KL strictly lower "
    "than the method's joint KL; lower percentile is better"
)
GATE_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(attn|mlp)_gate\.gate_logit$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="outputs/tinyllama_gated")
    parser.add_argument("--canonical-csv", default=CANONICAL_CSV)
    parser.add_argument("--random-subsets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--diagnose-only", action="store_true", help="Check fixed sets without evaluating random subsets")
    parser.add_argument("--verify-against", type=Path, help="Compare fingerprints and fixed KLs with prior diagnostic metadata")
    return parser.parse_args()


def output_paths(args):
    prefix = args.output_prefix
    if prefix is None:
        if args.diagnose_only:
            prefix = f"outputs/joint_intervention_diagnostic_{args.random_subsets}"
        else:
            prefix = "outputs/joint_intervention" if args.random_subsets == 100 else "outputs/joint_intervention_smoke"
    return {name: Path(f"{prefix}_{name}.{extension}") for name, extension in (
        ("summary", "csv"), ("random_sets", "csv"), ("metadata", "json")
    )}


def validate_checkpoint_gates(checkpoint_dir):
    checkpoint = Path(checkpoint_dir)
    safetensors_path = checkpoint / "model.safetensors"
    bin_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    elif bin_path.exists():
        keys = list(torch.load(str(bin_path), map_location="cpu").keys())
    else:
        raise FileNotFoundError(f"No gate checkpoint found in {checkpoint}")
    gate_names = []
    for key in keys:
        match = GATE_KEY_RE.search(key)
        if match:
            gate_names.append(f"L{int(match[1]):02d}.{match[2]}")
    expected = {f"L{layer:02d}.{kind}" for layer in range(22) for kind in ("attn", "mlp")}
    if len(gate_names) != 44 or set(gate_names) != expected:
        raise ValueError("Checkpoint must contain exactly one gate_logit for each of the 44 modules")
    return safetensors_path if safetensors_path.exists() else bin_path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensors(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors):
        if name.endswith("attn_gate.gate_logit") or name.endswith("mlp_gate.gate_logit"):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(data)
    return digest.hexdigest()


def backbone_sha256(model):
    return sha256_tensors(list(model.named_parameters()) + list(model.named_buffers()))


def reference_sha256(references):
    return sha256_tensors([(f"reference_{idx:02d}", value) for idx, value in enumerate(references)])


def validate_rankings(model, canonical_csv):
    names = get_module_names(model)
    expected = [f"L{layer:02d}.{kind}" for layer in range(22) for kind in ("attn", "mlp")]
    if names != expected:
        raise ValueError("Expected 44 gate modules in canonical L00.attn, L00.mlp, ... order")

    gate_values = get_gate_values(model)
    if len(gate_values) != 44 or not all(math.isfinite(value) for value in gate_values):
        raise ValueError("Checkpoint must contain 44 finite learned gate values")
    gate_order = sorted(range(44), key=lambda idx: gate_values[idx], reverse=True)
    gate_bottom = {k: [names[idx] for idx in gate_order[-k:]] for k in BUDGETS}

    for k, recorded in ((4, EXPECTED_BOTTOM_FOUR), (9, EXPECTED_BOTTOM_NINE)):
        if set(gate_bottom[k]) != recorded:
            raise ValueError(
                f"Checkpoint bottom-{k} differs from recorded Table 1 set: "
                f"found={sorted(gate_bottom[k])}, expected={sorted(recorded)}. "
                "Stopped before joint evaluation."
            )

    canonical_rows = load_canonical_kl(canonical_csv)
    if {row["module"] for row in canonical_rows} != set(names):
        raise ValueError("Canonical all-open KL CSV modules do not match the checkpoint")
    kl_bottom = {k: [row["module"] for row in canonical_rows[-k:]] for k in BUDGETS}
    return names, gate_values, gate_bottom, kl_bottom


def make_random_subsets(names, k, count, seed):
    if count < 0 or count > math.comb(len(names), k):
        raise ValueError(f"random-subsets must be 0..{math.comb(len(names), k)} for k={k}")
    rng = random.Random(seed)
    seen = set()
    subsets = []
    while len(subsets) < count:
        indices = tuple(sorted(rng.sample(range(len(names)), k)))
        if indices in seen:
            continue
        seen.add(indices)
        subsets.append(tuple(names[idx] for idx in indices))
    if len(seen) != count or any(len(set(subset)) != k for subset in subsets):
        raise AssertionError(f"Invalid random subsets at k={k}")
    return subsets


def load_batches(config, tokenizer):
    loader = build_wikitext_loader(config, tokenizer, "test", 32)
    if len(loader.dataset) != 32 or loader.batch_size != 1:
        raise ValueError("Canonical protocol requires 32 filtered WikiText-2 test examples at batch size 1")
    batches = list(loader)
    raw = load_dataset(config["data"]["dataset_name"], config["data"]["dataset_config"], split="test")
    raw = raw.filter(lambda row: len(row["text"].strip()) > 20).select(range(32))
    text_hashes = [hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in raw]
    digest = hashlib.sha256()
    for batch, row in zip(batches, raw):
        ids = batch["input_ids"].reshape(-1).tolist()
        expected_ids = tokenizer(row["text"], truncation=True, max_length=512, padding=False)["input_ids"]
        if ids != expected_ids:
            raise RuntimeError("Text/token alignment changed between dataset loads")
        digest.update(len(ids).to_bytes(4, "little"))
        for token_id in ids:
            digest.update(int(token_id).to_bytes(4, "little"))
    return batches, digest.hexdigest(), text_hashes


@torch.no_grad()
def cache_all_open_logits(model, batches):
    apply_all_modules_open(model)
    model.eval()
    assert_mask(model, get_module_names(model), ())
    references = []
    for batch in tqdm(batches, desc="Caching all-open references"):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        logits = model(**batch, use_cache=False).logits[:, -1, :]
        references.append(logits.detach().float().cpu())
    return references


def assert_mask(model, names, subset):
    actual = list(iter_gate_modules(model))
    if len(actual) != len(names) or len(actual) != 44:
        raise RuntimeError("Expected exactly 44 gates")
    removed = set(subset)
    for name, gate in zip(names, actual):
        expected = SKIP_LOGIT if name in removed else KEEP_LOGIT
        if not torch.all(gate.gate_logit.detach() == expected):
            raise RuntimeError(f"Incorrect binary mask for {name}: expected {expected}")


def assert_eval_mode(model):
    if any(module.training for module in model.modules()):
        raise RuntimeError("Model or submodule switched out of eval mode")


def joint_kl_from_logits(reference_logits, joint_logits):
    # F.kl_div(log_reference, joint_probs) computes KL(joint || reference).
    return F.kl_div(
        F.log_softmax(reference_logits.float(), dim=-1),
        F.softmax(joint_logits.float(), dim=-1),
        reduction="batchmean",
    )


@torch.no_grad()
def evaluate_subset(model, batches, references, names, subset):
    if list(names) != get_module_names(model):
        raise ValueError("Module names must be in canonical gate index order; unordered sets are invalid")
    if len(set(subset)) != len(subset) or not set(subset).issubset(names):
        raise ValueError("Subset must contain unique modules from the 44-module universe")
    if len(batches) != len(references) or not batches:
        raise ValueError("Every evaluation example needs one cached all-open reference")
    assert_eval_mode(model)
    # Reset before every subset, then apply only that subset's binary removals.
    apply_all_modules_open(model)
    kept_indices = {idx for idx, name in enumerate(names) if name not in subset}
    apply_binary_gate_mask(model, kept_indices)
    assert_mask(model, names, subset)

    total_kl = 0.0
    for batch, reference in zip(batches, references):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        joint_logits = model(**batch, use_cache=False).logits[:, -1, :]
        total_kl += float(joint_kl_from_logits(reference.to(joint_logits.device), joint_logits).item())
    mean_kl = total_kl / len(batches)
    if not math.isfinite(mean_kl):
        raise RuntimeError(f"Non-finite joint KL for subset {subset}")
    return mean_kl


def deterministic_self_test(model, batches, references, names, gate_bottom, seed):
    fixed = ("L18.attn", "L11.attn")
    probes = [subset for subset in make_random_subsets(names, 2, 4, seed) if set(subset) != set(fixed)][:3]
    repeated = [evaluate_subset(model, batches, references, names, fixed) for _ in range(3)]
    for subset in probes:
        evaluate_subset(model, batches, references, names, subset)
    repeated.append(evaluate_subset(model, batches, references, names, fixed))
    if not all(math.isclose(value, repeated[0], rel_tol=1e-6, abs_tol=1e-5) for value in repeated[1:]):
        raise RuntimeError(f"Fixed subset changed across replay: {repeated}")
    print(f"Determinism test {fixed}: {repeated}", flush=True)
    gate_repeated = repeated if set(gate_bottom[2]) == set(fixed) else [
        evaluate_subset(model, batches, references, names, gate_bottom[2]) for _ in range(3)
    ]
    if not all(math.isclose(value, gate_repeated[0], rel_tol=1e-6, abs_tol=1e-5) for value in gate_repeated[1:]):
        raise RuntimeError(f"CausalGate k=2 changed across replay: {gate_repeated}")
    return {"fixed_subset": list(fixed), "fixed_subset_kl_repeats": repeated,
            "causalgate_k2_kl_repeats": gate_repeated, "tolerance": {"rtol": 1e-6, "atol": 1e-5}}


def verify_against(path, current):
    previous = json.loads(path.read_text(encoding="utf-8"))
    keys = ("checkpoint", "checkpoint_sha256", "canonical_individual_kl_csv_sha256", "backbone_sha256",
            "tokenized_example_ids_sha256", "text_sha256", "reference_logits_sha256", "git_commit")
    for key in keys:
        if current[key] != previous.get(key):
            raise RuntimeError(f"Cross-run mismatch in {key}: current={current[key]}, previous={previous.get(key)}")
    for k, value in current["fixed_joint_kl"].items():
        prior = previous.get("fixed_joint_kl", {}).get(k, {})
        for policy, score in value.items():
            if policy not in prior or not math.isclose(score, prior[policy], rel_tol=1e-6, abs_tol=1e-5):
                raise RuntimeError(f"Cross-run fixed KL mismatch at k={k} policy={policy}: {score} vs {prior.get(policy)}")
    print(f"Fixed-set and provenance invariants match {path}", flush=True)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    args = parse_args()
    if args.seed != 123:
        raise ValueError("Canonical independent intervention protocol requires seed=123")
    if args.random_subsets == 0 and not args.diagnose_only:
        raise ValueError("--random-subsets 0 requires --diagnose-only")
    if args.random_subsets == 100 and not args.diagnose_only and not args.verify_against:
        raise ValueError("Full analysis requires --verify-against validated diagnostic metadata")
    paths = output_paths(args)
    set_seed(args.seed)
    config = load_config("utils/gate.yaml")
    if (config["model"]["name"] != "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            or config["data"]["dataset_name"] != "wikitext"
            or config["data"]["dataset_config"] != "wikitext-2-raw-v1"):
        raise ValueError("This analysis requires the canonical TinyLlama/WikiText-2 configuration")
    config["data"]["max_length"] = 512

    checkpoint_file = validate_checkpoint_gates(args.checkpoint_dir)
    checkpoint_hash = sha256_file(checkpoint_file)
    canonical_hash = sha256_file(args.canonical_csv)
    model, tokenizer = load_tinyllama_with_gates(config)
    load_gate_checkpoint(model, args.checkpoint_dir)
    model.eval()
    backbone_hash = backbone_sha256(model)
    names, gate_values, gate_bottom, kl_bottom = validate_rankings(model, args.canonical_csv)
    for k in BUDGETS:
        print(f"CausalGate bottom-{k}: {', '.join(gate_bottom[k])}", flush=True)
        print(f"Individual-KL bottom-{k}: {', '.join(kl_bottom[k])}", flush=True)

    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state().clone()
    random_sets = {k: make_random_subsets(names, k, args.random_subsets, args.seed) for k in BUDGETS}
    if random.getstate() != python_rng_state or not torch.equal(torch.get_rng_state(), torch_rng_state):
        raise RuntimeError("Random-subset generation mutated global Python or PyTorch RNG")
    batches, dataset_sha256, text_hashes = load_batches(config, tokenizer)
    references = cache_all_open_logits(model, batches)
    reference_hash = reference_sha256(references)
    print(f"Checkpoint SHA256: {checkpoint_hash}", flush=True)
    print(f"Backbone SHA256: {backbone_hash}", flush=True)
    print(f"32-text SHA256: {text_hashes}", flush=True)
    print(f"Token IDs SHA256: {dataset_sha256}", flush=True)
    print(f"All-open logits SHA256: {reference_hash}", flush=True)
    print(f"Git commit: {git_commit()}", flush=True)
    zero_kl = evaluate_subset(model, batches, references, names, ())
    print(f"All-open versus all-open joint KL: {zero_kl:.9g}", flush=True)
    if not math.isfinite(zero_kl) or abs(zero_kl) > 1e-5:
        raise RuntimeError(f"All-open k=0 sanity check failed: KL={zero_kl}")
    self_test = deterministic_self_test(model, batches, references, names, gate_bottom, args.seed)
    second_references = cache_all_open_logits(model, batches)
    if reference_sha256(second_references) != reference_hash:
        raise RuntimeError("All-open reference logits changed after subset evaluations")
    if backbone_sha256(model) != backbone_hash:
        raise RuntimeError("A non-gate model parameter or buffer changed during evaluation")

    fixed_scores = {}
    for k in BUDGETS:
        fixed_scores[str(k)] = {
            "causalgate": evaluate_subset(model, batches, references, names, gate_bottom[k]),
            "individual_kl": evaluate_subset(model, batches, references, names, kl_bottom[k]),
        }
    provenance = {
        "checkpoint": str(checkpoint_file.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "canonical_individual_kl_csv_sha256": canonical_hash,
        "backbone_sha256": backbone_hash,
        "reference_logits_sha256": reference_hash,
        "text_sha256": text_hashes,
        "tokenized_example_ids_sha256": dataset_sha256,
        "git_commit": git_commit(),
        "fixed_joint_kl": fixed_scores,
    }
    if args.verify_against:
        verify_against(args.verify_against, provenance)

    rows = []
    random_rows = []
    for k, target_budget in BUDGETS.items():
        gate_set = gate_bottom[k]
        kl_set = kl_bottom[k]
        gate_joint_kl = fixed_scores[str(k)]["causalgate"]
        individual_joint_kl = fixed_scores[str(k)]["individual_kl"]
        random_scores = []
        if not args.diagnose_only:
            for subset_id, subset in enumerate(tqdm(random_sets[k], desc=f"Random subsets k={k}"), start=1):
                score = evaluate_subset(model, batches, references, names, subset)
                random_scores.append(score)
                random_rows.append({
                    "k": k, "subset_id": subset_id, "module_names": ";".join(subset), "joint_kl": score,
                })
            for label, subset, previous_kl in (("CausalGate", gate_set, gate_joint_kl),
                                               ("Individual-KL", kl_set, individual_joint_kl)):
                replay_kl = evaluate_subset(model, batches, references, names, subset)
                if not math.isclose(previous_kl, replay_kl, rel_tol=1e-6, abs_tol=1e-5):
                    raise RuntimeError(f"k={k} {label} changed after random subsets: {previous_kl} vs {replay_kl}")
        random_tensor = torch.tensor(random_scores, dtype=torch.float64) if random_scores else None
        rows.append({
            "k": k,
            "target_budget": target_budget,
            "causalgate_modules": ";".join(gate_set),
            "causalgate_joint_kl": gate_joint_kl,
            "individual_kl_modules": ";".join(kl_set),
            "individual_kl_joint_kl": individual_joint_kl,
            "random_mean_joint_kl": random_tensor.mean().item() if random_scores else "",
            "random_std_joint_kl": statistics.stdev(random_scores) if len(random_scores) > 1 else "",
            "random_median_joint_kl": statistics.median(random_scores) if random_scores else "",
            "random_min_joint_kl": random_tensor.min().item() if random_scores else "",
            "random_max_joint_kl": random_tensor.max().item() if random_scores else "",
            "causalgate_percentile_among_random": 100.0 * sum(x < gate_joint_kl for x in random_scores) / len(random_scores) if random_scores else "",
            "individual_kl_percentile_among_random": 100.0 * sum(x < individual_joint_kl for x in random_scores) / len(random_scores) if random_scores else "",
            "causalgate_vs_individual_kl_overlap": len(set(gate_set) & set(kl_set)),
        })
        random_mean = f"{random_tensor.mean().item():.6g}" if random_scores else "not evaluated"
        print(
            f"k={k} CausalGate={gate_joint_kl:.6g} individual-KL={individual_joint_kl:.6g} "
            f"random_mean={random_mean}", flush=True,
        )

    if backbone_sha256(model) != backbone_hash:
        raise RuntimeError("A non-gate model parameter or buffer changed during the complete analysis")

    metadata = {
        "canonical_individual_kl_csv": str(args.canonical_csv),
        "diagnostic_only": args.diagnose_only,
        "self_test": self_test,
        **provenance,
        "model": config["model"]["name"],
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "test",
        "filter": "len(text.strip()) > 20",
        "sample_count": len(batches),
        "batch_size": 1,
        "max_length": 512,
        "seed": args.seed,
        "random_subsets_per_k": args.random_subsets,
        "random_generator": "Independent random.Random(seed) per k; sample 44 canonical module indices without replacement",
        "kl_direction": "KL(p_joint_pruned || p_all_open)",
        "kl_token_position": "final token",
        "reference": "Same cached all-open logits for every subset and example",
        "gate_mask_values": {"active": KEEP_LOGIT, "removed": SKIP_LOGIT},
        "module_count": len(names),
        "module_order": names,
        "gate_values": dict(zip(names, gate_values)),
        "causalgate_bottom_sets": {str(k): gate_bottom[k] for k in BUDGETS},
        "individual_kl_bottom_sets": {str(k): kl_bottom[k] for k in BUDGETS},
        "random_subset_csv": str(paths["random_sets"]),
        "percentile_definition": PERCENTILE_DEFINITION,
        "random_std_definition": "sample standard deviation (ddof=1)",
        "all_open_sanity_kl": zero_kl,
    }
    write_csv(paths["summary"], list(rows[0]), rows)
    write_csv(paths["random_sets"], ["k", "subset_id", "module_names", "joint_kl"], random_rows)
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
