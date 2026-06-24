import argparse
import csv
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoConfig


MODEL_SPECS = {
    "tinyllama": {
        "display_name": "TinyLlama-1.1B",
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "ranking": "outputs/tinyllama_gated",
        "ranking_fallbacks": [
            "outputs/gate_causal_correlation.csv",
            "outputs/tinyllama_gate_values.csv",
        ],
    },
    "qwen3b": {
        "display_name": "Qwen2.5-3B-Instruct",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "ranking": "outputs/qwen3b_causalgate/qwen3b_gate_state.pt",
        "ranking_fallbacks": [
            "outputs/qwen3b_module_ranking.csv",
            "outputs/qwen3b_gate_values.csv",
        ],
    },
}

MODULE_PATTERN = re.compile(r"(?:model\.)?layers\.(\d+)\.(attn|mlp)_gate\.gate_logit$")


def canonical_module_name(layer_idx, module_type):
    return f"L{layer_idx:02d}.{module_type}"


def expected_module_names(num_layers):
    return [
        canonical_module_name(layer_idx, module_type)
        for layer_idx in range(num_layers)
        for module_type in ("attn", "mlp")
    ]


def _gate_value(value):
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().cpu().reshape(-1)[0])
    else:
        value = float(value)
    return 1.0 / (1.0 + math.exp(-value))


def _ranking_from_state_dict(state_dict):
    values = {}
    for key, value in state_dict.items():
        match = MODULE_PATTERN.search(key)
        if match:
            name = canonical_module_name(int(match.group(1)), match.group(2))
            values[name] = _gate_value(value)
    return values


def _load_torch_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    return _ranking_from_state_dict(state)


def _load_checkpoint_directory(path):
    safetensors_path = path / "model.safetensors"
    bin_path = path / "pytorch_model.bin"
    gate_state_path = path / "qwen3b_gate_state.pt"

    if gate_state_path.exists():
        return _load_torch_state(gate_state_path)
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return _ranking_from_state_dict(load_file(str(safetensors_path)))
    if bin_path.exists():
        return _load_torch_state(bin_path)
    raise FileNotFoundError(f"No supported gate checkpoint found in {path}")


def _load_ranking_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "module" not in rows[0]:
        raise ValueError(f"{path} must contain a module column")

    values = {}
    for row in rows:
        module = row["module"].strip()
        if "gate" in row and row["gate"] not in ("", None):
            values[module] = float(row["gate"])
        elif "gate_value" in row and row["gate_value"] not in ("", None):
            values[module] = float(row["gate_value"])
        elif "rank" in row and row["rank"] not in ("", None):
            values[module] = -float(row["rank"])
        else:
            raise ValueError(f"{path} needs gate, gate_value, or rank values")
    return values


def load_gate_ranking(path, expected_names, fallbacks=()):
    candidates = [Path(path), *(Path(item) for item in fallbacks)]
    selected = next((candidate for candidate in candidates if candidate.exists()), None)
    if selected is None:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"No trained gate ranking found. Tried: {tried}")

    if selected.is_dir():
        gate_values = _load_checkpoint_directory(selected)
    elif selected.suffix.lower() == ".csv":
        gate_values = _load_ranking_csv(selected)
    else:
        gate_values = _load_torch_state(selected)

    missing = [name for name in expected_names if name not in gate_values]
    if missing:
        raise ValueError(
            f"Ranking {selected} is missing {len(missing)} modules: {missing[:8]}"
        )

    ranked = sorted(expected_names, key=lambda name: gate_values[name], reverse=True)
    print(f"Loaded {len(ranked)} gate values from {selected}")
    return ranked, gate_values, selected


def select_skipped_modules(ranked_modules, target_removal):
    num_modules = len(ranked_modules)
    num_skipped = max(0, min(num_modules - 1, round(num_modules * target_removal)))
    return ranked_modules[num_modules - num_skipped :] if num_skipped else []


def model_dimensions(config):
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    vocab_size = int(config.vocab_size)
    return (
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
    )


def module_flops(
    hidden_size,
    intermediate_size,
    num_heads,
    num_kv_heads,
    prompt_length,
    generated_tokens,
):
    # Multiply-adds are counted as two FLOPs. Attention assumes KV caching in decode.
    kv_width = hidden_size * num_kv_heads / num_heads
    projection_flops_per_token = 4 * hidden_size * (hidden_size + kv_width)
    prefill_attn = (
        prompt_length * projection_flops_per_token
        + 4 * prompt_length * prompt_length * hidden_size
    )
    decode_attn = sum(
        projection_flops_per_token
        + 4 * (prompt_length + step) * hidden_size
        for step in range(1, generated_tokens + 1)
    )
    token_count = prompt_length + generated_tokens
    mlp = 6 * token_count * hidden_size * intermediate_size
    return {"attn": float(prefill_attn + decode_attn), "mlp": float(mlp)}


def profile_model(model_key, ranking_path, removals, prompt_length, generated_tokens):
    spec = MODEL_SPECS[model_key]
    config = AutoConfig.from_pretrained(spec["model_name"], trust_remote_code=True)
    (
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
    ) = model_dimensions(config)
    names = expected_module_names(num_layers)
    ranked, _, source = load_gate_ranking(
        ranking_path or spec["ranking"],
        names,
        spec["ranking_fallbacks"],
    )
    per_module = module_flops(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        prompt_length,
        generated_tokens,
    )
    total_module_flops = num_layers * (per_module["attn"] + per_module["mlp"])
    lm_head_flops = (
        2
        * (prompt_length + generated_tokens)
        * hidden_size
        * vocab_size
    )
    total_model_flops = total_module_flops + lm_head_flops
    rows = []

    for removal in removals:
        skipped = select_skipped_modules(ranked, removal)
        skipped_flops = sum(
            per_module[module_name.rsplit(".", 1)[1]]
            for module_name in skipped
        )
        rows.append({
            "model": spec["display_name"],
            "removal": removal,
            "realized_removal": len(skipped) / len(names),
            "prompt_length": prompt_length,
            "generated_tokens": generated_tokens,
            "total_module_flops": total_module_flops,
            "lm_head_flops": lm_head_flops,
            "total_model_flops": total_model_flops,
            "skipped_module_flops": skipped_flops,
            "flop_reduction_percent": 100.0 * skipped_flops / total_model_flops,
            "num_modules": len(names),
            "num_skipped": len(skipped),
            "skipped_modules": ";".join(skipped),
            "ranking_source": str(source),
            "flop_scope": "theoretical attention+MLP+LM-head FLOPs; KV-cached generation",
        })
    return rows


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Profile theoretical CausalGate FLOP reduction."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--removals", nargs="+", type=float, default=[0.0, 0.10, 0.20])
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--tiny-ranking", default=None)
    parser.add_argument("--qwen-ranking", default=None)
    parser.add_argument("--output-csv", default="results/flops_profile.csv")
    args = parser.parse_args()
    if 0.0 not in args.removals:
        parser.error("--removals must include 0.0 for the full-model baseline")

    ranking_overrides = {
        "tinyllama": args.tiny_ranking,
        "qwen3b": args.qwen_ranking,
    }
    rows = []
    for model_key in args.models:
        rows.extend(profile_model(
            model_key,
            ranking_overrides[model_key],
            args.removals,
            args.prompt_length,
            args.generated_tokens,
        ))

    write_csv(rows, args.output_csv)
    print("\nTheoretical CausalGate FLOP Profile")
    print("| model | target removal | realized | FLOP reduction | skipped |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['removal']:.0%} | "
            f"{row['realized_removal']:.2%} | "
            f"{row['flop_reduction_percent']:.2f}% | {row['num_skipped']} |"
        )
    print(
        "\nFLOP reductions are theoretical. Latency improves only when the "
        "selected modules are bypassed before their attention/MLP computation."
    )
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
