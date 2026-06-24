import argparse
import csv
import inspect
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.profile_flops import (
    MODEL_SPECS,
    expected_module_names,
    load_gate_ranking,
    select_skipped_modules,
)


def attention_return_count(layer_forward):
    try:
        source = inspect.getsource(layer_forward)
    except (OSError, TypeError):
        return 3
    marker = "self.self_attn("
    index = source.find(marker)
    if index < 0:
        return 3
    assignment = source[:index].rsplit("=", 1)[0].splitlines()[-1]
    return max(2, assignment.count(",") + 1)


class TrueModuleBypass:
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}

    def restore(self):
        for module, original_forward in self.original_forwards.items():
            module.forward = original_forward
        self.original_forwards.clear()

    def apply(self, skipped_names):
        self.restore()
        skipped_names = set(skipped_names)
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn_name = f"L{layer_idx:02d}.attn"
            mlp_name = f"L{layer_idx:02d}.mlp"
            if attn_name in skipped_names:
                self._bypass_attention(layer.self_attn, layer.forward)
            if mlp_name in skipped_names:
                self._bypass_mlp(layer.mlp)

    def _bypass_attention(self, module, layer_forward):
        original_forward = module.forward
        self.original_forwards[module] = original_forward
        return_count = attention_return_count(layer_forward)

        def bypass_forward(_module, *args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            output = torch.zeros_like(hidden_states)
            if return_count <= 2:
                return output, None
            return output, None, kwargs.get("past_key_value")

        module.forward = MethodType(bypass_forward, module)

    def _bypass_mlp(self, module):
        original_forward = module.forward
        self.original_forwards[module] = original_forward

        def bypass_forward(_module, hidden_states, *args, **kwargs):
            return torch.zeros_like(hidden_states)

        module.forward = MethodType(bypass_forward, module)


def build_prompt(tokenizer, batch_size, prompt_length, device):
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = 1
    input_ids = torch.full(
        (batch_size, prompt_length),
        int(token_id),
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


@torch.inference_mode()
def timed_generation(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    generated_tokens,
    warmup_runs,
    measured_runs,
):
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "use_cache": True,
        "min_new_tokens": generated_tokens,
        "max_new_tokens": generated_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for _ in range(warmup_runs):
        model.generate(**generation_kwargs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(measured_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    return latencies


def benchmark_model(model_key, ranking_path, args):
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec["model_name"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    model.config.use_cache = True

    num_layers = len(model.model.layers)
    names = expected_module_names(num_layers)
    ranked, _, source = load_gate_ranking(
        ranking_path or spec["ranking"],
        names,
        spec["ranking_fallbacks"],
    )
    input_ids, attention_mask = build_prompt(
        tokenizer,
        args.batch_size,
        args.prompt_length,
        model.device,
    )
    bypass = TrueModuleBypass(model)
    rows = []

    try:
        for removal in args.removals:
            skipped = select_skipped_modules(ranked, removal)
            bypass.apply(skipped)
            latencies = timed_generation(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                args.generated_tokens,
                args.warmup_runs,
                args.measured_runs,
            )
            mean_latency = statistics.mean(latencies)
            rows.append({
                "model": spec["display_name"],
                "removal": removal,
                "realized_removal": len(skipped) / len(names),
                "latency_seconds": mean_latency,
                "latency_std_seconds": statistics.pstdev(latencies),
                "throughput_tokens_per_second": (
                    args.batch_size * args.generated_tokens / mean_latency
                ),
                "latency_speedup": "",
                "throughput_speedup": "",
                "batch_size": args.batch_size,
                "prompt_length": args.prompt_length,
                "generated_tokens": args.generated_tokens,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.measured_runs,
                "num_modules": len(names),
                "num_skipped": len(skipped),
                "skipped_modules": ";".join(skipped),
                "ranking_source": str(source),
                "execution": "true attention/MLP forward bypass",
            })
    finally:
        bypass.restore()
        del model
        torch.cuda.empty_cache()

    baseline = next(row for row in rows if float(row["removal"]) == 0.0)
    for row in rows:
        row["latency_speedup"] = baseline["latency_seconds"] / row["latency_seconds"]
        row["throughput_speedup"] = (
            row["throughput_tokens_per_second"]
            / baseline["throughput_tokens_per_second"]
        )
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
        description="Benchmark CausalGate generation latency with true module bypass."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--removals", nargs="+", type=float, default=[0.0, 0.10, 0.20])
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--measured-runs", type=int, default=50)
    parser.add_argument("--tiny-ranking", default=None)
    parser.add_argument("--qwen-ranking", default=None)
    parser.add_argument("--output-csv", default="results/latency_throughput.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    if 0.0 not in args.removals:
        parser.error("--removals must include 0.0 to compute speedups")

    ranking_overrides = {
        "tinyllama": args.tiny_ranking,
        "qwen3b": args.qwen_ranking,
    }
    rows = []
    for model_key in args.models:
        rows.extend(benchmark_model(model_key, ranking_overrides[model_key], args))
        write_csv(rows, args.output_csv)

    print("\nCausalGate Latency and Throughput")
    print("| model | removal | latency (s) | tok/s | latency speedup |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['realized_removal']:.2%} | "
            f"{row['latency_seconds']:.4f} | "
            f"{row['throughput_tokens_per_second']:.2f} | "
            f"{row['latency_speedup']:.3f}x |"
        )
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
