# CausalGate: Causal Module Selection for Efficient LLM Inference

This repository contains the code supplement for CausalGate, a causal intervention based framework for learning module-level importance rankings in transformer language models. The main experiments train scalar gates over attention and MLP modules, evaluate module removal under fixed compute budgets, and compare against several efficiency baselines.

The primary backbone used in the TinyLlama experiments is:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Additional scaling scripts are included for:

- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

## Repository Structure

```text
analysis/        Analysis utilities for causal scans, gate correlation, tradeoff evaluation, and hyperparameter sensitivity.
jobs/            SLURM job scripts used for HPC runs.
models/          Gate wrappers, intervention utilities, model loaders, and larger-model CausalGate scripts.
scripts/         Evaluation scripts, baseline implementations, profiling utilities, and plotting scripts.
training/        CausalGate training entrypoint, losses, and intervention tests.
utils/           YAML configs and config loader.
requirements.txt Python package requirements.
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Recommended optional packages for Hugging Face tokenizers:

```bash
pip install sentencepiece protobuf safetensors huggingface_hub numpy pandas matplotlib scipy
```

Some models and datasets are hosted on Hugging Face. If required, authenticate first:

```bash
huggingface-cli login
```

## Main Configuration

The default TinyLlama CausalGate configuration is:

```text
utils/gate.yaml
```

Key defaults:

- Backbone: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dataset: WikiText-2 train split
- Max length: 512
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 0.01
- Max steps: 1000
- EMA beta: 0.9
- Target floor: 0.25
- Lambda causal: 10.0
- Lambda rank: 2.0
- Rank margin: 0.05
- Rank pairs: 128

## Training CausalGate

Run TinyLlama CausalGate training:

```bash
python -m training.train_gates
```

The trained model and gate checkpoint are saved by default to:

```text
outputs/tinyllama_gated/
```

On HPC, use the provided training launcher:

```bash
sbatch scripts/run_train_gates_hpc.sh
```

## Core Analysis Pipeline

Run a full module causal scan:

```bash
python analysis/full_module_scan.py
```

Evaluate correlation between learned gates and fresh intervention scores:

```bash
python analysis/gate_causal_correlation.py
```

Evaluate gate top-k module removal:

```bash
python analysis/evaluate_gate_topk_ppl.py
```

Compare seed runs:

```bash
python analysis/compare_seed_runs.py logs/seed42.out logs/seed123.out
```

Run hyperparameter sensitivity:

```bash
python analysis/hyperparameter_sensitivity.py \
  --config utils/gate.yaml \
  --oracle-kl-csv outputs/oracle_kl_module_ranking.csv \
  --output-csv results/hyperparameter_sensitivity.csv \
  --figure-dir figures \
  --resume
```

or on HPC:

```bash
sbatch jobs/run_hyperparameter_sensitivity.sh
```

## Baseline Evaluation Scripts

The repository includes the following baseline implementations:

| Method | Script | Description |
|---|---|---|
| CausalGate + CALM | `scripts/eval_calm_vs_causalgate.py` | Compares CausalGate against CALM softmax confidence and hidden-state saturation baselines. |
| MoD-style router | `scripts/eval_mod_baselines.py` | Inference-only MoD-style random and magnitude routing baselines. |
| GateSkip-style | `scripts/eval_gateskip_style.py` | GateSkip-style residual gating baseline with gate-only fine-tuning. |
| AdaSkip-style | `scripts/eval_adaskip_style.py` | IO-similarity based zero-shot sublayer skipping baseline. |
| Activation pruning | `scripts/eval_activation_pruning.py` | Activation norm/variance based zero-shot structural pruning baseline. |
| Oracle-KL | `scripts/eval_oracle_kl_baseline.py` | Directly skips modules using raw intervention-derived KL ranking. |

Most baseline jobs can be launched with:

```bash
sbatch scripts/run_calm_vs_causalgate_hpc.sh
sbatch jobs/run_mod_baselines.sh
sbatch jobs/run_gateskip_style_eval.slurm
sbatch jobs/run_adaskip_style.sh
sbatch jobs/run_activation_pruning.sh
sbatch jobs/run_oracle_kl_baseline.sh
```

## Unified Dataset Evaluation

The consolidated all-method evaluator is:

```text
scripts/eval_all_methods.py
```

It supports the shared evaluation path for:

- WikiText-2
- C4
- HellaSwag
- PIQA
- CommonsenseQA
- WinoGrande
- OpenBookQA
- ARC-Challenge
- Penn Treebank
- LAMBADA

Example:

```bash
python scripts/eval_all_methods.py \
  --datasets wikitext c4 hellaswag piqa csqa winogrande \
  --target-saved 0.05 0.10 0.20 0.30 0.40 \
  --output-csv outputs/all_methods.csv
```

Compatibility wrappers are also included:

```bash
python scripts/eval_new_datasets_all_methods.py
python scripts/eval_arcc_all_methods.py
python scripts/eval_ptb_all_methods.py
```

## Scaling Experiments

Qwen2.5-3B:

```bash
python models/qwen3b_causalgate.py
```

or:

```bash
sbatch jobs/run_qwen3b_causalgate.sh
```

Llama-3.1-8B:

```bash
python models/llama31_8b_causalgate.py
```

or:

```bash
sbatch jobs/run_llama31_8b_causalgate.sh
```

Associated configs:

```text
utils/qwen3b_gate.yaml
utils/llama31_8b_gate.yaml
```

## Efficiency Profiling

Theoretical FLOP profile:

```bash
python scripts/profile_flops.py
```

Latency and throughput benchmark:

```bash
python scripts/benchmark_latency.py
```

Summarize efficiency results:

```bash
python scripts/summarize_efficiency_results.py
```

Outputs:

```text
results/flops_profile.csv
results/latency_throughput.csv
results/efficiency_summary.csv
```

Note: FLOP reduction is theoretical unless the script uses true module bypass. If modules are masked only after computation, wall-clock latency may not improve proportionally.

## Plotting Scripts

Publication figures can be regenerated with:

```bash
python scripts/plot_causal_importance_heatmap.py
python scripts/plot_gate_vs_causal_rank.py
python scripts/plot_active_module_ablation.py
python scripts/plot_ptb_relative_degradation.py
python scripts/plot_ppl_3B8B.py
```

Figures are saved to:

```text
figures/
```

## Statistical Analysis

Aggregate-result significance checks:

```bash
python scripts/statistical_tests.py
```

Outputs:

```text
results/statistical_tests.csv
results/statistical_tests_summary.tex
```

Important: significance tests should only be interpreted when per-example predictions, per-sequence losses, or repeated matched seed results are available. Single aggregate scores are not sufficient for paired significance testing.

## Expected Outputs

Common output directories:

```text
outputs/   Model checkpoints, gate rankings, causal rankings, and evaluation CSVs.
results/   Final summarized result tables.
figures/   Publication-quality plots.
logs/      SLURM stdout/stderr logs.
```

Large generated artifacts such as model checkpoints, Hugging Face cache files, and raw logs may be omitted from the code supplement. They can be regenerated using the scripts above.

## Reproducibility Notes

- The default random seed in `utils/gate.yaml` is `42`.
- Most TinyLlama experiments use fp16 with `device_map: auto`.
- CausalGate removes modules by sorting learned gate values and keeping the highest-ranked modules under a target compute budget.
- Oracle-KL removes modules by sorting raw intervention KL scores and skipping the lowest-KL modules.
- CALM baselines calibrate thresholds on a small calibration subset to match target saved-compute budgets.
- AdaSkip and activation-pruning baselines are zero-shot and use calibration statistics rather than gate training.
- GateSkip-style uses gate-only fine-tuning with the frozen TinyLlama backbone.

## Citation

If this repository accompanies a paper submission, cite the paper once the final bibliographic entry is available.
