# Causal Gates for TinyLlama - Current Project Status

## Core Idea

Learn adaptive transformer computation gates using causal interventions.

Instead of pruning weights statically, the model learns which transformer modules are causally important for prediction.

The current research claim is now more specific:

> Pairwise causal ranking supervision helps learned gates align with independently measured module-level causal importance.

---

# Current Architecture

Base Model:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

Gate Design:
- One scalar gate per transformer module
- Each transformer layer has:
  - attention gate
  - MLP gate

Gate Formula:
```text
gate = sigmoid(gate_logit)
```

Intervention Type:
- Module-level interventions
- Entire module output is zeroed:
  - attention output OR
  - MLP output

No architecture change has been made beyond adding scalar gates to attention and MLP residual branches.

---

# Current Pipeline

1. Normal forward pass
2. Intervene on each module independently
3. Compute KL divergence between:
   - original logits
   - intervened logits
4. Use KL delta as causal importance signal
5. Normalize causal deltas
6. Apply target floor
7. Optionally smooth causal targets with EMA
8. Train gates using:
   - LM loss
   - sparsity loss
   - MSE causal target loss
   - pairwise causal ranking loss
9. Validate learned gates against a fresh full-module causal scan

---

# Current Training Objective

```text
loss =
    lambda_lm * lm_loss
    + lambda_sparsity * sparsity_loss
    + lambda_causal * mse(gates, ema_causal_targets)
    + lambda_rank * pairwise_rank_loss(gates, ema_causal_targets)
```

Pairwise ranking loss trains the ordering:

```text
if target_i > target_j, then gate_i should be > gate_j by a margin
```

This was added because the research claim is mostly about whether gates learn causal importance ordering, not exact target magnitudes.

---

# Current Best Configuration

```yaml
training:
  learning_rate: 0.01
  max_steps: 1000
  grad_accum_steps: 8

loss:
  lambda_lm: 1.0
  lambda_sparsity: 0.001

causal:
  lambda_causal: 10.0
  target_floor: 0.25
  use_ema_targets: true
  ema_beta: 0.9
  use_rank_loss: true
  lambda_rank: 2.0
  rank_margin: 0.05
  rank_pairs: 128
```

Gate initialization:
```yaml
gate:
  init_bias: 0.0
```

This starts gates at `0.5`, where sigmoid gradients are healthy.

---

# Major Findings

## 1. Token-level interventions failed

Reason:
- KL deltas were too sparse
- Most deltas were near zero

## 2. Module-level interventions worked

Observed:
- Strong KL deltas
- Clear causal structure
- Module-level MLP and attention interventions produce useful training targets

## 3. Original gates moved too slowly

Initial issue:
- Gates stayed around `0.879 -> 0.882`

Primary causes:
- `init_bias: 2.0` started gates near `0.88`
- sigmoid gradients were weaker there
- learning rate was too small
- LM loss favored keeping modules open

Fixes:
- changed `init_bias` to `0.0`
- increased gate learning rate to `0.01`
- increased causal supervision weight
- used target floor and EMA targets

Result:
- gates now separate reliably over 1000 steps

Typical final range:
```text
gate_min ~= 0.30
gate_max ~= 0.71
gate_range ~= 0.42
```

## 4. Learned gates are stable across seeds

Stable high-gate modules include:
- `L01.attn`
- `L21.attn`
- `L01.mlp`
- `L00.attn`
- `L00.mlp`
- `L03.mlp`
- `L21.mlp`
- `L04.mlp`
- `L19.mlp` after stronger ranking supervision

Stable low-gate modules include:
- `L11.attn`
- `L12.attn`
- `L17.attn`
- `L18.attn`
- `L16.attn`
- `L10.attn`
- `L13.attn`
- `L15.attn`

This suggests the learned gate structure is not just random seed noise.

---

# Validation Results

Validation method:
- Train gates for 1000 steps
- Save trained checkpoint to `outputs/tinyllama_gated`
- Run fresh full-module causal scan
- Compare learned gates against fresh causal deltas
- Metrics:
  - Pearson correlation
  - Spearman correlation
  - Top-10 overlap
  - Bottom-10 overlap

## Baseline without ranking loss

```text
Pearson  ~= 0.406
Spearman ~= 0.486
Top-10 overlap    = 6/10
Bottom-10 overlap = 6/10
```

## Ranking loss, lambda_rank = 1.0

```text
Pearson  ~= 0.480
Spearman ~= 0.659
Top-10 overlap    = 5/10
Bottom-10 overlap = 7/10
```

## Ranking loss, lambda_rank = 2.0, seed 123, 8-sample scan

```text
Pearson  = 0.5192
Spearman = 0.7617
Top-10 overlap    = 5/10
Bottom-10 overlap = 8/10
```

## Ranking loss, lambda_rank = 2.0, seed 42, 8-sample scan

```text
Pearson  = 0.5116
Spearman = 0.7814
Top-10 overlap    = 6/10
Bottom-10 overlap = 8/10
```

## Ranking loss, lambda_rank = 2.0, seed 42, 32-sample scan

```text
Pearson  = 0.5044
Spearman = 0.7846
Top-10 overlap    = 6/10
Bottom-10 overlap = 8/10
```

Main validation result:

> With pairwise causal ranking supervision, learned gates correlate strongly with independently measured module-level causal importance. On a 32-sample fresh causal scan, gates achieved Spearman correlation `0.7846`, Pearson correlation `0.5044`, top-10 overlap `6/10`, and bottom-10 overlap `8/10`.

---

# Interpretation

The strongest evidence is rank-order alignment.

Spearman improved from approximately:

```text
0.49 -> 0.76-0.78
```

after adding pairwise ranking supervision.

Bottom-10 overlap improved from:

```text
6/10 -> 8/10
```

This means the method is especially good at identifying low-causal modules that can potentially be suppressed.

Top-10 overlap is moderate, usually `5/10` to `6/10`. Some high-causal modules are consistently captured:
- `L21.mlp`
- `L21.attn`
- `L00.mlp`
- `L01.mlp`
- `L04.mlp`
- `L19.mlp`

Persistent mismatches remain:
- `L01.attn` receives a very high gate despite only moderate fresh causal delta
- `L02.mlp` often has high fresh causal delta but a lower learned gate
- `L07.mlp` and `L20.mlp` can be high in fresh scans but not always top learned gates

Current interpretation:

> The gates do not perfectly reproduce a fresh causal scan ranking, but they learn a stable, reproducible causal-importance structure with strong rank correlation and robust low-module suppression.

---

# Current Files

Main training:
- `training/train_gates.py`

Gate definition:
- `models/gates.py`

Interventions:
- `models/intervention.py`

Model loading:
- `models/load_model.py`

Analysis:
- `analysis/full_module_scan.py`
- `analysis/compare_seed_runs.py`
- `analysis/gate_causal_correlation.py`

HPC scripts:
- `scripts/run_train_gates_hpc.sh`
- `scripts/run_gate_causal_correlation_hpc.sh`

Config:
- `utils/gate.yaml`

---

# Current Recommended Workflow

1. Train gates:
```bash
sbatch scripts/run_train_gates_hpc.sh
```

2. Run fresh causal correlation scan:
```bash
sbatch scripts/run_gate_causal_correlation_hpc.sh
```

3. Compare seed runs:
```bash
python analysis/compare_seed_runs.py logs --csv outputs/seed_gate_comparison.csv
```

4. Inspect gate-causal correlation output:
```text
outputs/gate_causal_correlation.csv
```

---

# Next Experimental Directions

## Priority 1: Validate across more data

Run fresh causal scans with larger sample counts:
- 64 samples
- 128 samples

Goal:
- confirm Spearman remains high
- confirm bottom-10 overlap remains stable

## Priority 2: Evaluate task/prompt generalization

Test whether learned gates align with causal deltas across:
- different prompt styles
- different datasets
- instruction-like text vs raw WikiText

## Priority 3: Measure quality vs sparsity

Measure:
- perplexity impact
- gate thresholding behavior
- quality when low-gate modules are disabled or skipped

## Priority 4: Dynamic compute evaluation

Eventually measure:
- actual inference speedup
- module skipping policy
- compute saved vs quality loss

---

# Research Direction

The project evolved from:

```text
simple adaptive gating
```

toward:

```text
causal ranking supervision for adaptive transformer computation
```

Main insight:

> Interventional causal attribution provides a useful supervision signal for adaptive computation, especially when trained with a ranking objective that directly matches the causal-importance claim.
