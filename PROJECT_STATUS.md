# Causal Gates for TinyLlama — Current Project Status

## Core Idea

Learn adaptive transformer computation gates using causal interventions.

Instead of pruning weights statically, the model learns which transformer modules are causally important for prediction.

---

# Current Architecture

Base Model:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

Gate Design:
- One scalar gate per module
- Each transformer layer has:
  - attention gate
  - MLP gate

Gate Formula:
gate = sigmoid(gate_logit)

Intervention Type:
- Module-level interventions
- Entire module output is zeroed:
  - attention output OR
  - MLP output

---

# Current Pipeline

1. Normal forward pass
2. Intervene on module
3. Compute KL divergence between:
   - original logits
   - intervened logits
4. Use KL delta as causal importance signal
5. Train gates toward causal importance

---

# Major Findings So Far

## 1. Token-level interventions failed
Reason:
- KL deltas too sparse
- Most deltas near zero

## 2. Module-level interventions worked
Observed:
- Strong KL deltas
- Clear causal structure

Example:
- Early MLP layers dominate causal contribution

## 3. Random pairwise ranking supervision unstable
Reason:
- Noisy local supervision
- Contradictory ranking signals

## 4. Full-module causal supervision works better
Current method:
- Compute causal deltas for ALL modules
- Normalize deltas
- Train gates using MSE:
    gate_i ≈ normalized_delta_i

---

# Current Problem

Gates move VERY slowly.

Observed:
- gates remain around:
    0.879 → 0.882

Despite:
- strong causal delta variation
- healthy gradients

Possible reasons:
- gate parameterization weak
- LM loss dominates
- optimization dynamics slow

---

# Important Experimental Results

Full causal scan showed:

Top modules:
- Layer 2 MLP
- Layer 0 MLP
- Layer 1 Attention

Many later modules:
- low causal contribution

This suggests:
- early semantic construction layers are critical
- later layers may be more redundant

---

# Current Files

Main training:
- training/train_gates.py

Gate definition:
- models/gates.py

Interventions:
- models/intervention.py

Model loading:
- models/load_model.py

Analysis:
- analysis/full_module_scan.py

Config:
- utils/gate.yaml

---

# Current Training Method

Loss:
loss =
    lm_loss
    + lambda_sparsity * sparsity_loss
    + lambda_causal * causal_loss

Current causal loss:
MSE(gates, normalized_module_deltas)

---

# Current Hyperparameters

learning_rate: 0.001
lambda_causal: 5.0

Gate:
- scalar module gates
- sigmoid activation

---

# Next Experimental Directions

## Priority 1
Investigate why gates separate slowly.

Potential ideas:
- remove LM loss temporarily
- train gates only
- EMA-smoothed causal targets
- softmax-normalized gates
- top-k causal supervision

## Priority 2
Analyze whether causal importance is stable across:
- prompts
- tasks
- datasets

## Priority 3
Measure:
- perplexity vs sparsity tradeoff
- actual inference speedup
- dynamic compute usage

---

# Research Direction

The project evolved from:
"simple adaptive gating"

toward:
"causal supervision for adaptive transformer computation"

Main insight:
Interventional causal attribution may provide a better supervision signal for adaptive computation than activation magnitude alone.