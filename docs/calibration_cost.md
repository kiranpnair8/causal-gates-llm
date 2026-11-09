# CausalGate calibration cost profiling

The canonical TinyLlama training entrypoint is `python -m training.train_gates` with
`utils/gate.yaml`. Its default seed is 42. Profiling does not alter the training
configuration or its current stop condition. The optimizer step count and
intervention forward count are measured from completed operations, not inferred
from `max_steps`.

On Lawrence, request one V100 GPU and activate the existing environment:

```bash
srun --partition=gpu --nodelist=gpu004 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=03:00:00 --pty bash
cd /home/rizk_lab/shared/kiran_m2dn/casual_contribution
eval "$(conda shell.bash hook)"
conda activate /home/rizk_lab/shared/kiran_m2dn/envs/env_gate
nvidia-smi --query-gpu=name --format=csv,noheader
```

Confirm that `nvidia-smi` reports a V100 before launching either command.
For the complete CausalGate cost run, use a fresh output directory so the
paper checkpoint is not overwritten:

```bash
python -m training.train_gates --seed 42 --profile-cost \
  --output-dir outputs/tinyllama_gated_cost_run \
  --profile-output results/calibration_cost/causalgate_cost.json
```

This produces `results/calibration_cost/causalgate_cost.json`. The
`training_calibration_seconds` measurement starts after model and data setup and
ends after the training loop. `end_to_end_seconds` also includes setup, model
loading, and checkpoint saving. Peak CUDA memory is reset just before the
training/calibration loop, so the peak fields describe that phase only.
`examples_processed` counts dataloader examples, not unique corpus documents.

For standalone Direct-KL train-split ranking cost, use the existing paper
checkpoint for the loading audit and do not run downstream evaluation:

```bash
python scripts/eval_direct_allopen_kl.py --ranking-only --profile-cost \
  --checkpoint-dir outputs/tinyllama_gated \
  --profile-output results/calibration_cost/direct_kl_ranking_cost.json
```

This produces `results/calibration_cost/direct_kl_ranking_cost.json` and no
ranking/evaluation result files. Its ranking timer includes loading and
processing the first 32 filtered WikiText-2 train examples but excludes model
loading. Its end-to-end timer includes model loading and the loading audit.
The optional Figure-2 comparison remains diagnostic-only.

Both profiling commands refuse to overwrite an existing profile JSON.
