#!/bin/bash
#SBATCH --job-name=gate-topk-ppl
#SBATCH --output=logs/gate_topk_ppl_%j.out
#SBATCH --error=logs/gate_topk_ppl_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=kiran.prasannannair@coyotes.usd.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate /home/rizk_lab/shared/kiran_m2dn/envs/env_gate

python -m analysis.evaluate_gate_topk_ppl \
    --checkpoint-dir outputs/tinyllama_gated \
    --num-samples 64 \
    --keep-ratios 1.0 0.75 0.5 0.25 \
    --random-trials 5 \
    --seed 123 \
    --output-csv outputs/gate_topk_ppl.csv
