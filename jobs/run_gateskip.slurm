#!/bin/bash
#SBATCH --job-name=gateskip-style
#SBATCH --output=logs/gateskip_style_%j.out
#SBATCH --error=logs/gateskip_style_%j.err
#SBATCH --time=08:00:00
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

python scripts/eval_gateskip.py \
    --target-saved 0.05 0.10 0.20 0.30 0.40 \
    --train-steps 1000 \
    --train-samples 2048 \
    --batch-size 1 \
    --learning-rate 0.001 \
    --sparsity-loss-weight 0.01 \
    --entropy-loss-weight 0.0 \
    --gate-init-bias 5.0 \
    --gate-init-std 0.01 \
    --wikitext-samples 128 \
    --hellaswag-samples 256 \
    --piqa-samples 256 \
    --commonsenseqa-samples 256 \
    --max-length 512 \
    --seed 123 \
    --save-gates outputs/gateskip_style_gates.pt \
    --output-csv outputs/gateskip_style_tradeoff.csv
