#!/bin/bash
#SBATCH --job-name=compute-tradeoff
#SBATCH --output=logs/compute_tradeoff_%j.out
#SBATCH --error=logs/compute_tradeoff_%j.err
#SBATCH --time=02:30:00
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

python -m analysis.evaluate_compute_tradeoff \
    --checkpoint-dir outputs/tinyllama_gated \
    --keep-ratios 1.0 0.95 0.90 0.85 0.80 0.75 \
    --wikitext-samples 128 \
    --hellaswag-samples 256 \
    --max-length 512 \
    --seed 123 \
    --output-csv outputs/compute_tradeoff_wikitext_hellaswag.csv
