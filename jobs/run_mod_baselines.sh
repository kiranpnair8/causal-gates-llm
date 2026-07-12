#!/bin/bash
#SBATCH --job-name=mod-baselines
#SBATCH --output=logs/mod_baselines_%j.out
#SBATCH --error=logs/mod_baselines_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
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

python scripts/eval_mod_baselines.py \
    --target-saved 0.05 0.10 0.20 0.30 0.40 \
    --wikitext-samples 128 \
    --hellaswag-samples 256 \
    --piqa-samples 256 \
    --commonsenseqa-samples 256 \
    --max-length 512 \
    --seed 123 \
    --output-csv outputs/mod_router_baselines.csv
