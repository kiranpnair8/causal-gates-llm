#!/bin/bash
#SBATCH --job-name=block-influence
#SBATCH --output=logs/block_influence_submodule_%j.out
#SBATCH --error=logs/block_influence_submodule_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

python scripts/eval_block_influence_submodule.py \
    --target-saved 0.05 0.10 0.20 0.30 0.40 \
    --calibration-samples 128 \
    --wikitext-samples 128 \
    --c4-samples 128 \
    --hellaswag-samples 256 \
    --piqa-samples 256 \
    --commonsenseqa-samples 256 \
    --winogrande-samples 500 \
    --max-length 512 \
    --seed 123 \
    --ranking-csv outputs/block_influence_submodule_ranking.csv \
    --output-csv outputs/block_influence_submodule_tradeoff.csv \
    --metadata-json outputs/block_influence_submodule_metadata.json

