#!/bin/bash
#SBATCH --job-name=direct-allopen-kl
#SBATCH --output=direct_allopen_kl_%j.out
#SBATCH --error=direct_allopen_kl_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -n "${CONDA_ENV_PATH:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_PATH"
fi

python scripts/eval_direct_allopen_kl.py
