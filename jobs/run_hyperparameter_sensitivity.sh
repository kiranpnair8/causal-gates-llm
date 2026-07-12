#!/bin/bash
#SBATCH --job-name=cg-hp-sensitivity
#SBATCH --output=logs/hyperparameter_sensitivity_%j.out
#SBATCH --error=logs/hyperparameter_sensitivity_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=kiran.prasannannair@coyotes.usd.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results figures

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate /home/rizk_lab/shared/kiran_m2dn/envs/env_gate

python analysis/hyperparameter_sensitivity.py \
    --config utils/gate.yaml \
    --oracle-kl-csv outputs/oracle_kl_module_ranking.csv \
    --wikitext-samples 128 \
    --eval-split test \
    --output-csv results/hyperparameter_sensitivity.csv \
    --figure-dir figures \
    --resume
