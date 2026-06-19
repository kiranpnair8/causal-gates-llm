#!/bin/bash
#SBATCH --job-name=qwen3b-causalgate
#SBATCH --output=logs/qwen3b_causalgate_%j.out
#SBATCH --error=logs/qwen3b_causalgate_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=kiran.prasannannair@coyotes.usd.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs outputs results

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate /home/rizk_lab/shared/kiran_m2dn/envs/env_gate

python models/qwen3b_causalgate.py \
    --config utils/qwen3b_gate.yaml \
    --results-csv results/qwen3b_causalgate.csv \
    --gate-values-csv outputs/qwen3b_gate_values.csv \
    --module-ranking-csv outputs/qwen3b_module_ranking.csv
