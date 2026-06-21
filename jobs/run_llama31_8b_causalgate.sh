#!/bin/bash
#SBATCH --job-name=llama31-8b-cg
#SBATCH --output=logs/llama31_8b_causalgate_%j.out
#SBATCH --error=logs/llama31_8b_causalgate_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

# Keep large model shards and temporary downloads out of the quota-limited
# home directory while continuing to use the token created by `hf auth login`.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HOME="/home/rizk_lab/shared/kiran_m2dn/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="/home/rizk_lab/shared/kiran_m2dn/tmp/$USER"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set; relying on a cached Hugging Face login for gated Llama access."
fi

python models/llama31_8b_causalgate.py \
    --config utils/llama31_8b_gate.yaml \
    --results-csv results/llama31_8b_causalgate.csv \
    --gate-values-csv outputs/llama31_8b_gate_values.csv \
    --module-ranking-csv outputs/llama31_8b_module_ranking.csv
