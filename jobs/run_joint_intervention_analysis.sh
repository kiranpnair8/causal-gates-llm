#!/bin/bash
#SBATCH --job-name=joint-intervention
#SBATCH --output=logs/joint_intervention_%j.out
#SBATCH --error=logs/joint_intervention_%j.err
#SBATCH --time=12:00:00
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

extra_args=()
if [ "${DIAGNOSE_ONLY:-0}" = "1" ]; then
    extra_args+=(--diagnose-only)
fi
if [ "${RANDOM_SUBSETS:-100}" = "100" ] && [ "${DIAGNOSE_ONLY:-0}" != "1" ] && [ -z "${VERIFY_AGAINST:-}" ]; then
    VERIFY_AGAINST=outputs/joint_intervention_diagnostic_100_metadata.json
fi
if [ -n "${VERIFY_AGAINST:-}" ]; then
    extra_args+=(--verify-against "$VERIFY_AGAINST")
fi

python scripts/eval_joint_intervention_analysis.py \
    --checkpoint-dir outputs/tinyllama_gated \
    --canonical-csv outputs/canonical_allopen_kl_tinyllama.csv \
    --seed 123 \
    --random-subsets "${RANDOM_SUBSETS:-100}" \
    "${extra_args[@]}"
