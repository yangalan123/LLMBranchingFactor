#!/bin/bash
#SBATCH --job-name=mvhn-rand-summary
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

# Required:
#   PROJECT_DIR: repository root
#   ENV_PATH: conda environment path or name
#   RESULTS_ROOT: root containing generation-randomness *_bf.pt outputs
# Optional:
#   OUTPUT_CSV

: "${PROJECT_DIR:?Set PROJECT_DIR to the repository root.}"
: "${ENV_PATH:?Set ENV_PATH to your conda env path or name.}"
: "${RESULTS_ROOT:?Set RESULTS_ROOT to the prefix-source BF output root.}"

OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_DIR}/reviewer_mvhn_experiments/outputs/generation_randomness_summary.csv}"

mkdir -p "${PROJECT_DIR}/reviewer_mvhn_experiments/slurm/logs"
mkdir -p "$(dirname "${OUTPUT_CSV}")"

cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

python reviewer_mvhn_experiments/summarize_prefix_source_results.py \
  --results_root "${RESULTS_ROOT}" \
  --output_csv "${OUTPUT_CSV}"
