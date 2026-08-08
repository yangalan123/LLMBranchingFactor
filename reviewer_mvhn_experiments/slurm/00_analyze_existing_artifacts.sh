#!/bin/bash
#SBATCH --job-name=mvhn-artifact-diag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Required:
#   PROJECT_DIR: repository root
#   ENV_PATH: conda environment path or name
#   ARTIFACT_KIND: one of bf_file, entropy_profile, raw_vllm
#   ARTIFACT_PATH: artifact to analyze
# Optional:
#   OUTPUT_DIR, MODEL, METADATA_PATH, TOP_P, WINDOW_SIZE

: "${PROJECT_DIR:?Set PROJECT_DIR to the repository root.}"
: "${ENV_PATH:?Set ENV_PATH to your conda env path or name.}"
: "${ARTIFACT_KIND:?Set ARTIFACT_KIND to bf_file, entropy_profile, or raw_vllm.}"
: "${ARTIFACT_PATH:?Set ARTIFACT_PATH to the artifact path.}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/reviewer_mvhn_experiments/outputs/artifact_diag}"
TOP_P="${TOP_P:-0.9}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"

mkdir -p "${PROJECT_DIR}/reviewer_mvhn_experiments/slurm/logs"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

COMMON_ARGS=(
  --repo_root "${PROJECT_DIR}"
  --window_size "${WINDOW_SIZE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${ARTIFACT_KIND}" == "bf_file" ]]; then
  python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --bf_file "${ARTIFACT_PATH}"
elif [[ "${ARTIFACT_KIND}" == "entropy_profile" ]]; then
  python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --entropy_profile_ckpt "${ARTIFACT_PATH}"
elif [[ "${ARTIFACT_KIND}" == "raw_vllm" ]]; then
  : "${MODEL:?Set MODEL for raw_vllm analysis.}"
  python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
    "${COMMON_ARGS[@]}" \
    --raw_vllm_file "${ARTIFACT_PATH}" \
    --metadata_path "${METADATA_PATH:-}" \
    --model "${MODEL}" \
    --top_p "${TOP_P}"
else
  echo "Unknown ARTIFACT_KIND: ${ARTIFACT_KIND}" >&2
  exit 1
fi
