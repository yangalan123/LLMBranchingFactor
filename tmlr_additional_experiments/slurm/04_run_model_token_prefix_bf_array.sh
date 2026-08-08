#!/bin/bash
#SBATCH --job-name=mvhn-token-prefix-bf
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --array=0-0

set -euo pipefail

# Required:
#   PROJECT_DIR: repository root
#   ENV_PATH: conda environment path or name
#   MANIFEST: manifest.csv or manifest.jsonl from build_model_token_prefix_random_string_datasets.py
# Optional:
#   SAMPLE_COUNTS, MAX_TOKENS, LOG_PROBS, TOP_P, MIN_P, SEED
#   CHAT_TEMPLATE_MAP_JSON, DEFAULT_CHAT_TEMPLATE_PATH

: "${PROJECT_DIR:?Set PROJECT_DIR to the repository root.}"
: "${ENV_PATH:?Set ENV_PATH to your conda env path or name.}"
: "${MANIFEST:?Set MANIFEST to the model-token-prefix manifest.}"

SAMPLE_COUNTS="${SAMPLE_COUNTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LOG_PROBS="${LOG_PROBS:-50}"
TOP_P="${TOP_P:-0.9}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-42}"

mkdir -p "${PROJECT_DIR}/tmlr_additional_experiments/slurm/logs"

cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

EXTRA_ARGS=()
if [[ -n "${CHAT_TEMPLATE_MAP_JSON:-}" ]]; then
  EXTRA_ARGS+=(--chat_template_map_json "${CHAT_TEMPLATE_MAP_JSON}")
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_PATH:-}" ]]; then
  EXTRA_ARGS+=(--default_chat_template_path "${DEFAULT_CHAT_TEMPLATE_PATH}")
fi

python tmlr_additional_experiments/run_demo_from_manifest.py \
  --manifest "${MANIFEST}" \
  --sample_counts "${SAMPLE_COUNTS}" \
  --max_tokens "${MAX_TOKENS}" \
  --log_probs "${LOG_PROBS}" \
  --top_p "${TOP_P}" \
  --min_p "${MIN_P}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
