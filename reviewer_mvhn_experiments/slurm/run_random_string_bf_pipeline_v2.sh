#!/bin/bash
#SBATCH --job-name=mvhn-rand-bf-v2
#SBATCH --output=reviewer_mvhn_experiments/slurm/logs/%x-%A-%a.out
#SBATCH --error=reviewer_mvhn_experiments/slurm/logs/%x-%A-%a.err
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

# Required:
#   ARTIFACT_PATTERNS_FILE: patterns for *_bf.pt and/or raw vLLM *.pt artifacts
#   RANDOM_STRINGS_PT: original random-string .pt list
# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   OUTPUT_DIR, MODELS, MODEL_MAP_JSON, EXTERNAL_DELTAS, DEFAULT_MULTIPLIER
#   MIN_BASELINE_CONSTRAINT_LEVEL defaults to 1 to skip empty constraint-0 prompts
#   SAMPLE_COUNTS, DATASET_SAMPLE_COUNTS, MAX_EXAMPLES, MAX_TOKENS, LOG_PROBS
#   TOP_P, MIN_P, SEED, CHAT_TEMPLATE_MAP_JSON, DEFAULT_CHAT_TEMPLATE_PATH
#   TRUST_REMOTE_CODE=1, FORCE_EXTERNAL=1, SKIP_EXTERNAL=1, DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/reviewer_mvhn_experiments/outputs/random_string_bf_pipeline_v2}"

: "${ARTIFACT_PATTERNS_FILE:?Set ARTIFACT_PATTERNS_FILE to a path-pattern list.}"
: "${RANDOM_STRINGS_PT:?Set RANDOM_STRINGS_PT to the original random-string .pt list.}"

MODELS_DEFAULT=("Qwen/Qwen3-4B-Base" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo2-7B-1124" "allenai/OLMo-2-1124-7B-DPO")
if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -r -a MODELS_ARRAY <<< "${MODELS}"
else
  MODELS_ARRAY=("${MODELS_DEFAULT[@]}")
fi

MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( MODEL_INDEX < 0 || MODEL_INDEX >= ${#MODELS_ARRAY[@]} )); then
  echo "Invalid MODEL_INDEX=${MODEL_INDEX}; expected 0-$(( ${#MODELS_ARRAY[@]} - 1 ))." >&2
  exit 1
fi
MODEL_FILTER="${MODEL_FILTER:-${MODELS_ARRAY[$MODEL_INDEX]}}"

SAMPLE_COUNTS="${SAMPLE_COUNTS:-20}"
DATASET_SAMPLE_COUNTS="${DATASET_SAMPLE_COUNTS:-${SAMPLE_COUNTS}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LOG_PROBS="${LOG_PROBS:-50}"
TOP_P="${TOP_P:-0.9}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-42}"
EXTERNAL_DELTAS="${EXTERNAL_DELTAS:-2,4}"
DEFAULT_MULTIPLIER="${DEFAULT_MULTIPLIER:-15}"
MIN_BASELINE_CONSTRAINT_LEVEL="${MIN_BASELINE_CONSTRAINT_LEVEL:-1}"

mkdir -p "${PROJECT_DIR}/reviewer_mvhn_experiments/slurm/logs"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

EXTRA_ARGS=()
if [[ -n "${MODEL_MAP_JSON:-}" ]]; then
  EXTRA_ARGS+=(--model_map_json "${MODEL_MAP_JSON}")
fi
if [[ -n "${CHAT_TEMPLATE_MAP_JSON:-}" ]]; then
  EXTRA_ARGS+=(--chat_template_map_json "${CHAT_TEMPLATE_MAP_JSON}")
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_PATH:-}" ]]; then
  EXTRA_ARGS+=(--default_chat_template_path "${DEFAULT_CHAT_TEMPLATE_PATH}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--trust_remote_code)
fi
if [[ "${FORCE_EXTERNAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force_external)
fi
if [[ "${SKIP_EXTERNAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_external)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

python reviewer_mvhn_experiments/run_random_string_bf_pipeline_v2.py \
  --artifact_patterns_file "${ARTIFACT_PATTERNS_FILE}" \
  --random_strings_pt "${RANDOM_STRINGS_PT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_filter "${MODEL_FILTER}" \
  --external_deltas "${EXTERNAL_DELTAS}" \
  --default_multiplier "${DEFAULT_MULTIPLIER}" \
  --min_baseline_constraint_level "${MIN_BASELINE_CONSTRAINT_LEVEL}" \
  --max_examples "${MAX_EXAMPLES}" \
  --dataset_sample_counts "${DATASET_SAMPLE_COUNTS}" \
  --sample_counts "${SAMPLE_COUNTS}" \
  --max_tokens "${MAX_TOKENS}" \
  --log_probs "${LOG_PROBS}" \
  --top_p "${TOP_P}" \
  --min_p "${MIN_P}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
