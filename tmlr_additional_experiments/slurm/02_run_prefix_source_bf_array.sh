#!/bin/bash
#SBATCH --job-name=mvhn-rand-bf
#SBATCH --output=tmlr_additional_experiments/slurm/logs/%x-%A-%a.out
#SBATCH --error=tmlr_additional_experiments/slurm/logs/%x-%A-%a.err
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
# Manifest and legacy modes both use 0-(models - 1).
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

# Manifest mode:
#   MANIFEST: manifest.csv/jsonl produced by 01a_build_model_token_prefix_datasets.sh
#   If MANIFEST is unset, ${CONTROL_DATA_DIR}/manifest.csv or .jsonl is used when present.
#   SLURM_ARRAY_TASK_ID selects a model from MODELS, then filters manifest rows to that model.
# Legacy flat-dataset mode:
#   CONTROL_DATA_DIR: directory produced by legacy_build_prefix_source_datasets.sh
#   Set USE_LEGACY_DATASETS=1 to force this mode when a manifest is present.
# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   CHAT_TEMPLATE_PATH, CHAT_TEMPLATE_MAP_JSON, DEFAULT_CHAT_TEMPLATE_PATH
#   MANIFEST_SELECTION=model_all|model_max_constraint
#   OUTPUT_ROOT, SAMPLE_COUNTS, MAX_TOKENS, LOG_PROBS, TOP_P, MIN_P, SEED

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODELS=("Qwen/Qwen3-4B-Base" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo2-7B-1124" "allenai/OLMo-2-1124-7B-DPO")
CONTROL_DATA_DIR="${CONTROL_DATA_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/generated_randomness_control_datasets}"
MANIFEST="${MANIFEST:-}"
USE_LEGACY_DATASETS="${USE_LEGACY_DATASETS:-0}"
MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( MODEL_INDEX < 0 || MODEL_INDEX >= ${#MODELS[@]} )); then
  echo "Invalid MODEL_INDEX=${MODEL_INDEX}; expected 0-$(( ${#MODELS[@]} - 1 ))." >&2
  exit 1
fi
MODEL="${MODELS[$MODEL_INDEX]}"
MODEL_OUTPUT_NAME="${MODEL//\//_}"

mkdir -p "${PROJECT_DIR}/tmlr_additional_experiments/slurm/logs"

DATASETS=(
  "random_strings_control_unstructured_random.pt"
  "random_strings_control_self_conditioned_random.pt"
  "random_strings_control_shuffled_self_conditioned_random.pt"
  "random_strings_control_iid_vocab_random.pt"
  "random_strings_control_structured_feedback_control.pt"
  "random_strings_control_structured_feedback_adversarial.pt"
  "random_strings_control_structured_feedback_random_noise.pt"
)

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/tmlr_additional_experiments/outputs/generation_randomness_bf}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LOG_PROBS="${LOG_PROBS:-50}"
TOP_P="${TOP_P:-0.9}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-42}"
MANIFEST_SELECTION="${MANIFEST_SELECTION:-model_all}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

EXTRA_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH:-}" ]]; then
  EXTRA_TEMPLATE_ARGS=(--chat_template_path "${CHAT_TEMPLATE_PATH}")
fi

if [[ "${USE_LEGACY_DATASETS}" != "1" ]]; then
  if [[ -z "${MANIFEST}" && -f "${CONTROL_DATA_DIR}/manifest.csv" ]]; then
    MANIFEST="${CONTROL_DATA_DIR}/manifest.csv"
  elif [[ -z "${MANIFEST}" && -f "${CONTROL_DATA_DIR}/manifest.jsonl" ]]; then
    MANIFEST="${CONTROL_DATA_DIR}/manifest.jsonl"
  fi
fi

if [[ -n "${MANIFEST}" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}" >&2
    exit 1
  fi

  MANIFEST_ARGS=()
  if [[ -n "${CHAT_TEMPLATE_MAP_JSON:-}" ]]; then
    MANIFEST_ARGS+=(--chat_template_map_json "${CHAT_TEMPLATE_MAP_JSON}")
  fi
  if [[ -n "${DEFAULT_CHAT_TEMPLATE_PATH:-}" ]]; then
    MANIFEST_ARGS+=(--default_chat_template_path "${DEFAULT_CHAT_TEMPLATE_PATH}")
  elif [[ -n "${CHAT_TEMPLATE_PATH:-}" ]]; then
    MANIFEST_ARGS+=(--default_chat_template_path "${CHAT_TEMPLATE_PATH}")
  fi

  echo "Running manifest rows for model=${MODEL} from ${MANIFEST}"
  python tmlr_additional_experiments/run_demo_from_manifest.py \
    --manifest "${MANIFEST}" \
    --model_filter "${MODEL}" \
    --selection "${MANIFEST_SELECTION}" \
    --sample_counts "${SAMPLE_COUNTS}" \
    --max_tokens "${MAX_TOKENS}" \
    --log_probs "${LOG_PROBS}" \
    --top_p "${TOP_P}" \
    --min_p "${MIN_P}" \
    --seed "${SEED}" \
    "${MANIFEST_ARGS[@]}"
  exit 0
fi

for DATASET_NAME in "${DATASETS[@]}"; do
  DATASET_PATH="${CONTROL_DATA_DIR}/${DATASET_NAME}"
  MODE_NAME="${DATASET_NAME%.pt}"
  RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${MODE_NAME}/${MODEL_OUTPUT_NAME}"

  if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "Missing dataset: ${DATASET_PATH}" >&2
    exit 1
  fi

  echo "Running model=${MODEL} dataset=${MODE_NAME}"
  python demo/demo.py \
    --task_type language_modeling \
    --model "${MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_name "" \
    --dataset_sample_counts "${SAMPLE_COUNTS}" \
    --sample_counts "${SAMPLE_COUNTS}" \
    --max_tokens "${MAX_TOKENS}" \
    --log_probs "${LOG_PROBS}" \
    --top_p "${TOP_P}" \
    --min_p "${MIN_P}" \
    --seed "${SEED}" \
    --min_word_count 0 \
    --max_constraint_level 0 \
    --constraint_level 0 \
    --output_root_dir "${RUN_OUTPUT_ROOT}" \
    "${EXTRA_TEMPLATE_ARGS[@]}"
done
