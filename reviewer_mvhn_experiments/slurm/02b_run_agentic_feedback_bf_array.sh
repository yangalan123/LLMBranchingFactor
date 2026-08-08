#!/bin/bash
#SBATCH --job-name=mvhn-agentic-bf
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

# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   AGENTIC_DATA_DIR: directory produced by 01b_build_agentic_feedback_datasets.sh
#   CHAT_TEMPLATE_PATH, OUTPUT_ROOT, SAMPLE_COUNTS, MAX_TOKENS, LOG_PROBS, TOP_P
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODELS=("Qwen/Qwen3-4B-Base" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo2-7B-1124" "allenai/OLMo-2-1124-7B-DPO")
MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( MODEL_INDEX < 0 || MODEL_INDEX >= ${#MODELS[@]} )); then
  echo "Invalid MODEL_INDEX=${MODEL_INDEX}; expected 0-$(( ${#MODELS[@]} - 1 ))." >&2
  exit 1
fi
MODEL="${MODELS[$MODEL_INDEX]}"
MODEL_OUTPUT_NAME="${MODEL//\//_}"
AGENTIC_DATA_DIR="${AGENTIC_DATA_DIR:-${PROJECT_DIR}/reviewer_mvhn_experiments/generated_agentic_feedback_datasets}"

mkdir -p "${PROJECT_DIR}/reviewer_mvhn_experiments/slurm/logs"

DATASETS=(
  "random_strings_agentic_feedback_control.pt"
  "random_strings_agentic_feedback_adversarial.pt"
  "random_strings_agentic_feedback_random_noise.pt"
)

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/reviewer_mvhn_experiments/outputs/agentic_feedback_bf}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LOG_PROBS="${LOG_PROBS:-50}"
TOP_P="${TOP_P:-0.9}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

EXTRA_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH:-}" ]]; then
  EXTRA_TEMPLATE_ARGS=(--chat_template_path "${CHAT_TEMPLATE_PATH}")
fi

for DATASET_NAME in "${DATASETS[@]}"; do
  DATASET_PATH="${AGENTIC_DATA_DIR}/${DATASET_NAME}"
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
    --min_p 0 \
    --min_word_count 0 \
    --max_constraint_level 0 \
    --constraint_level 0 \
    --output_root_dir "${RUN_OUTPUT_ROOT}" \
    "${EXTRA_TEMPLATE_ARGS[@]}"
done
