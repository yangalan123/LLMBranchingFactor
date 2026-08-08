#!/bin/bash
#SBATCH --job-name=mvhn-agentic-multistep-bf
#SBATCH --output=tmlr_additional_experiments/slurm/logs/%x-%A-%a.out
#SBATCH --error=tmlr_additional_experiments/slurm/logs/%x-%A-%a.err
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

# Multi-turn agentic BF harness (option #5): a real generate -> feedback -> regenerate
# loop that measures BF at every interaction turn.
#
# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   CHAT_TEMPLATE_PATH, OUTPUT_DIR, NUM_TURNS, SAMPLE_COUNTS, MAX_TOKENS, LOG_PROBS,
#   TOP_P, ASYMPTOTIC_LIMIT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODELS=("Qwen/Qwen3-4B-Base" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo2-7B-1124" "allenai/OLMo-2-1124-7B-DPO")
MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( MODEL_INDEX < 0 || MODEL_INDEX >= ${#MODELS[@]} )); then
  echo "Invalid MODEL_INDEX=${MODEL_INDEX}; expected 0-$(( ${#MODELS[@]} - 1 ))." >&2
  exit 1
fi
MODEL="${MODELS[$MODEL_INDEX]}"

mkdir -p "${PROJECT_DIR}/tmlr_additional_experiments/slurm/logs"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/outputs/agentic_multistep_bf}"
NUM_TURNS="${NUM_TURNS:-5}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-200}"
LOG_PROBS="${LOG_PROBS:-50}"
TOP_P="${TOP_P:-0.9}"
ASYMPTOTIC_LIMIT="${ASYMPTOTIC_LIMIT:-50}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

EXTRA_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH:-}" ]]; then
  EXTRA_TEMPLATE_ARGS=(--chat_template_path "${CHAT_TEMPLATE_PATH}")
fi

echo "Running multi-turn agentic BF for model=${MODEL} (${NUM_TURNS} turns)"
python tmlr_additional_experiments/run_agentic_multistep_bf.py \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_turns "${NUM_TURNS}" \
  --sample_counts "${SAMPLE_COUNTS}" \
  --max_tokens "${MAX_TOKENS}" \
  --log_probs "${LOG_PROBS}" \
  --top_p "${TOP_P}" \
  --min_p 0 \
  --asymptotic_limit "${ASYMPTOTIC_LIMIT}" \
  "${EXTRA_TEMPLATE_ARGS[@]}"
