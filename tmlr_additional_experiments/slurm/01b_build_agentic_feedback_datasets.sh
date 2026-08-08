#!/bin/bash
#SBATCH --job-name=mvhn-agentic-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   OUTPUT_DATA_DIR, MAX_EXAMPLES, SEED, NOISE_CHARS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/generated_agentic_feedback_datasets}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
SEED="${SEED:-42}"
NOISE_CHARS="${NOISE_CHARS:-80}"

mkdir -p "${PROJECT_DIR}/tmlr_additional_experiments/slurm/logs"
mkdir -p "${OUTPUT_DATA_DIR}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

python tmlr_additional_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_control \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_control.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_control.jsonl"

python tmlr_additional_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_adversarial \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_adversarial.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_adversarial.jsonl"

python tmlr_additional_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_random_noise \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --noise_chars "${NOISE_CHARS}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_random_noise.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/random_strings_agentic_feedback_random_noise.jsonl"
