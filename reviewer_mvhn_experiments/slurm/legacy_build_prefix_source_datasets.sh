#!/bin/bash
#SBATCH --job-name=mvhn-rand-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Required:
#   RANDOM_STRINGS_PT: existing random-string .pt list
#   RAW_VLLM_FILE: existing random-string response dump
#   MODEL: tokenizer/model used for the dump
# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
#   METADATA_PATH, OUTPUT_DATA_DIR, PREFIX_TOKENS, MAX_EXAMPLES, SEED
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

: "${RANDOM_STRINGS_PT:?Set RANDOM_STRINGS_PT to an existing random-string .pt list.}"
: "${RAW_VLLM_FILE:?Set RAW_VLLM_FILE to an existing random-string response dump.}"
: "${MODEL:?Set MODEL to the tokenizer/model path or HF id.}"

OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-${PROJECT_DIR}/reviewer_mvhn_experiments/generated_randomness_control_datasets}"
PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
SEED="${SEED:-42}"

mkdir -p "${PROJECT_DIR}/reviewer_mvhn_experiments/slurm/logs"
mkdir -p "${OUTPUT_DATA_DIR}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode unstructured_random \
  --random_strings_pt "${RANDOM_STRINGS_PT}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_unstructured_random.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_unstructured_random.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode self_conditioned_random \
  --raw_vllm_file "${RAW_VLLM_FILE}" \
  --model "${MODEL}" \
  --prefix_tokens "${PREFIX_TOKENS}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_self_conditioned_random.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_self_conditioned_random.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode shuffled_self_conditioned_random \
  --raw_vllm_file "${RAW_VLLM_FILE}" \
  --model "${MODEL}" \
  --prefix_tokens "${PREFIX_TOKENS}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_shuffled_self_conditioned_random.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_shuffled_self_conditioned_random.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode iid_vocab_random \
  --model "${MODEL}" \
  --prefix_tokens "${PREFIX_TOKENS}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_iid_vocab_random.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_iid_vocab_random.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_control \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_structured_feedback_control.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_structured_feedback_control.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_adversarial \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_structured_feedback_adversarial.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_structured_feedback_adversarial.jsonl"

python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
  --mode structured_feedback_random_noise \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_structured_feedback_random_noise.pt" \
  --output_jsonl "${OUTPUT_DATA_DIR}/control_structured_feedback_random_noise.jsonl"

if [[ -n "${METADATA_PATH:-}" ]]; then
  python reviewer_mvhn_experiments/build_generation_randomness_control_dataset.py \
    --mode original_prompts \
    --metadata_path "${METADATA_PATH}" \
    --max_examples "${MAX_EXAMPLES}" \
    --seed "${SEED}" \
    --output_pt "${OUTPUT_DATA_DIR}/random_strings_control_original_prompts.pt" \
    --output_jsonl "${OUTPUT_DATA_DIR}/control_original_prompts.jsonl"
fi
