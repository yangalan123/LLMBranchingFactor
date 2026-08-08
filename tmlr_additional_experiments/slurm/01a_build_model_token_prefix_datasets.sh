#!/bin/bash
#SBATCH --job-name=mvhn-token-prefix-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Required:
#   RANDOM_STRINGS_PT: original random-string .pt list
#   OUTPUT_DATA_DIR: destination for model-token-prefix datasets
# Optional:
#   PROJECT_DIR: repository root (auto-detected by default)
#   ENV_PATH: conda environment path or name (uses the active environment by default)
# One of:
#   ARTIFACT_PATTERN: one glob pattern for existing artifacts
#   ARTIFACT_PATTERNS_FILE: file with one glob pattern per line
#   MODEL_MAP_JSON, CONSTRAINTS, LOWEST_CONSTRAINT_LEVEL, TOKEN_MULTIPLIER, MAX_EXAMPLES, SEED
#   ALLOW_SHORT_PREFIXES=1 keeps source strings shorter than the model's longest requested prefix.
#   TRUST_REMOTE_CODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/generated_randomness_control_datasets}"

: "${RANDOM_STRINGS_PT:?Set RANDOM_STRINGS_PT to the original random-string .pt list.}"
if [[ -z "${ARTIFACT_PATTERN:-}" && -z "${ARTIFACT_PATTERNS_FILE:-}" ]]; then
  echo "Set ARTIFACT_PATTERN or ARTIFACT_PATTERNS_FILE." >&2
  exit 1
fi

MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
SEED="${SEED:-42}"

mkdir -p "${PROJECT_DIR}/tmlr_additional_experiments/slurm/logs"
mkdir -p "${OUTPUT_DATA_DIR}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

CMD=(
  python tmlr_additional_experiments/build_model_token_prefix_random_string_datasets.py
  --random_strings_pt "${RANDOM_STRINGS_PT}"
  --output_dir "${OUTPUT_DATA_DIR}"
  --max_examples "${MAX_EXAMPLES}"
  --seed "${SEED}"
)

if [[ -n "${ARTIFACT_PATTERN:-}" ]]; then
  CMD+=(--artifact_pattern "${ARTIFACT_PATTERN}")
fi
if [[ -n "${ARTIFACT_PATTERNS_FILE:-}" ]]; then
  CMD+=(--artifact_patterns_file "${ARTIFACT_PATTERNS_FILE}")
fi
if [[ -n "${MODEL_MAP_JSON:-}" ]]; then
  CMD+=(--model_map_json "${MODEL_MAP_JSON}")
fi
if [[ -n "${CONSTRAINTS:-}" ]]; then
  CMD+=(--constraints "${CONSTRAINTS}")
fi
if [[ -n "${LOWEST_CONSTRAINT_LEVEL:-}" ]]; then
  CMD+=(--lowest_constraint_level "${LOWEST_CONSTRAINT_LEVEL}")
fi
if [[ -n "${TOKEN_MULTIPLIER:-}" ]]; then
  CMD+=(--token_multiplier "${TOKEN_MULTIPLIER}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi
if [[ "${ALLOW_SHORT_PREFIXES:-0}" == "1" ]]; then
  CMD+=(--allow_short_prefixes)
fi

"${CMD[@]}"
