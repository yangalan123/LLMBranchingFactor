#!/bin/bash
set -euo pipefail

NUM_WORKERS="${NUM_WORKERS:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/reviewer_mvhn_experiments/outputs/random_string_bf_diag}"

: "${ARTIFACT_PATTERN:?Set ARTIFACT_PATTERN to a glob for raw vLLM or *_bf.pt artifacts.}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
  --artifact_pattern "${ARTIFACT_PATTERN}" \
  --output_dir "${OUTPUT_DIR}" \
  --window_size 10 \
  --num_workers "${NUM_WORKERS}"