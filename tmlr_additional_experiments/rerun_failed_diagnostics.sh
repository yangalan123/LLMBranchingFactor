#!/bin/bash
set -euo pipefail

# Re-run the random-string artifact diagnostics for ONLY the artifacts that
# errored in a previous run. It reads <PRIOR_OUTPUT_DIR>/artifact_index.csv,
# collects every row whose status == "error", and re-runs the diagnostics on
# just those artifact paths into a separate output dir so the original full
# run is not clobbered.

NUM_WORKERS="${NUM_WORKERS:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Output dir of the previous (full) run that produced artifact_index.csv.
PRIOR_OUTPUT_DIR="${PRIOR_OUTPUT_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/outputs/random_string_bf_diag}"
# Where the re-run results are written (kept separate from the full run).
RERUN_OUTPUT_DIR="${RERUN_OUTPUT_DIR:-${PROJECT_DIR}/tmlr_additional_experiments/outputs/random_string_bf_diag_rerun}"

cd "${PROJECT_DIR}"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

INDEX_CSV="${PRIOR_OUTPUT_DIR}/artifact_index.csv"
if [[ ! -f "${INDEX_CSV}" ]]; then
  echo "Could not find artifact_index.csv at: ${INDEX_CSV}" >&2
  echo "Set PRIOR_OUTPUT_DIR to the previous run's --output_dir." >&2
  exit 1
fi

# Extract artifact_path for rows with status == error (robust CSV parsing).
FAILED_PATHS_FILE="$(mktemp)"
trap 'rm -f "${FAILED_PATHS_FILE}"' EXIT
python - "${INDEX_CSV}" > "${FAILED_PATHS_FILE}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (row.get("status") or "").strip() == "error":
            path = (row.get("artifact_path") or "").strip()
            if path:
                print(path)
PY

NUM_FAILED="$(wc -l < "${FAILED_PATHS_FILE}" | tr -d '[:space:]')"
if [[ "${NUM_FAILED}" -eq 0 ]]; then
  echo "No errored artifacts found in ${INDEX_CSV}. Nothing to re-run."
  exit 0
fi
echo "Re-running ${NUM_FAILED} previously-failed artifact(s) into ${RERUN_OUTPUT_DIR}"

# Build one --artifact_path argument per failed path.
ARTIFACT_ARGS=()
while IFS= read -r path; do
  [[ -z "${path}" ]] && continue
  ARTIFACT_ARGS+=(--artifact_path "${path}")
done < "${FAILED_PATHS_FILE}"

python tmlr_additional_experiments/random_string_artifact_diagnostics.py \
  "${ARTIFACT_ARGS[@]}" \
  --output_dir "${RERUN_OUTPUT_DIR}" \
  --window_size 10 \
  --num_workers "${NUM_WORKERS}"
