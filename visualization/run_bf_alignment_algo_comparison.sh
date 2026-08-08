#!/bin/bash
#SBATCH --output=visualization/slurm_output/%j.stdout
#SBATCH --error=visualization/slurm_output/%j.stderr
#SBATCH --job-name=run_bf_alignment_algo_comparison
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=11:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/demo/response_storywriting}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/visualization/bf_comparison_storywriting.pdf}"

mkdir -p "${PROJECT_DIR}/visualization/slurm_output"
cd "${PROJECT_DIR}/visualization"
if [[ -n "${ENV_PATH:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_PATH}"
fi

python plot_bf_histogram.py \
  --results_dir "${RESULTS_DIR}" \
  --constraint_level 3 \
  --rename "Meta-Llama-3-8B=Base,Llama-3-8b-sft-mixture=SFT,Llama3-v2-iterative-DPO-iter1=DPO,LLaMA3-iterative-DPO-final=Iterative DPO,Llama-3-8b-rlhf-100k=PPO" \
  --output "${OUTPUT}"
