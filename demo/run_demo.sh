#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch2/chenghao/bf_formal_codebase/demo/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch2/chenghao/bf_formal_codebase/demo/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch2/chenghao/bf_formal_codebase/demo/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_demo
#SBATCH --nodes=1
#SBATCH --mem=500gb
#SBATCH --ntasks=4
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120

#export PATH="/home/chenghao/miniconda3/bin:$PATH"
#alias sgpu="bash ~/slurm_interactive.sh"
#alias scpu="bash ~/slurm_interactive_cpu.sh"
#export PATH="/home/chenghao/miniconda3/bin:$PATH"
#conda init bash
echo $PATH
# please uncomment -- just comment for debugging at 10/1/2023
cd /net/scratch2/chenghao/bf_formal_codebase/demo
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch2/chenghao/multimodal_tokenizer/env
# GET ALL MODELS * 4 CHOICES in one array, then use SLURM_ARRAY_TASK_ID to index into the array
models=("allenai/OLMo2-7B-1124" "allenai/OLMo-2-1124-7B-SFT" "allenai/OLMo-2-1124-7B-DPO" "allenai/OLMo-2-1124-7B-Instruct"
        "allenai/OLMo2-13B-1124" "allenai/OLMo-2-1124-13B-SFT" "allenai/OLMo-2-1124-13B-DPO" "allenai/OLMo-2-1124-13B-Instruct" "allenai/OLMo-2-1124-7B-RM")
total_tasks=${#models[@]}
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
model_idx=$((SLURM_ARRAY_TASK_ID % ${#models[@]}))
model="${models[$model_idx]}"
echo "Running demo for ${model}"
python demo.py --model "${model}" --prompt_log_probs 0 --output_root_dir "response_wikitext_natural_level1" --constraint_level 1
