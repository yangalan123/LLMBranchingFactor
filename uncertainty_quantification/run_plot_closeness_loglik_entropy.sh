#!/bin/bash
echo $PATH
# please uncomment -- just comment for debugging at 10/1/2023
cd /path/to/your/project
conda activate ./env
cd uncertainty_quantification

# Define arrays for tasks and source directories
TASKS=(
  "cognac"
  "mmlu" "mmlu"
  "language_modeling" "language_modeling"
  "language_modeling"
)

SOURCE_DIRS=(
  "cognac_responses_200"
  "response_mmlu_256" "response_mmlu_256_with_filler_wa"
  "response_random_strings" "response_news"
  "response_news_1024_max_tokens_2048"
)

# Get the task ID from SLURM or use default (0)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Print the selected task and source directory
echo "Running task: ${TASKS[$TASK_ID]} with source directory: ${SOURCE_DIRS[$TASK_ID]}"

# Run the Python script with the selected task and source directory
python plot_closeness_loglik_entropy.py \
  --task ${TASKS[$TASK_ID]} \
  --source_dir ${SOURCE_DIRS[$TASK_ID]} \
  --output_dir /net/scratch2/chenghao/persona_following/visualization/closeness_loglik_entropy/${TASKS[$TASK_ID]} \
  --output_pkl_dir /net/scratch2/chenghao/persona_following/visualization/closeness_loglik_entropy_pkl/${TASKS[$TASK_ID]}
#TASK="storytelling"
#SOURCE_DIR="response_storytelling_local_story_gen_full_word_level_constraint"
#python plot_closeness_loglik_entropy.py \
  #--task ${TASK} \
  #--source_dir ${SOURCE_DIR} \
  #--output_dir /net/scratch/chenghao/persona_following/visualization/closeness_loglik_entropy/${TASK}
