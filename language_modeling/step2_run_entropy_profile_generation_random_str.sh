#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd language_modeling

models=("meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Meta-Llama-3-70B-Instruct"
        "meta-llama/Meta-Llama-3-70B")
#top_ps=(0.9 0.95)
top_ps=(0.9)
num_top_ps=${#top_ps[@]}
model_index=$((SLURM_ARRAY_TASK_ID / num_top_ps))
top_p_index=$((SLURM_ARRAY_TASK_ID % num_top_ps))
model=${models[model_index]}
top_p=${top_ps[top_p_index]}
# for latest_eval/bbc_news_all
#source_dir="response_news"
source_dir="response_random_strings"

python ../uncertainty_quantification/entropy_profile_generation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens 512 \
  --model "${model}"

python ../uncertainty_quantification/entropy_profile_generation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens 512 \
  --enforce_min_p \
  --model "${model}"
