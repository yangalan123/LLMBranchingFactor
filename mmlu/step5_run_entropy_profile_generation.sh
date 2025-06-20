#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

models=("meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-70b-hf")

models=("meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-70B"
        "meta-llama/Meta-Llama-3-70B")
nudging_model_patterns=("Meta-Llama-3-8B-Instruct"
                        "Meta-Llama-3-70B-Instruct"
                        "Meta-Llama-3-8B-Instruct"
                        "Meta-Llama-3-70B-Instruct")
#top_ps=(0.9 0.95)
top_ps=(0.9)
num_top_ps=${#top_ps[@]}
model_index=$((SLURM_ARRAY_TASK_ID / num_top_ps))
top_p_index=$((SLURM_ARRAY_TASK_ID % num_top_ps))
model=${models[model_index]}
nudging_model_pattern=${nudging_model_patterns[model_index]}
top_p=${top_ps[top_p_index]}
max_tokens=256
source_dir="response_mmlu_256_nudging"

python ../uncertainty_quantification/entropy_profile_generation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens ${max_tokens} \
  --constraints 5 \
  --additional_file_search_pattern "nudging_${nudging_model_pattern}" \
  --additional_output_dir_pattern "_nudging_${nudging_model_pattern}" \
  --model "${model}"

python ../uncertainty_quantification/entropy_profile_generation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens ${max_tokens} \
  --constraints 5 \
  --additional_file_search_pattern "nudging_${nudging_model_pattern}" \
  --additional_output_dir_pattern "_nudging_${nudging_model_pattern}" \
  --enforce_min_p \
  --model "${model}"
