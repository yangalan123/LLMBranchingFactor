#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac

models=("meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-70b-hf"
       "mistralai/Mixtral-8x7B-v0.1"
       "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
#source_dir="cognac_responses_200"
for mode in 1 2
do
    source_dir="cognac_responses_keywords_mode_${mode}"
    max_tokens=512
    python ../uncertainty_quantification/entropy_profile_generation.py \
      --source_dir ${source_dir} \
      --constraints "0,1,2,3,4,5" \
      --top_p ${top_p} \
      --max_tokens ${max_tokens} \
      --model "${model}"

    python ../uncertainty_quantification/entropy_profile_generation.py \
      --source_dir ${source_dir} \
      --constraints "0,1,2,3,4,5" \
      --top_p ${top_p} \
      --max_tokens ${max_tokens} \
      --enforce_min_p \
      --model "${model}"
done
