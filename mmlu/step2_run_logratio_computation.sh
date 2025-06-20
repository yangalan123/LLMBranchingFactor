#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

models=("meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Meta-Llama-3-8B"
       "meta-llama/Meta-Llama-3-8B-Instruct"
       "meta-llama/Meta-Llama-3-70B-Instruct"
       "meta-llama/Meta-Llama-3-70B")
top_ps=(0.9)
num_top_ps=${#top_ps[@]}
model_index=$((SLURM_ARRAY_TASK_ID / num_top_ps))
top_p_index=$((SLURM_ARRAY_TASK_ID % num_top_ps))
model=${models[model_index]}
top_p=${top_ps[top_p_index]}
max_tokens=32
source_dir="response_mmlu_expand_options"

python ../uncertainty_quantification/log_ratio_computation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens ${max_tokens} \
  --constraints 0,1,2,3,4,5 \
  --model "${model}"

python ../uncertainty_quantification/log_ratio_computation.py \
  --source_dir ${source_dir} \
  --top_p ${top_p} \
  --max_tokens ${max_tokens} \
  --constraints 0,1,2,3,4,5 \
  --enforce_min_p \
  --model "${model}"
