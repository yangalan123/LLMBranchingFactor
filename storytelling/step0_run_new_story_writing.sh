#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd storytelling

#models=("/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-70b-chat-hf"
#       "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-13b-chat-hf"
#       "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-13b-hf"
#       "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-70b-hf"
#       "mistralai/Mixtral-8x7B-v0.1"
#       "mistralai/Mixtral-8x7B-Instruct-v0.1")
#
#template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
#               "../chat_templates/chat_templates/llama-2-chat.jinja"
#               "../chat_templates/chat_templates/llama-2-chat.jinja"
#               "../chat_templates/chat_templates/llama-2-chat.jinja"
#               "../chat_templates/chat_templates/mistral-instruct.jinja"
#               "../chat_templates/chat_templates/mistral-instruct.jinja")

models=("meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Meta-Llama-3-8B"
       "meta-llama/Meta-Llama-3-8B-Instruct"
       "meta-llama/Meta-Llama-3-70B-Instruct"
       "meta-llama/Meta-Llama-3-70B")
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")

#top_ps=("0.95" "0.9")
top_ps=("0.9")
sequence_length=1024

total_tasks=${#models[@]}*${#top_ps[@]}
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

model_idx=$((SLURM_ARRAY_TASK_ID / ${#top_ps[@]}))
top_p_idx=$((SLURM_ARRAY_TASK_ID % ${#top_ps[@]}))

min_p="1e-1"
# Extract the corresponding values
model="${models[$model_idx]}"
template="${template_list[$model_idx]}"
top_p="${top_ps[$top_p_idx]}"
#sequence_length="${sequence_lengths[$seq_len_idx]}"

#echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}"

python story_generation.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "local_generated_story_full_ttcw" \
  --chat_template_path "${template}" \
  --top_p "${top_p}"
