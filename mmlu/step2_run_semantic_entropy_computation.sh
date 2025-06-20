#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

#models=("meta-llama/Meta-Llama-3-8B"
#       "meta-llama/Meta-Llama-3-8B-Instruct"
#       "meta-llama/Meta-Llama-3-70B-Instruct"
#       "meta-llama/Meta-Llama-3-70B")
#template_list=("../chat_templates/chat_templates/llama-3-instruct.jinja"
#               "../chat_templates/chat_templates/llama-3-instruct.jinja"
#               "../chat_templates/chat_templates/llama-3-instruct.jinja"
#               "../chat_templates/chat_templates/llama-3-instruct.jinja")
models=("meta-llama/Llama-2-7b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-hf"
       "mistralai/Mixtral-8x7B-v0.1"
       "mistralai/Mixtral-8x7B-Instruct-v0.1"
       "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Meta-Llama-3-70B-Instruct"
        "meta-llama/Meta-Llama-3-70B")
#
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/mistral-instruct.jinja"
               "../chat_templates/chat_templates/mistral-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
#
constraints=("0" "1" "2" "3" "4" "5")
top_ps=("0.9")
sequence_length=512

total_tasks=${#models[@]}*${#constraints[@]}*${#top_ps[@]}
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

model_idx=$((SLURM_ARRAY_TASK_ID / (${#constraints[@]} * ${#top_ps[@]})))
constraint_idx=$((SLURM_ARRAY_TASK_ID / ${#top_ps[@]} % ${#constraints[@]}))
top_p_idx=$((SLURM_ARRAY_TASK_ID % ${#top_ps[@]}))

min_p="1e-1"
# Extract the corresponding values
model="${models[$model_idx]}"
template="${template_list[$model_idx]}"
multi_constraints="${constraints[$constraint_idx]}"
top_p="${top_ps[$top_p_idx]}"
#sequence_length="${sequence_lengths[$seq_len_idx]}"

echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}"

  #--output_root_dir "response_storywriting_local_story_gen/application_ctrlgen_multi_constraints_${multi_constraints}" \
python semantic_entropy_computation.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "response_mmlu/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --constraint_level "${multi_constraints}"

