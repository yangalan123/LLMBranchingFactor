#!/bin/bash
echo $PATH
# please uncomment -- just comment for debugging at 10/1/2023
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

models=("meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "mistralai/Mixtral-8x7B-v0.1"
       "mistralai/Mixtral-8x7B-Instruct-v0.1"
       "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Meta-Llama-3-70B-Instruct"
        "meta-llama/Meta-Llama-3-70B")
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/mistral-instruct.jinja"
               "../chat_templates/chat_templates/mistral-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
#
constraints=("0" "1" "2" "3" "4" "5")
#top_ps=("0.95" "0.9")
top_ps=("0.9")
sequence_length=5

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
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --sample_counts 1 \
  --output_root_dir "response_mmlu_expand_options/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --log_probs 50 \
  --expand_options \
  --constraint_level "${multi_constraints}"

# for llama-3 models we need to re-run the experiment -- due to some weird problems with vllm (0.4.3, 0.5.1, 0.5.4)
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --sample_counts 1 \
  --log_probs 50 \
  --output_root_dir "response_mmlu_expand_options/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --expand_options \
  --constraint_level "${multi_constraints}"
