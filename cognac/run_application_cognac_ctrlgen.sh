#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac

# Define the list of models, min_p values, and sequence lengths
models=(
        "meta-llama/Llama-2-13b-chat-hf"
        "meta-llama/Llama-2-13b-hf"
        "meta-llama/Llama-2-70b-hf"
        "meta-llama/Llama-2-70b-chat-hf"
        "meta-llama/Meta-Llama-3-8B"
       "meta-llama/Meta-Llama-3-8B-Instruct"
       "meta-llama/Meta-Llama-3-70B-Instruct"
       "meta-llama/Meta-Llama-3-70B")

# Correctly create an array of corresponding chat_template (use empty string or specific placeholder for None)
# do not worry about non-chat models using template -- in utils.setup_tokenizer() and format_prompt(), we only apply chat templates to chat model
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
constraints=("1" "2" "3" "4" "5")
top_ps=("0.95" "0.9")
sequence_length=512

# Compute the total number of tasks
total_tasks=${#models[@]}*${#constraints[@]}*${#top_ps[@]}

# Check if the current task ID is valid
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

# Compute the indices for the current task ID
model_idx=$((SLURM_ARRAY_TASK_ID / (${#constraints[@]} * ${#top_ps[@]})))
constraint_idx=$((SLURM_ARRAY_TASK_ID / ${#top_ps[@]} % ${#constraints[@]}))
top_p_idx=$((SLURM_ARRAY_TASK_ID % ${#top_ps[@]}))

min_p="1e-1"
# Extract the corresponding values
model="${models[$model_idx]}"
template="${template_list[$model_idx]}"
multi_constraints="${constraints[$constraint_idx]}"
top_p="${top_ps[$top_p_idx]}"

echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}"

python application_cognac_ctrlgen.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "cognac_responses_200/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --task_selection_filename "sampled_task_cognac_app_200.pt" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --multi_constraints "${multi_constraints}"

python application_cognac_ctrlgen_get_full_prob.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "cognac_responses_200/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --task_selection_filename "sampled_task_cognac_app_200.pt" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --multi_constraints "${multi_constraints}"
#done

