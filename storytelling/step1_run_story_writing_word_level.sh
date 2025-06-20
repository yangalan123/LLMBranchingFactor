#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd storytelling

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

#constraints=("0" "1" "2" "3" "4" "5")
constraints=("0" "1" "3" "5")
#top_ps=("0.95" "0.9")
#top_ps=("0.9")
temperatures=("0.1" "0.5" "0.7")
top_p="0.9"
sequence_length=1024

#total_tasks=${#models[@]}*${#constraints[@]}*${#top_ps[@]}
total_tasks=${#models[@]}*${#constraints[@]}*${#temperatures[@]}
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

#model_idx=$((SLURM_ARRAY_TASK_ID / (${#constraints[@]} * ${#top_ps[@]})))
#constraint_idx=$((SLURM_ARRAY_TASK_ID / ${#top_ps[@]} % ${#constraints[@]}))
#top_p_idx=$((SLURM_ARRAY_TASK_ID % ${#top_ps[@]}))
model_idx=$((SLURM_ARRAY_TASK_ID / (${#constraints[@]} * ${#temperatures[@]})))
constraint_idx=$((SLURM_ARRAY_TASK_ID / ${#temperatures[@]} % ${#constraints[@]}))
temperature_idx=$((SLURM_ARRAY_TASK_ID % ${#temperatures[@]}))

min_p="1e-1"
# Extract the corresponding values
model="${models[$model_idx]}"
template="${template_list[$model_idx]}"
multi_constraints="${constraints[$constraint_idx]}"
#top_p="${top_ps[$top_p_idx]}"
top_p="0.9"
temperature="${temperatures[$temperature_idx]}"
#sequence_length="${sequence_lengths[$seq_len_idx]}"

echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}, temperature: ${temperature}"

  #--output_root_dir "response_storywriting_local_story_gen/application_ctrlgen_multi_constraints_${multi_constraints}" \
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "response_storywriting_local_story_gen_full_word_level_constraint_temperature_${temperature}/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --word_level_constraint \
  --temperature ${temperature} \
  --input_file "local_generated_story_full_ttcw/local_generated_story_extracted.pt" \
  --constraint_level "${multi_constraints}"
# for llama-3 models we need to re-run the experiment -- due to some weird problems with vllm (0.4.3, 0.5.1, 0.5.4)
#python main.py \
  #--model "${model}" \
  #--max_tokens "${sequence_length}" \
  #--output_root_dir "response_storywriting_local_story_gen_full_word_level_constraint/application_ctrlgen_multi_constraints_${multi_constraints}" \
  #--chat_template_path "${template}" \
  #--top_p "${top_p}" \
  #--word_level_constraint \
  #--input_file "local_generated_story_full_ttcw/local_generated_story_extracted.pt" \
  #--constraint_level "${multi_constraints}"
