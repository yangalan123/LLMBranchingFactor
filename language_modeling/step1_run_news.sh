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
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
#
#constraints=("0" "1" "2" "3" "4" "5")
constraints=("0" "3" "5")
#top_ps=("0.95" "0.9")
top_ps=("0.9")
#sequence_length=512
sequence_length=2048

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
#output_root="response_news"
min_tokens=1024
output_root="response_news_${min_tokens}_max_tokens_${sequence_length}"
#sequence_length="${sequence_lengths[$seq_len_idx]}"

echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}"

  #--output_root_dir "response_storywriting_local_story_gen/application_ctrlgen_multi_constraints_${multi_constraints}" \
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --log_probs 50 \
  --dataset_path "RealTimeData/bbc_news_alltime" \
  --dataset_name "2024_01-07" \
  --sample_count 10 \
  --output_root_dir "${output_root}/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --min_tokens "${min_tokens}" \
  --word_level_constraint \
  --word_level_constraint_multiplier 15 \
  --min_word_count 100 \
  --constraint_level "${multi_constraints}"

# for llama-3 models we need to re-run the experiment -- due to some weird problems with vllm (0.4.3, 0.5.1, 0.5.4)
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --log_probs 50 \
  --dataset_path "RealTimeData/bbc_news_alltime" \
  --dataset_name "2024_01-07" \
  --sample_count 10 \
  --output_root_dir "${output_root}/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --top_p "${top_p}" \
  --min_tokens "${min_tokens}" \
  --word_level_constraint \
  --word_level_constraint_multiplier 15 \
  --min_word_count 100 \
  --constraint_level "${multi_constraints}"
