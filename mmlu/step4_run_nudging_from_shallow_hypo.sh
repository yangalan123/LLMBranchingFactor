#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

models=("meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Meta-Llama-3-8B"
       "meta-llama/Meta-Llama-3-8B-Instruct"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Meta-Llama-3-70B-Instruct"
       "meta-llama/Meta-Llama-3-70B")
template_list=("../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-2-chat.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
eval_models=("meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Meta-Llama-3-8B-Instruct"
       "meta-llama/Meta-Llama-3-8B"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Meta-Llama-3-70B"
       "meta-llama/Meta-Llama-3-70B-Instruct")

constraints=("5")
top_ps=("0.9")
sequence_length=256

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
eval_model="${eval_models[$model_idx]}"
template="${template_list[$model_idx]}"
multi_constraints="${constraints[$constraint_idx]}"
top_p="${top_ps[$top_p_idx]}"
#sequence_length="${sequence_lengths[$seq_len_idx]}"

echo "model: ${model}, min_p: ${min_p}, constraint: ${multi_constraints}, sequence_length: ${sequence_length}, top_p: ${top_p}, eval_model: ${eval_model}"

  #--output_root_dir "response_storywriting_local_story_gen/application_ctrlgen_multi_constraints_${multi_constraints}" \
  #--output_root_dir "response_mmlu_${sequence_length}/application_ctrlgen_multi_constraints_${multi_constraints}" \
#  --eval_output_dir "/net/scratch2/chenghao/persona_following/mmlu/response_mmlu_${sequence_length}_with_filler_wa/nudging_application_ctrlgen_multi_constraints_${multi_constraints}" \
python main.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "response_mmlu_${sequence_length}_nudging/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --log_probs 50 \
  --prompt_log_probs 5 \
  --top_p "${top_p}" \
  --nudging \
  --nudging_ckpt_path "mmlu_nudging_prompts_prefix_tree/response_mmlu_256_prefix_tree_max_prefix_length_15.pt" \
  --nudging_model "${eval_model}" \
  --ckpt_freq 128 \
  --constraint_level "${multi_constraints}"

# for llama-3 models we need to re-run the experiment -- due to some weird problems with vllm (0.4.3, 0.5.1, 0.5.4)
#python main.py \
  #--model "${model}" \
  #--max_tokens "${sequence_length}" \
  #--output_root_dir "response_mmlu_${sequence_length}/application_ctrlgen_multi_constraints_${multi_constraints}" \
  #--chat_template_path "${template}" \
  #--top_p "${top_p}" \
  #--constraint_level "${multi_constraints}"
