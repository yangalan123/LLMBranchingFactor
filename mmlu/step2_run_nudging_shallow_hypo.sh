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

VLLM_HOST_IP=0.0.0.0 python nudging_probabilistic_computation.py \
  --model "${model}" \
  --max_tokens "${sequence_length}" \
  --output_root_dir "response_mmlu_${sequence_length}_with_filler_wa/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --chat_template_path "${template}" \
  --log_probs 50 \
  --top_p "${top_p}" \
  --eval_output_dir "response_mmlu_${sequence_length}_with_filler_wa/application_ctrlgen_multi_constraints_${multi_constraints}" \
  --eval_model "${eval_model}" \
  --eval_log_probs 15 \
  --ckpt_freq 32 \
  --constraint_level "${multi_constraints}"

