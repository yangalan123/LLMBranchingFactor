#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

sequence_length=256

models=("meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-70B"
        "meta-llama/Meta-Llama-3-70B")
template_list=("../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja"
               "../chat_templates/chat_templates/llama-3-instruct.jinja")
eval_models=("meta-llama/Meta-Llama-3-8B-Instruct"
             "meta-llama/Meta-Llama-3-70B-Instruct"
             "meta-llama/Meta-Llama-3-8B-Instruct"
             "meta-llama/Meta-Llama-3-70B-Instruct")
constraints=("5")
top_ps=("0.9")

model_idx=${SLURM_ARRAY_TASK_ID}
model="${models[$model_idx]}"
eval_model="${eval_models[$model_idx]}"
template="${template_list[$model_idx]}"
multi_constraints="${constraints[$constraint_idx]}"
top_p="${top_ps[$top_p_idx]}"

python nudging_top_k_computation.py \
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
  --force_recompute \
  --constraint_level "${multi_constraints}"
