#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd language_modeling

output_root="response_just_eval_instruct"
eval_models=("meta-llama/Meta-Llama-3-8B-Instruct"
             "meta-llama/Meta-Llama-3-70B-Instruct")
eval_model="${eval_models[$SLURM_ARRAY_TASK_ID]}"
max_prefix_lengths=(5 10 15 20)

echo "eval_model: ${eval_model}, output_root: ${output_root}"

for max_prefix_length in "${max_prefix_lengths[@]}"
do
  python ../uncertainty_quantification/demo_find_common_prefix_for_nudging.py \
    --nudging_ckpt_path "just_eval_instruct_nudging_prompts_prefix_tree/${output_root}_prefix_tree_max_prefix_length_20.pt" \
    --nudging_model "${eval_model}" \
    --nudging_max_prefix_length ${max_prefix_length} \
    --nudging_freq_threshold 50
done
