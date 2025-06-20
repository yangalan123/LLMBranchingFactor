#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd language_modeling
#source_dir="response_mmlu_256"
source_dir="response_just_eval_instruct"
output_root_dir="just_eval_instruct_nudging_prompts_prefix_tree"
max_prefix_length=20

python ../uncertainty_quantification/nudging_find_common_prefix.py \
  --root_dir ${source_dir} \
  --output_root_dir ${output_root_dir} \
  --constraints "-1" \
  --max_prefix_length ${max_prefix_length}
