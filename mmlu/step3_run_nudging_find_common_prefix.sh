#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu
source_dir="response_mmlu_256"
output_root_dir="mmlu_nudging_prompts_prefix_tree"
max_prefix_length=50

python ../uncertainty_quantification/nudging_find_common_prefix.py \
  --root_dir ${source_dir} \
  --output_root_dir ${output_root_dir} \
  --max_prefix_length ${max_prefix_length}
