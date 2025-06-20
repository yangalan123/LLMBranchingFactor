#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd storytelling

source_dir="response_storywriting_local_story_gen_full_word_level_constraint"
output_root_dir="storytelling_nudging_prompts_prefix_tree"
max_prefix_length=30

python ../uncertainty_quantification/nudging_find_common_prefix.py \
  --root_dir ${source_dir} \
  --output_root_dir ${output_root_dir} \
  --max_prefix_length ${max_prefix_length}
