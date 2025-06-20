#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd mmlu

offsets=(0 50 100)
nudging_model_patterns=("Meta-Llama-3-8B-Instruct"
                        "Meta-Llama-3-8B-Instruct"
                        "Meta-Llama-3-70B-Instruct"
                        "Meta-Llama-3-70B-Instruct")
# create an array of offsets rather than manual setup, so that we can easily change the offsets
# start from 0, end at 600, step 50
#offsets=("$(seq 0 50 1000)")
#root_dir="response_storywriting"
#root_dir="response_storywriting_local_story_gen"
root_dir="response_mmlu_256_nudging"
max_tokens=256
#ckpt_dirs=("output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9"
#            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9_enforce_min_p_0.1")
ckpt_dirs=("output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9_nudging_Meta-Llama-3-8B-Instruct"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9_enforce_min_p_0.1_nudging_Meta-Llama-3-8B-Instruct"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9_nudging_Meta-Llama-3-70B-Instruct"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_${max_tokens}_min_p_0_top_p_0.9_enforce_min_p_0.1_nudging_Meta-Llama-3-70B-Instruct")
#            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95_enforce_min_p_0.1"
#           "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95")
high_entropy_thresholds=(0.7)

# get offset_index, ckpt_dir_index, high_entropy_threshold_index
# slurm_array_task_id = offset_index * #(ckpt_dirs) * #(high_entropy_thresholds) + ckpt_dir_index * #(high_entropy_thresholds) + high_entropy_threshold_index
offset_index=$((SLURM_ARRAY_TASK_ID / ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ))
ckpt_dir_index=$(( (SLURM_ARRAY_TASK_ID % ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ) / ${#high_entropy_thresholds[@]} ))
nudging_model_pattern=${nudging_model_patterns[ckpt_dir_index]}
high_entropy_threshold_index=$((SLURM_ARRAY_TASK_ID % ${#high_entropy_thresholds[@]} ))
offset=${offsets[offset_index]}
ckpt_dir=${ckpt_dirs[ckpt_dir_index]}
high_entropy_threshold=${high_entropy_thresholds[high_entropy_threshold_index]}
echo "offset: ${offset}, ckpt_dir: ${ckpt_dir}, high_entropy_threshold: ${high_entropy_threshold}"


#python stat_lm.py \
#--ckpt_dir ${ckpt_dir} \
#--offset ${offset} \
#--high_entropy_threshold ${high_entropy_threshold}
#python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
#  --ckpt_dir ${ckpt_dir} \
#  --offset ${offset} \
#  --high_entropy_threshold ${high_entropy_threshold} \
#  --constraints "0,1,3,5" \
#  --output_root_dir "stat_mmlu_app_ctrlgen_multi_constraints"

for smoothing_factor in 0.1 0.5 1.0
do
  python stat_mmlu.py \
    --ckpt_dir ${ckpt_dir} \
    --offset ${offset} \
    --high_entropy_threshold ${high_entropy_threshold} \
    --constraints "5" \
    --smoothing_factor ${smoothing_factor} \
    --piecewise_ebf \
    --source_dir ${root_dir} \
    --additional_file_search_pattern "nudging_${nudging_model_pattern}" \
    --output_root_dir "stat_mmlu_nudging_app_ctrlgen_multi_constraints"

  for maxlen in 5 50
  do
    python stat_mmlu.py \
      --ckpt_dir ${ckpt_dir} \
      --offset ${offset} \
      --high_entropy_threshold ${high_entropy_threshold} \
      --constraints "5" \
      --smoothing_factor ${smoothing_factor} \
      --piecewise_ebf \
      --maxlen ${maxlen} \
      --source_dir ${root_dir} \
      --additional_file_search_pattern "nudging_${nudging_model_pattern}" \
      --output_root_dir "stat_mmlu_nudging_app_ctrlgen_multi_constraints_maxlen_${maxlen}"
  done
done
