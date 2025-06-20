#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac

#offsets=(0 50 100 150 200 250 300 350 400 450 500 550 600)
offsets=(0)
#root_dir="response_storywriting"
#root_dir="response_storywriting_local_story_gen"
root_dir="cognac_responses_200"
ckpt_dirs=("output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9_enforce_min_p_0.1"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95_enforce_min_p_0.1"
           "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95")
high_entropy_thresholds=(0.7)

# get offset_index, ckpt_dir_index, high_entropy_threshold_index
# slurm_array_task_id = offset_index * #(ckpt_dirs) * #(high_entropy_thresholds) + ckpt_dir_index * #(high_entropy_thresholds) + high_entropy_threshold_index
offset_index=$((SLURM_ARRAY_TASK_ID / ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ))
ckpt_dir_index=$(( (SLURM_ARRAY_TASK_ID % ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ) / ${#high_entropy_thresholds[@]} ))
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
#  --constraints "0,1,2,3,4,5" \
#  --piecewise_ebf \
#  --output_root_dir "stat_cognac_app_ctrlgen_multi_constraints"

for smoothing_factor in 0.1 0.5 1.0
do
  python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
    --ckpt_dir ${ckpt_dir} \
    --offset ${offset} \
    --high_entropy_threshold ${high_entropy_threshold} \
    --constraints "0,1,2,3,4,5" \
    --piecewise_ebf \
    --smoothing_factor ${smoothing_factor} \
    --output_root_dir "stat_cognac_app_ctrlgen_multi_constraints"

  for maxlen in 5 50
  do
    python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
      --ckpt_dir ${ckpt_dir} \
      --offset ${offset} \
      --maxlen ${maxlen} \
      --high_entropy_threshold ${high_entropy_threshold} \
      --constraints "0,1,2,3,4,5" \
      --piecewise_ebf \
      --smoothing_factor ${smoothing_factor} \
      --output_root_dir "stat_cognac_app_ctrlgen_multi_constraints_maxlen_${maxlen}"
  done
done
for smoothing_factor in 0.1 0.5 1.0
do
  python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
    --ckpt_dir ${ckpt_dir} \
    --offset ${offset} \
    --high_entropy_threshold ${high_entropy_threshold} \
    --constraints "0,1,2,3,4,5,6,8,10" \
    --piecewise_ebf \
    --smoothing_factor ${smoothing_factor} \
    --output_root_dir "stat_cognac_app_ctrlgen_multi_constraints_extended"

  for maxlen in 5 50
  do
    python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
      --ckpt_dir ${ckpt_dir} \
      --offset ${offset} \
      --maxlen ${maxlen} \
      --high_entropy_threshold ${high_entropy_threshold} \
      --constraints "0,1,2,3,4,5,6,8,10" \
      --piecewise_ebf \
      --smoothing_factor ${smoothing_factor} \
      --output_root_dir "stat_cognac_app_ctrlgen_multi_constraints_extended_maxlen_${maxlen}"
  done
done
