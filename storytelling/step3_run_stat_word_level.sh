#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd storytelling

#top_ps=(0.9 0.95)
#offsets=(0 50 100 150 200 250 300 350 400 450 500 550 600)
offsets=(0)
# create an array of offsets rather than manual setup, so that we can easily change the offsets
# start from 0, end at 600, step 50
#offsets=("$(seq 0 50 1000)")
#root_dir="response_storywriting"
#root_dir="response_storywriting_local_story_gen"
root_dir="response_storywriting_local_story_gen_full_word_level_constraint"
ckpt_dirs=("output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9"
            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9_enforce_min_p_0.1")
#            "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95_enforce_min_p_0.1"
#           "output_manual_check_${root_dir}_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95")
high_entropy_thresholds=(0.7)

# get offset_index, ckpt_dir_index, high_entropy_threshold_index
# slurm_array_task_id = offset_index * #(ckpt_dirs) * #(high_entropy_thresholds) + ckpt_dir_index * #(high_entropy_thresholds) + high_entropy_threshold_index
offset_index=$((SLURM_ARRAY_TASK_ID / ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ))
ckpt_dir_index=$(( (SLURM_ARRAY_TASK_ID % ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ) / ${#high_entropy_thresholds[@]} ))
high_entropy_threshold_index=$((SLURM_ARRAY_TASK_ID % ${#high_entropy_thresholds[@]} ))
offset=${offsets[offset_index]}
ckpt_dir=${ckpt_dirs[ckpt_dir_index]}
high_entropy_threshold=${high_entropy_thresholds[high_entropy_threshold_index]}
source_dir="response_storywriting_local_story_gen_full_word_level_constraint"
echo "offset: ${offset}, ckpt_dir: ${ckpt_dir}, high_entropy_threshold: ${high_entropy_threshold}, source_dir: ${source_dir}"


#python stat_storytelling.py \
#--ckpt_dir ${ckpt_dir} \
#--offset ${offset} \
#--source_dir ${source_dir} \
#--high_entropy_threshold ${high_entropy_threshold}

#python ../uncertainty_quantification/stat_computations_from_entropy_profile.py \
# as storytelling features special visualization and analysis (mainly, show the continued writing from other model generated plots),
# we have to use a special script to do the analysis
#python stat_storytelling.py \
#  --constraints "0,1,2,3,4,5" \
#  --output_root_dir "stat_storytelling_app_ctrlgen_multi_constraints" \
#  --no_modelwise_plot \
#  --piecewise_ebf \
#  --ckpt_dir ${ckpt_dir} \
#  --offset ${offset} \
#  --source_dir ${source_dir} \
#  --high_entropy_threshold ${high_entropy_threshold}

#    --no_modelwise_plot \
for smoothing_factor in 0.1 0.5 1.0
do
  python stat_storytelling.py \
    --constraints "0,1,2,3,4,5" \
    --output_root_dir "stat_storytelling_app_ctrlgen_multi_constraints" \
    --piecewise_ebf \
    --smoothing_factor ${smoothing_factor} \
    --ckpt_dir ${ckpt_dir} \
    --offset ${offset} \
    --source_dir ${source_dir} \
    --high_entropy_threshold ${high_entropy_threshold}

  for maxlen in 5 50
  do
    python stat_storytelling.py \
      --constraints "0,1,2,3,4,5" \
      --output_root_dir "stat_storytelling_app_ctrlgen_multi_constraints_maxlen_${maxlen}" \
      --piecewise_ebf \
      --smoothing_factor ${smoothing_factor} \
      --ckpt_dir ${ckpt_dir} \
      --maxlen ${maxlen} \
      --offset ${offset} \
      --source_dir ${source_dir} \
      --high_entropy_threshold ${high_entropy_threshold}
  done

done