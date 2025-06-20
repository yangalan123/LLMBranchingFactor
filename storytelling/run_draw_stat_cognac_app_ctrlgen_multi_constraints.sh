#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd storytelling

#top_ps=(0.9 0.95)
#offsets=(0 50 100 150 200 250)
root_dir="stat_storytelling_app_ctrlgen_multi_constraints"
ckpt_dirs=("output_manual_check_response_storywriting_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9"
            "output_manual_check_response_storywriting_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9_enforce_min_p_0.1")
#            "output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95_enforce_min_p_0.1"
#           "output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95")
#high_entropy_thresholds=(0.7 1.0 2.0)
high_entropy_threshold=0.7

# get offset_index, ckpt_dir_index, high_entropy_threshold_index
# slurm_array_task_id = offset_index * #(ckpt_dirs) * #(high_entropy_thresholds) + ckpt_dir_index * #(high_entropy_thresholds) + high_entropy_threshold_index
for ckpt_dir in "${ckpt_dirs[@]}"
do
  echo "ckpt_dir: ${ckpt_dir}, high_entropy_threshold: ${high_entropy_threshold}"
  python draw_stat_cognac_app_ctrlgen.py \
    --stat_dir "${root_dir}/${ckpt_dir}" \
    --high_entropy_threshold "${high_entropy_threshold}"
done
