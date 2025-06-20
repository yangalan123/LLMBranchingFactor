#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac

offsets=(0 50 100 150 200 250 300 350 400 450 500)
maxlens=(50 100 150 200 250 300 350 400 450 500)
ckpt_dirs=("output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9"
            "output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9_enforce_min_p_0.1"
            "output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95_enforce_min_p_0.1"
           "output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.95")
#high_entropy_thresholds=(0.7 1.0 2.0)
#high_entropy_thresholds=(0.7 1.0)
high_entropy_thresholds=(0.7)

# get offset_index, ckpt_dir_index, high_entropy_threshold_index
# slurm_array_task_id = offset_index * #(ckpt_dirs) * #(high_entropy_thresholds) + ckpt_dir_index * #(high_entropy_thresholds) + high_entropy_threshold_index
offset_index=$((SLURM_ARRAY_TASK_ID / ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ))
ckpt_dir_index=$(( (SLURM_ARRAY_TASK_ID % ( ${#ckpt_dirs[@]} * ${#high_entropy_thresholds[@]} ) ) / ${#high_entropy_thresholds[@]} ))
high_entropy_threshold_index=$((SLURM_ARRAY_TASK_ID % ${#high_entropy_thresholds[@]} ))
offset=${offsets[offset_index]}
maxlen=${maxlens[offset_index]}
ckpt_dir=${ckpt_dirs[ckpt_dir_index]}
high_entropy_threshold=${high_entropy_thresholds[high_entropy_threshold_index]}
echo "offset: ${offset}, ckpt_dir: ${ckpt_dir}, high_entropy_threshold: ${high_entropy_threshold}"


#--offset ${offset} \
python stat_cognac_increasing_ppl.py \
--ckpt_dir ${ckpt_dir} \
--maxlen ${maxlen} \
--high_entropy_threshold ${high_entropy_threshold}
