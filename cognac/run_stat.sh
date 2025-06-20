ckpt_dir="output_manual_check_cognac_app_ctrlgen_multi_constraints"
python stat_cognac_increasing_ppl.py --ckpt_dir ${ckpt_dir}  | tee stat_cognac_increasing_ppl.log
ckpt_dir="output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9"
python stat_cognac_increasing_ppl.py --ckpt_dir ${ckpt_dir}  | tee stat_cognac_increasing_ppl.top_p_09.log
