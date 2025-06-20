#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac
# original step 2 - entropy profile generation for cognac
# task selection file is actually useless as we focus on normal prompting for many experiments, no more CoT
# but actually, think about this -- for entropy profile generation, it should not know whether the prompt is CoT or not
# in fact, it should be agnostic to the prompt, even the task

models=("meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-70b-hf"
       "mistralai/Mixtral-8x7B-v0.1"
       "mistralai/Mixtral-8x7B-Instruct-v0.1")
top_ps=(0.9 0.95)
num_top_ps=${#top_ps[@]}
model_index=$((SLURM_ARRAY_TASK_ID / num_top_ps))
top_p_index=$((SLURM_ARRAY_TASK_ID % num_top_ps))
model=${models[model_index]}
top_p=${top_ps[top_p_index]}
#for source_dir in $(find cognac_responses -mindepth 1 -type d)
#do
#  echo "source_dir: ${source_dir}, model: ${model}"
#  --task_selection_filename "sampled_task_cognac_app_1000.pt" \
#python Investigate_Cognac_Increasing_PPL.py \
python investigate_cognac_increasing_ppl.py \
  --task_selection_filename "sampled_task_cognac_app_200.pt" \
  --source_dir "cognac_responses_200" \
  --top_p ${top_p} \
  --max_tokens 512 \
  --enforce_min_p \
  --model "${model}"
#done
#done