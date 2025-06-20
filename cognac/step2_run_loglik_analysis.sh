#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd cognac

models=("meta-llama/Llama-2-70b-chat-hf"
       "meta-llama/Llama-2-13b-chat-hf"
       "meta-llama/Llama-2-13b-hf"
       "meta-llama/Llama-2-70b-hf"
       "meta-llama/Meta-Llama-3-8B"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Meta-Llama-3-70B-Instruct"
        "meta-llama/Meta-Llama-3-70B"
        "ALL"
        )
#top_ps=(0.9 0.95)
top_ps=(0.9)
num_top_ps=${#top_ps[@]}
model_index=$((SLURM_ARRAY_TASK_ID / num_top_ps))
top_p_index=$((SLURM_ARRAY_TASK_ID % num_top_ps))
model=${models[model_index]}
top_p=${top_ps[top_p_index]}
# for latest_eval/bbc_news_all
#source_dir="response_news"
# for cnn/dm
#source_dir="response_cnn_dm_news"
#source_dir="response_storywriting_local_story_gen_full_word_level_constraint"
source_dir="cognac_responses_200"
source_dir="cognac_responses_keywords_mode_2"
for source_dir in "cognac_responses_200" "cognac_responses_keywords_mode_2"
do
    python ../uncertainty_quantification/loglik_analysis.py \
      --source_dir ${source_dir} \
      --top_p ${top_p} \
      --force_recompute \
      --max_tokens 512 \
      --model "${model}"
done

