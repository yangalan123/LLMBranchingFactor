import os.path
import random

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from mmlu_prompt_utils import DATA_DIR, COT_PROMPT_PATH, STANDARD_PROMPT_PATH, load_task_prompts, get_test_prompts
# from consts import ALL_MODELS, CHAT_TEMPLATES_PATH
from uncertainty_quantification.consts import ALL_MODELS, CHAT_TEMPLATES_PATH
from uncertainty_quantification.tokenizer_utils import format_prompt, setup_tokenizer
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMLUArgsParsing.')
    parser.add_argument("--max_constraint_level", type=int, default=5,)
    parser.add_argument("--min_test_sample_per_task", type=int, default=5)
    parser.add_argument("--max_test_sample_per_task", type=int, default=10)
    args = parser.parse_args()
    model_names = ALL_MODELS
    standard_prompt = load_task_prompts(STANDARD_PROMPT_PATH)
    cot_prompt = load_task_prompts(COT_PROMPT_PATH)
    tasks = list(standard_prompt.keys())
    task_instance_to_token_num = dict()
    backup_test_prompts = dict()
    for model_name in tqdm(model_names, desc="Processing models", leave=False):
        print("processing model: {}".format(model_name))
        tokenizer = setup_tokenizer(model_name, CHAT_TEMPLATES_PATH.get(os.path.basename(model_name), None))
        config = AutoConfig.from_pretrained(model_name)
        for task in tasks:
            train_prompt_cot = cot_prompt[task]
            train_prompt_standard = standard_prompt[task]
            # -1 because the first line is the task description
            _constraints_num_cot = len(train_prompt_cot.split("\n\n")) - 1
            _constraints_num_standard = len(train_prompt_standard.split("\n\n")) - 1
            if min(_constraints_num_cot, _constraints_num_standard) < args.max_constraint_level:
                continue
            # cot prompt is longer than standard prompt,
            # so if it won't hit the token limit with cot prompt, it won't hit the token limit with standard prompt
            test_prompts = get_test_prompts(task, train_prompt_cot, tokenizer)
            if task not in backup_test_prompts:
                backup_test_prompts[task] = test_prompts
            for instance_i, test_prompt in enumerate(test_prompts):
                prompt = format_prompt(model_name, test_prompt['prompt'], tokenizer)
                token_ids = tokenizer.encode(prompt)
                token_num = len(token_ids)
                key = (task, instance_i)
                if key not in task_instance_to_token_num:
                    task_instance_to_token_num[key] = []
                if token_num + 512 < config.max_position_embeddings:
                    task_instance_to_token_num[key].append(token_num)

    qualified_tasks = set()
    qualified_task_to_ids = dict()
    for key in task_instance_to_token_num:
        task, instance_i = key
        if len(task_instance_to_token_num[key]) == len(model_names):
            qualified_tasks.add(task)
            if task not in qualified_task_to_ids:
                qualified_task_to_ids[task] = []
            qualified_task_to_ids[task].append(instance_i)
    print("Num of qualified_tasks: {}".format(len(qualified_tasks)))
    all_instances_num = [len(qualified_task_to_ids[task]) for task in qualified_tasks]
    print("Min, max, mean of instances assigned to each task: {}, {}, {}".format(np.min(all_instances_num), np.max(all_instances_num), np.mean(all_instances_num)))
    all_instances_num_single_model = [len(backup_test_prompts[task]) for task in qualified_tasks]
    print("Min, max, mean of instances for each task: {}, {}, {}".format(np.min(all_instances_num_single_model), np.max(all_instances_num_single_model), np.mean(all_instances_num_single_model)))
    sampled_tasks = []
    new_task_to_ids = dict()
    for task in qualified_tasks:
        if len(qualified_task_to_ids[task]) >= args.min_test_sample_per_task:
            assert len(qualified_task_to_ids[task]) <= len(backup_test_prompts[task]), "The number of instances is not correct: qualify: {} <= single_model: {}.".format(len(qualified_task_to_ids[task]), len(backup_test_prompts[task]))
            sampled_tasks.append(task)
            new_task_to_ids[task] = random.sample(qualified_task_to_ids[task], min(len(qualified_task_to_ids[task]), args.max_test_sample_per_task))
            assert np.max(new_task_to_ids[task]) < len(backup_test_prompts[task]), "The instance index is out of range: {} - {}.".format(np.max(new_task_to_ids[task]), len(backup_test_prompts[task]))
    # torch.save([sampled_tasks, new_task_to_ids], "sampled_task.pt")
    # to assist a quick debug run for open-source community, the code below downsample 25 tasks from sampled_tasks, and generate new_task_to_ids
    # through a quick run, we did not find a significant difference for main findings in the paper
    random.seed(1)
    sampled_tasks = random.sample(sampled_tasks, 25)
    task_to_ids = dict()
    for task in sampled_tasks:
        task_to_ids[task] = new_task_to_ids[task]
    torch.save([sampled_tasks, task_to_ids], "sampled_task_small.pt")


