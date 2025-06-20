import os.path
import random

import torch
import numpy as np
from transformers import AutoConfig
from consts import CHAT_TEMPLATES_PATH
from tqdm import tqdm
from utils import setup_tokenizer
from cognac.cognac_utils import get_cognac_data

if __name__ == '__main__':
    model_names = ("meta-llama/Llama-2-70b-hf",
                   "meta-llama/Llama-2-13b-chat-hf",
                   "meta-llama/Llama-2-13b-hf",
                   "meta-llama/Llama-2-70b-chat-hf",
                   "01-ai/Yi-34B-Chat",
                   "01-ai/Yi-34B",
                   "mistralai/Mixtral-8x7B-v0.1",
                   "mistralai/Mixtral-8x7B-Instruct-v0.1"
                   )
    # all_data = get_cognac_data()
    instance_to_token_num = dict()

    for model_name in tqdm(model_names, desc="Processing models", leave=False):
        print("processing model: {}".format(model_name))
        chat_templates_path = CHAT_TEMPLATES_PATH.get(os.path.basename(model_name), None)
        tokenizer = setup_tokenizer(model_name, chat_templates_path)
        config = AutoConfig.from_pretrained(model_name)
        # cot prompt is longer than standard prompt,
        # so if it won't hit the token limit with cot prompt, it won't hit the token limit with standard prompt
        # train_prompt = cot_prompt[task]
        # test_prompts = get_test_prompts(task, train_prompt, tokenizer)
        # new_args = Namespace(multi_constraint=3)
        # prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data(model_name, chat_templates_path, update_args=new_args)
        prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data(model_name, chat_templates_path)
        # show a sample prompt
        random_idx = random.randint(0, len(prompts))
        print(prompts[random_idx])

        for instance_i, prompt in enumerate(prompts):
            # prompt = format_prompt(model_name, test_prompt['prompt'], tokenizer)
            token_ids = tokenizer.encode(prompt)
            # print(answers[instance_i])
            answer_token_ids = tokenizer.encode(answers[instance_i]['topic'] + " " + answers[instance_i]['constraint'])
            input_token_num = len(token_ids)
            token_num = len(answer_token_ids)
            key = instance_i
            if key not in instance_to_token_num:
                instance_to_token_num[key] = []
            max_len = 1e10
            if hasattr(config, "max_position_embeddings"):
                max_len = config.max_position_embeddings
            if hasattr(config, "seq_len"):
                max_len = config.seq_len
            if input_token_num + 200 < max_len:
                # at most, cognac_origin will only look-ahead 200 tokens, max_gen_length=60 by default
                instance_to_token_num[key].append((input_token_num, token_num))

    # qualified_tasks = set()
    qualified_tasks = ["cognac"]
    qualified_task_to_ids = {"cognac": []}
    token_nums_mean_output = []
    token_nums_mean_input = []
    for key in instance_to_token_num:
        instance_i = key
        if len(instance_to_token_num[key]) == len(model_names):
            qualified_task_to_ids["cognac"].append(instance_i)
            token_nums_mean_output.append(np.mean([x[1] for x in instance_to_token_num[key]]))
            token_nums_mean_input.append(np.mean([x[0] for x in instance_to_token_num[key]]))
    # print("Num of qualified_tasks: {}".format(len(qualified_tasks)))
    all_instances_num = [len(qualified_task_to_ids[task]) for task in qualified_task_to_ids]
    print("Min, max, mean of instances assigned to each task: {}, {}, {}".format(np.min(all_instances_num), np.max(all_instances_num), np.mean(all_instances_num)))
    # all_instances_num_single_model = [len(original_data[task]) for task in qualified_tasks]
    # print("Min, max, mean of instances for each task: {}, {}, {}".format(np.min(all_instances_num_single_model), np.max(all_instances_num_single_model), np.mean(all_instances_num_single_model)))
    print("Min, max, mean of output_token_nums per instance: {}, {}, {}".format(np.min(token_nums_mean_output), np.max(token_nums_mean_output), np.mean(token_nums_mean_output)))
    print("Min, max, mean of input_token_nums per instance: {}, {}, {}".format(np.min(token_nums_mean_input), np.max(token_nums_mean_input), np.mean(token_nums_mean_input)))
    sampled_tasks = []
    new_task_to_ids = dict()
    print("Qualified tasks: {}".format(qualified_tasks))
    sample_num = 1000
    task_instance_to_token_num_sampled = dict()
    token_nums_mean = dict()
    token_nums_mean_input = dict()
    for task in qualified_tasks:
        if len(qualified_task_to_ids[task]) > sample_num:
            assert len(qualified_task_to_ids[task]) <= len(prompts), "The number of instances is not correct: qualify: {} <= single_model: {}.".format(len(qualified_task_to_ids[task]), len(prompts))
            sampled_tasks.append(task)
            #new_task_to_ids[task] = random.sample(qualified_task_to_ids[task], 25)
            new_task_to_ids[task] = random.sample(qualified_task_to_ids[task], sample_num)
            # new_task_to_ids[task] = qualified_task_to_ids[task]
            assert np.max(new_task_to_ids[task]) < len(prompts), "The instance index is out of range: {} - {}.".format(np.max(new_task_to_ids[task]), len(prompts))
            for instance_i in new_task_to_ids[task]:
                key = instance_i
                task_instance_to_token_num_sampled[key] = instance_to_token_num[key]
                if task not in token_nums_mean:
                    token_nums_mean[task] = []
                    token_nums_mean_input[task] = []
                token_nums_mean[task].append(np.mean([x[1] for x in instance_to_token_num[key]]))
                token_nums_mean_input[task].append(np.mean([x[0] for x in instance_to_token_num[key]]))
    print("Num of sampled tasks: {}".format(len(sampled_tasks)))
    for task in token_nums_mean:
        print("Min, max, mean of output_token_nums per instance for task {}: {}, {}, {}".format(task, np.min(token_nums_mean[task]), np.max(token_nums_mean[task]), np.mean(token_nums_mean[task])))
        print("Min, max, mean of input_token_nums per instance for task {}: {}, {}, {}".format(task, np.min(token_nums_mean_input[task]), np.max(token_nums_mean_input[task]), np.mean(token_nums_mean_input[task])))

    # torch.save([sampled_tasks, new_task_to_ids], "sampled_task.pt")
    # to support fast OSS replication, we provide codes to downsample 25 tasks from sampled_tasks, and simply new_task_to_ids
    random.seed(1)
    sampled_tasks = random.sample(sampled_tasks, 25)
    task_to_ids = dict()
    for task in sampled_tasks:
        task_to_ids[task] = new_task_to_ids[task]
    #torch.save([sampled_tasks, task_to_ids], "sampled_task.pt")
    # torch.save([sampled_tasks, task_to_ids], "sampled_task_small_translation.pt")
    torch.save([sampled_tasks, task_to_ids], f"sampled_task_cognac_app_{sample_num}.pt")


