import copy
import os
import random

import torch

from vllm import LLM, SamplingParams
from cognac_utils import get_hierarchy
from uncertainty_quantification.consts import CHAT_TEMPLATES_PATH
from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt

def solve_node2level(hierarchy, node2level, current_node):
    # assert current_node not in node2level, f"Node {current_node} already solved"
    if current_node in node2level:
        return node2level[current_node]

    if current_node not in hierarchy:
        node2level[current_node] = 0
        return 0

    cur_best = 0
    for child in hierarchy[current_node]:
        cur_best = max(cur_best, solve_node2level(hierarchy, node2level, child))
    node2level[current_node] = cur_best + 1
    return cur_best + 1

def random_walk(current_node, remaining_steps, node2level, hierarchy):
    if remaining_steps == 0:
        return current_node
    if current_node not in hierarchy:
        return current_node
    if node2level[current_node] == 0:
        return current_node
    next_node = random.choice(hierarchy[current_node])
    return random_walk(next_node, remaining_steps - 1, node2level, hierarchy)

def compose_prompts_from_keywords(keywords, model, tokenizer):
    system_prompt = ("You will be given a list of keywords, please generate a story using the keywords as the main theme of the story.\n"
                     "The keywords must occur in the given order in the story.\n"
                     "The story should be given in bullet points. Each bullet point should start with * and should be seperated by two newlines. You should use [END] to mark the end the bullet point generation.\n\n")
    user_prompt = "Keywords: " + ", ".join(keywords) + "\n\n"
    return format_prompt(model, user_prompt, tokenizer, system_message=system_prompt)

def generate_mode1_keywords(sample_num, topics, hierarchy, node2level, max_exploration_steps):
    mode1_all_keywords = []
    for data_i in range(sample_num):
        current_keywords = copy.deepcopy(topics)
        distributed_random_walk_times = []
        current_counter = max_exploration_steps
        for topic_j in range(len(topics)):
            if current_counter <= 0:
                distributed_random_walk_times.append(0)
                continue
            # randomly sample an integer from 1 to current_counter
            random_walk_times = random.randint(1, current_counter)
            distributed_random_walk_times.append(random_walk_times)
            current_counter -= random_walk_times
        sampled_keywords = []
        for topic_j in range(len(topics)):
            sampled_keywords.append(random_walk(current_keywords[topic_j], distributed_random_walk_times[topic_j], node2level, hierarchy))
        mode1_all_keywords.append(sampled_keywords)
    return mode1_all_keywords

def generate_mode2_keywords(sample_num, topics, hierarchy, node2level, min_exploration_steps):
    mode2_all_keywords = []
    while len(mode2_all_keywords) < sample_num:
        # randomly select a topic
        topic = random.choice(topics)
        current_keywords = [topic]
        cur_node = topic
        for _ in range(min_exploration_steps - 1):
            # next step must be chosen from the children with at least node2level[child] >= min_exploration_steps - len(current_keywords)
            next_candidates = [child for child in hierarchy[cur_node] if
                               node2level[child] >= min_exploration_steps - len(current_keywords)]
            if len(next_candidates) == 0:
                break
            cur_node = random.choice(next_candidates)
            current_keywords.append(cur_node)
        if len(current_keywords) >= min_exploration_steps:
            mode2_all_keywords.append(copy.deepcopy(current_keywords))
    return mode2_all_keywords


if __name__ == '__main__':
    hierarchy = get_hierarchy("cognac_origin/wordnet/topic_to_leafs.json")
    topics = ['animal', "vehicle", "art", "food", "sport"]
    sample_per_mode = 200
    max_exploration_steps = 20
    min_exploration_steps = 5
    # mode 1: random walking through different categories
    # mode 2: dive deep into a specific category
    # mode 3: combine mode 1 and mode 2
    # implementing mode 1
    random.seed(42)
    node2level = dict()
    for topic_i in topics:
        solve_node2level(hierarchy, node2level, topic_i)

    keyword_dir = "cognac_keywords_experiments"
    os.makedirs(keyword_dir, exist_ok=True)
    keyword_ckpt_name = os.path.join(keyword_dir, "sampled_keywords_mode1.pt")
    if os.path.exists(keyword_ckpt_name):
        mode1_all_keywords = torch.load(keyword_ckpt_name)
        print("Loaded mode1 keywords from checkpoint")
    else:
        mode1_all_keywords = generate_mode1_keywords(sample_per_mode, topics, hierarchy, node2level, max_exploration_steps)
        torch.save(mode1_all_keywords, keyword_ckpt_name)

    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer = setup_tokenizer(model, CHAT_TEMPLATES_PATH.get(model, None))
    mode1_prompts = [compose_prompts_from_keywords(keywords, model, tokenizer) for keywords in mode1_all_keywords]
    _param = SamplingParams(n=1, max_tokens=1024, top_p=0.9, seed=42)
    llm = LLM(model, tensor_parallel_size=4, max_num_seqs=32, gpu_memory_utilization=0.8)
    responses = llm.generate(mode1_prompts, _param)
    torch.save(responses, os.path.join(keyword_dir, "mode1_responses.pt"))
    # print a few sample of outputs
    for i in range(1):
        sample_i = random.randint(0, sample_per_mode - 1)
        print(f"Sample {sample_i}:")
        print(mode1_prompts[sample_i])
        print(responses[sample_i].outputs[0].text)

    print("Mode 1 done")
    # implementing mode 2
    keyword_ckpt_name = os.path.join(keyword_dir, "sampled_keywords_mode2.pt")
    if os.path.exists(keyword_ckpt_name):
        mode2_all_keywords = torch.load(keyword_ckpt_name)
        print("Loaded mode2 keywords from checkpoint")
    else:
        mode2_all_keywords = generate_mode2_keywords(sample_per_mode, topics, hierarchy, node2level, min_exploration_steps)
        torch.save(mode2_all_keywords, keyword_ckpt_name)

    mode2_prompts = [compose_prompts_from_keywords(keywords, model, tokenizer) for keywords in mode2_all_keywords]
    responses = llm.generate(mode2_prompts, _param)
    torch.save(responses, os.path.join(keyword_dir, "mode2_responses.pt"))
    # print a few sample of outputs
    for i in range(1):
        sample_i = random.randint(0, sample_per_mode - 1)
        print(f"Sample {sample_i}:")
        print(mode2_prompts[sample_i])
        print(responses[sample_i].outputs[0].text)
    print("Mode 2 done")
    # implementing mode 3
    keyword_ckpt_name = os.path.join(keyword_dir, "sampled_keywords_mode3.pt")
    if os.path.exists(keyword_ckpt_name):
        mode3_all_keywords = torch.load(keyword_ckpt_name)
        print("Loaded mode3 keywords from checkpoint")
    else:
        mode3_all_keywords = []
        # generate a new batch of mode1 and mode2 keywords
        mode1_all_keywords = generate_mode1_keywords(sample_per_mode, topics, hierarchy, node2level, max_exploration_steps)
        mode2_all_keywords = generate_mode2_keywords(sample_per_mode, topics, hierarchy, node2level, min_exploration_steps)
        for i in range(sample_per_mode):
            candidates = mode1_all_keywords[i] + mode2_all_keywords[i]
            random.shuffle(candidates)
            mode3_all_keywords.append(candidates[:random.randint(min_exploration_steps, len(candidates))])
        torch.save(mode3_all_keywords, keyword_ckpt_name)

    mode3_prompts = [compose_prompts_from_keywords(keywords, model, tokenizer) for keywords in mode3_all_keywords]
    responses = llm.generate(mode3_prompts, _param)
    torch.save(responses, os.path.join(keyword_dir, "mode3_responses.pt"))
    # print a few sample of outputs
    for i in range(1):
        sample_i = random.randint(0, sample_per_mode - 1)
        print(f"Sample {sample_i}:")
        print(mode3_prompts[sample_i])
        print(responses[sample_i].outputs[0].text)






