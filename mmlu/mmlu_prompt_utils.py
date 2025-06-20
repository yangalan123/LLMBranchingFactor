# processing mmlu 5-shot cot prompt to be original prompt
# Answer-only /Standard prompting codes copied from chain-of-thought hub: https://github.com/FranxYao/chain-of-thought-hub/blob/main/MMLU/run_mmlu_open_source.py
import random

import pandas as pd
import json
import os

import torch
from transformers import AutoTokenizer

from uncertainty_quantification.nudging_find_common_prefix import search_for_common_prefix
from uncertainty_quantification.consts import DATA_DIR, COT_PROMPT_PATH, STANDARD_PROMPT_PATH, FILLER_COT_PROMPT_PATH, WRONGANSWER_COT_PROMPT_PATH

TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions']

choices = ["A", "B", "C", "D"]
# by default, mmlu is 5-shot prompting (though may need truncation for input window limit)
ntrain = 5


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True, cot=False):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    if not cot:
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    else:
        for j in range(k):
            prompt += "\n({}) {}".format(choices[j], df.iloc[idx, j + 1]) + " "
        prompt += "\nA: Let's think step by step."
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    # [ych]: fix here, the original version has an additional space
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def get_task_specific_original_data(task, train_prompt, cot=False):
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test", task + "_test.csv"), header=None)
    records = []
    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False, cot=cot)
        # why sometimes no \n\n before prompt_end? look at format_example()! it already has a \n\n when we call it with include_answer=True (by default)
        # Chain-of-Thought hub original CoT prompt (chain-of-thought-hub/MMLU/lib_prompt/mmlu-cot.json) for mmlu does not have \n\n at the end
        if train_prompt.endswith("\n\n"):
            prompt = train_prompt + prompt_end
        else:
            prompt = train_prompt + "\n\n" + prompt_end
        label = test_df.iloc[i, test_df.shape[1] - 1]
        options = [" {}\n\n".format(test_df.iloc[i, j]) for j in range(1, test_df.shape[1] - 1)]
        records.append({'prompt': prompt, 'answer': label, "prompt_end": prompt_end, "options": options})
    return records


def get_test_prompts(task, train_prompt, tokenizer):
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test", task + "_test.csv"), header=None)
    records = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        if "Let's think step by step." in train_prompt:
            prompt_end = format_example(test_df, i, include_answer=False, cot=True)
        else:
            prompt_end = format_example(test_df, i, include_answer=False)
        prompt = train_prompt + prompt_end
        while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        records.append({'prompt': prompt, 'answer': label})
    return records

def load_task_prompts(path):
    with open(path, "r") as f_in:
        return json.load(f_in)


# this version has tokenizer chat template applied
def create_mmlu_data(model: str, chat_template_path: str, tasks, task_to_ids, metadata_filename=None):
    standard_prompt = load_task_prompts(STANDARD_PROMPT_PATH)
    cot_prompt = load_task_prompts(COT_PROMPT_PATH)
    filler_cot_prompt = load_task_prompts(FILLER_COT_PROMPT_PATH)
    wrong_answer_cot_prompt = load_task_prompts(WRONGANSWER_COT_PROMPT_PATH)
    prompts = []
    answers = []
    sources = []
    tokenizer = AutoTokenizer.from_pretrained(model)
    if "chat" in model.lower() or "instruct" in model.lower():
        # chat-based model
        if chat_template_path is not None:
            chat_template = open(chat_template_path).read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            tokenizer.chat_template = chat_template
        for task in tasks:
            for prompt_family, prompt_name in zip([standard_prompt, cot_prompt, filler_cot_prompt, wrong_answer_cot_prompt], ["standard", "cot", "filler_cot", "wrong_answer_cot"]):
                records = get_test_prompts(task, prompt_family[task], tokenizer)
                records = [records[i] for i in task_to_ids[task]]
                answers.extend([x['answer'] for x in records])
                for record in records:
                    messages = [
                        {'role': "system",
                         'content': "Follow the given examples and answer the question."},
                        {'role': "user", 'content': record['prompt']}
                    ]
                    prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                sources.extend([prompt_name] * len(records))
    else:
        for task in tasks:
            for prompt_family, prompt_name in zip([standard_prompt, cot_prompt, filler_cot_prompt, wrong_answer_cot_prompt], ["standard", "cot", "filler_cot", "wrong_answer_cot"]):
                records = get_test_prompts(task, prompt_family[task], tokenizer)
                records = [records[i] for i in task_to_ids[task]]
                answers.extend([x['answer'] for x in records])
                prompts.extend([record['prompt'] for record in records])
                sources.extend([prompt_name] * len(records))

    if metadata_filename is not None and not os.path.exists(metadata_filename):
        torch.save([prompts, answers, sources, tasks], metadata_filename)
    del tokenizer
    return prompts, answers, sources, tasks

# this version does not use chat templates -- just keep the original prompt, as we will introduce the constraints in the next step
def create_mmlu_data_constraints(tasks, task_to_ids, constraint_level=-1, prompt_families=None, expand_options=False, expand_options_using_choice=False, nudging=False, nudging_kwargs=None):
    if prompt_families is None:
        standard_prompt = load_task_prompts(STANDARD_PROMPT_PATH)
        cot_prompt = load_task_prompts(COT_PROMPT_PATH)
        filler_cot_prompt = load_task_prompts(FILLER_COT_PROMPT_PATH)
        wrong_answer_cot_prompt = load_task_prompts(WRONGANSWER_COT_PROMPT_PATH)
        prompt_families = {"standard": standard_prompt, "cot": cot_prompt, "filler_cot": filler_cot_prompt, "wrong_answer_cot": wrong_answer_cot_prompt}
    ret_prompts = []
    ret_roles = []
    ret_answers = []
    ret_options = []
    task_to_data = dict()
    generated_prefix = ""
    if nudging:
        ckpt_path = nudging_kwargs["ckpt_path"]
        nudging_dict_tree = torch.load(ckpt_path)
        candidate_prefixes = search_for_common_prefix(nudging_dict_tree, model=nudging_kwargs["model"], max_prefix_length=nudging_kwargs["nudging_max_prefix_length"], freq_threshold=nudging_kwargs["nudging_freq_threshold"])
        tokenizer = AutoTokenizer.from_pretrained(nudging_kwargs["model"])
        logger = nudging_kwargs["logger"]
        logger.info("candidate_prefixes: {}".format(candidate_prefixes[0]))
        generated_prefix = tokenizer.decode(candidate_prefixes[0][0])
        logger.info(f"Generated prefix: {generated_prefix}, original prefix: {candidate_prefixes[0]}")
    for task in tasks:
        task_to_data[task] = dict()
        for prompt_family in prompt_families:
            train_prompt = prompt_families[prompt_family][task]
            if constraint_level >= 0:
                prompt_split = train_prompt.split("\n\n")
                train_prompt = "\n\n".join(prompt_split[:constraint_level + 1])
            records = get_task_specific_original_data(task, train_prompt, cot=("cot" in prompt_family))
            task_to_data[task][prompt_family] = records
        for prompt_family in prompt_families:
            sampled_ids = task_to_ids[task]
            if not expand_options:
                ret_roles.extend(["{}\t{}\t{}".format(prompt_family, task, i) for i in sampled_ids])
                ret_prompts.extend([task_to_data[task][prompt_family][i]["prompt"] + generated_prefix for i in sampled_ids])
                ret_answers.extend([task_to_data[task][prompt_family][i]["answer"] for i in sampled_ids])
                ret_options.extend([task_to_data[task][prompt_family][i]["options"] for i in sampled_ids])
            else:
                for sampled_id in sampled_ids:
                    answer_j = choices.index(task_to_data[task][prompt_family][sampled_id]["answer"])
                    for option_j, option in enumerate(task_to_data[task][prompt_family][sampled_id]["options"]):
                        ret_roles.append("{}\t{}\t{}\t{}\t{}".format(prompt_family, task, sampled_id, option_j, answer_j == option_j))
                        expand_suffix_str = option if not expand_options_using_choice else choices[option_j]
                        ret_prompts.append(task_to_data[task][prompt_family][sampled_id]["prompt"] + generated_prefix + expand_suffix_str)
                        ret_answers.append(task_to_data[task][prompt_family][sampled_id]["answer"])
                        ret_options.append(option)


    return ret_prompts, ret_roles, ret_answers, ret_options




if __name__ == '__main__':
    ret = {}
    for task in TASKS:
        # 5-shot prompting by default
        train_df = pd.read_csv(f"{DATA_DIR}/dev/{task}_dev.csv", header=None)[:ntrain]
        prompt = gen_prompt(train_df, task, ntrain)
        ret[task] = prompt
    with open(STANDARD_PROMPT_PATH, "w") as f_out:
        json.dump(ret, f_out, indent=4)


