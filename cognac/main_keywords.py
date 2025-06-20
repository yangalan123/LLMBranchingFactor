# adapted from storytelling/main.py
import argparse
import gc
from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams
from nltk.tokenize import sent_tokenize, word_tokenize

from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.model_utils import configure_model
# from language_modeling.data_utils import get_data_from_huggingface, roles
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CognacKeywordPlanningArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_storywriting")
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--dataset_sample_counts", type=int, default=50, help="sample counts for dataset")
    parser.add_argument("--input_file", type=str, help='input file', default=None)
    parser.add_argument("--input_keyword_file", type=str, help='input file', default=None)
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    assert constraint_level >= 0, "constraint level should be non-negative"

    # for model, chat_template_path in models:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        '_input_file_{}'.format(os.path.basename(args.input_file)) if args.input_file is not None else "")
    file_name = os.path.join(output_root_dir, file_name)
    manager = ForwardManager(args, ckpt_freq=128)
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
    else:
        all_original_prompts = []
        all_task_prompts = []
        keywords = torch.load(args.input_keyword_file)
        bullet_point_responses = torch.load(args.input_file)
        for keyword_list, bullet_point_response in zip(keywords, bullet_point_responses):
            if len(all_task_prompts) >= args.dataset_sample_counts:
                break
            keyword_num = len(keyword_list)
            output_text = bullet_point_response.outputs[0].text
            if "[END]" not in output_text or "*" not in output_text:
                continue
            output_text = output_text.split("[END]")[0]
            output_text.replace("**", "")
            output_text_segments = output_text.split("\n\n")
            bullet_points = [x[1:].strip() for x in output_text_segments if x.startswith("*")]
            try:
                assert len(bullet_points) == keyword_num, "length mismatch: bullet_points={} vs keyword_num={}".format(len(bullet_points), keyword_num)
                # check whether every keyword is in bullet points
                for keyword_i in range(keyword_num):
                    assert keyword_list[keyword_i] in bullet_points[keyword_i], "keyword {} not in bullet points {}".format(keyword_list[keyword_i], bullet_points[keyword_i])
                all_original_prompts.append((keyword_list, bullet_points))
                if constraint_level == 0:
                    all_task_prompts.append([""])
                else:
                    all_task_prompts.append(bullet_points[:constraint_level])
            except AssertionError as e:
                continue

        prompts = []
        for _task_prompt_list in all_task_prompts:
            # for language modeling experiments, we will stick to empty system message
            system_message = ""
            # if len(role) == 0:
            #     # empty role, no need to add system message
            #     system_message = ""
            # else:
            #     if role.startswith("#"):
            #         system_message = role.strip("#")
            #     else:
            #         system_message = "Now let's play a role-playing game. Please never share your identity in your output. For example, it is forbidden to say 'As a ...' or 'I am a ...'. Now Pretending " + role
            _task_prompt = " ".join(_task_prompt_list)
            prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message))
        torch.save((prompts, all_task_prompts, all_original_prompts, args), metadata_filename)
    maybe_exist_flag = False
    response = None
    # step-1: get original generated story
    if os.path.exists(file_name):
        print("File exists: {}".format(file_name))
        try:
            response = torch.load(file_name)
            assert len(response) == len(prompts), "length mismatch"
            maybe_exist_flag = True
        except Exception as e:
            print("File exists but cannot be loaded: {} (Exception: {})".format(file_name, e))
            # if response is already loaded, delete it to save memory as forward manager would load again
            if response is not None:
                del response
                gc.collect()
            maybe_exist_flag = False
    if not maybe_exist_flag:
        gpu_memory_utilization = compute_gpu_memory_utilization(model)
        max_num_seqs = 32
        # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
        if "70b" in model.lower() and "llama-3" in model.lower():
            gpu_memory_utilization = 0.5
            max_num_seqs = 64
        if "8b" in model.lower():
            gpu_memory_utilization = 0.3
            max_num_seqs = 128
        response = manager.forward(prompts, file_name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
        torch.save(response, file_name)
        if "llama-3" in model.lower():
            print("Llama-3: have to restart to avoid weird problems from vllm hanging (0.4.3 and 0.5.4), please relaunch the script")
            exit()
    # step-2: get full spectrum of probability
    manager.fillin_logits_routine(response, file_name, max_num_seqs=4, gpu_memory_utilization=0.5)
