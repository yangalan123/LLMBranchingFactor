import argparse
import gc
from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams
from nltk.tokenize import sent_tokenize, word_tokenize

from storytelling.consts import task_prompt, roles, ablate_task_prompt
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StoryWritingArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, chat_template_path=None, output_root_dir="response_storywriting")
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--word_level_constraint", action="store_true", help="constraint level in word level")
    parser.add_argument("--word_level_constraint_multiplier", type=int, default=25, help="constraint level multiplier in word level")
    parser.add_argument("--max_constraint_level", type=int, default=10,)
    parser.add_argument("--input_file", type=str, help='input file', default=None)
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    assert constraint_level >= 0, "constraint level should be non-negative"

    # for model, chat_template_path in models:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        "_word_level_constraint_multiplier_{}".format(args.word_level_constraint_multiplier) if args.word_level_constraint else "",
        '_input_file_{}'.format(os.path.basename(args.input_file)) if args.input_file is not None else "")
    file_name = os.path.join(output_root_dir, file_name)
    manager = ForwardManager(args, ckpt_freq=128)
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, roles, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
    else:
        constrained_prompt = []
        if args.input_file is not None:
            plot_id_to_story, all_plot_id_to_story, input_file_args = torch.load(args.input_file)
            all_original_prompts = []
            model_families = None
            for plot_id in plot_id_to_story:
                if model_families is None:
                    model_families = list(plot_id_to_story[plot_id].keys())
                for model_family in model_families:
                    all_original_prompts.append((plot_id_to_story[plot_id][model_family], model_family))
        else:
            all_original_prompts = ablate_task_prompt + task_prompt
        tokenize_func = sent_tokenize if not args.word_level_constraint else word_tokenize
        _constraint_level = constraint_level * args.word_level_constraint_multiplier if args.word_level_constraint else constraint_level
        _max_constraint_level = args.max_constraint_level * args.word_level_constraint_multiplier if args.word_level_constraint else args.max_constraint_level
        for _prompt in all_original_prompts:
            if isinstance(_prompt, tuple):
                _prompt_text, _model = _prompt
            else:
                _prompt_text = _prompt
            tokens = tokenize_func(_prompt_text)
            if _constraint_level >= 0:
                tokens = tokens[:_constraint_level]
            tokens = tokens[:_max_constraint_level]
            constrained_prompt.append(" ".join(tokens))
        # all_task_prompts = ablate_task_prompt + constrained_prompt
        all_task_prompts = constrained_prompt
        prompts = []
        for _task_prompt in all_task_prompts:
            for role in roles:
                if len(role) == 0:
                    # empty role, no need to add system message
                    system_message = ""
                else:
                    if role.startswith("#"):
                        system_message = role.strip("#")
                    else:
                        system_message = "Now let's play a role-playing game. Please never share your identity in your output. For example, it is forbidden to say 'As a ...' or 'I am a ...'. Now Pretending " + role
                prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message))
        torch.save((prompts, roles, all_task_prompts, all_original_prompts, args), metadata_filename)
    # step-1: get original generated story
    maybe_exist_flag = False
    response = None
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
        max_num_seqs = 128
        # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
        if "70b" in model.lower() and "llama-2" in model.lower():
            gpu_memory_utilization = 0.8
            max_num_seqs = 64
        if "70b" in model.lower() and "llama-3" in model.lower():
            gpu_memory_utilization = 0.5
            max_num_seqs = 128
        if "8b" in model.lower():
            gpu_memory_utilization = 0.3
            max_num_seqs = 128

        response = manager.forward(prompts, file_name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
        torch.save(response, file_name)
        if "llama-3" in model.lower():
            print("Llama-3: have to restart to avoid weird problems from vllm hanging (0.4.3 and 0.5.4), please relaunch the script")
            exit()

    # step-2: get full spectrum of probability
    manager.fillin_logits_routine(response, file_name, max_num_seqs=1, gpu_memory_utilization=0.99)


