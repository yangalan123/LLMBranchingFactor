# adapted from storytelling/main.py
import argparse
import gc
import os
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from language_modeling.data_utils import get_data_from_huggingface, roles
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from uncertainty_quantification.nudging_find_common_prefix import search_for_common_prefix
from transformers import AutoTokenizer
from loguru import logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StoryWritingArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_storywriting")
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--max_constraint_level", type=int, default=10,)
    parser.add_argument("--dataset_path", type=str, default="Salesforce/wikitext", help="task/dataset path, first argument of datasets.load_dataset")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-v1", help="task/dataset name, second argument of datasets.load_dataset")
    parser.add_argument("--dataset_sample_counts", type=int, default=50, help="sample counts for dataset")
    parser.add_argument("--min_word_count", type=int, default=50, help="minimum word count per instance for dataset")
    parser.add_argument("--min_tokens", type=int, default=0, help="minimum token number per instance")
    parser.add_argument("--word_level_constraint", action="store_true", help="constraint level in word level")
    parser.add_argument("--word_level_constraint_multiplier", type=int, default=10, help="constraint level multiplier in word level")
    parser.add_argument("--input_file", type=str, help='input file', default=None)
    parser.add_argument("--nudging", action="store_true", help="nudging mode")
    parser.add_argument("--nudging_ckpt_path", type=str, default=None, help="nudging ckpt path")
    parser.add_argument("--nudging_model", type=str, default=None, help="nudging model")
    parser.add_argument("--nudging_max_prefix_length", type=int, default=5, help="nudging max prefix length")
    parser.add_argument("--nudging_freq_threshold", type=int, default=50, help="nudging freq threshold")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    # assert constraint_level >= 0, "constraint level should be non-negative"

    # for model, chat_template_path in models:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        "_word_level_constraint_multiplier_{}".format(args.word_level_constraint_multiplier) if args.word_level_constraint else "",
        '_input_file_{}'.format(os.path.basename(args.input_file)) if args.input_file is not None else "",
        '_nudging_{}_{}_{}'.format(os.path.basename(args.nudging_model), args.nudging_max_prefix_length,
                                   args.nudging_freq_threshold) if args.nudging else ""
    )
    file_name = os.path.join(output_root_dir, file_name)
    logger.add(file_name.replace(".pt", ".log"), rotation="10 MB")
    manager = ForwardManager(args, ckpt_freq=args.ckpt_freq, accumulation_batch_size=args.ckpt_freq * 1)
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, roles, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
    else:
        constrained_prompt = []
        all_original_prompts = get_data_from_huggingface(args)
        tokenize_func = sent_tokenize if not args.word_level_constraint else word_tokenize
        _constraint_level = constraint_level * args.word_level_constraint_multiplier if args.word_level_constraint else constraint_level
        _max_constraint_level = args.max_constraint_level * args.word_level_constraint_multiplier if args.word_level_constraint else args.max_constraint_level
        for _prompt in all_original_prompts:
            if isinstance(_prompt, tuple):
                _prompt_text, _model = _prompt
            else:
                _prompt_text = _prompt
            if _max_constraint_level > 0:
                tokens = tokenize_func(_prompt_text)
                if _constraint_level >= 0:
                    tokens = tokens[:_constraint_level]
                tokens = tokens[:_max_constraint_level]
                constrained_prompt.append(" ".join(tokens))
            else:
                constrained_prompt.append(_prompt_text)
        # all_task_prompts = ablate_task_prompt + constrained_prompt
        all_task_prompts = constrained_prompt
        if args.nudging:
            ckpt_path = args.nudging_ckpt_path
            nudging_model = args.nudging_model
            nudging_dict_tree = torch.load(ckpt_path)
            candidate_prefixes = search_for_common_prefix(nudging_dict_tree, model=nudging_model,
                                                          max_prefix_length=args.nudging_max_prefix_length,
                                                          freq_threshold=args.nudging_freq_threshold)
            tokenizer = AutoTokenizer.from_pretrained(nudging_model)
            chosen_idx = 0
            if "8B" in nudging_model:
                chosen_idx = 2
            elif "70B" in nudging_model:
                chosen_idx = 1
            logger.info("candidate_prefixes: {}".format(candidate_prefixes[chosen_idx]))
            generated_prefix = tokenizer.decode(candidate_prefixes[chosen_idx][0])
            logger.info(f"Generated prefix: {generated_prefix}, original prefix: {candidate_prefixes[chosen_idx]}")
        else:
            generated_prefix = ""
        prompts = []
        for _task_prompt in all_task_prompts:
            for role in roles:
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
                prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message) + generated_prefix)
        torch.save((prompts, roles, all_task_prompts, all_original_prompts, args), metadata_filename)
    maybe_exist_flag = False
    response = None
    print("Total prompts: {}".format(len(prompts)))
    gpu_memory_utilization = compute_gpu_memory_utilization(model)
    max_num_seqs = 16
    # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
    # if "70b" in model.lower() and "llama-3" in model.lower():
    if "70b" in model.lower():
        gpu_memory_utilization = 0.55
        max_num_seqs = 32
    if "8b" in model.lower():
        gpu_memory_utilization = 0.35
        max_num_seqs = 32
    manager.setup_model(max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
    # step-1: get original generated story
    if os.path.exists(file_name):
        print("File exists: {}".format(file_name))
        try:
            response = torch.load(file_name)
            assert len(response) == len(prompts), "length mismatch: {} (responses) vs. {} (prompts)".format(len(response), len(prompts))
            maybe_exist_flag = True
        except Exception as e:
            print("File exists but cannot be loaded: {} (Exception: {})".format(file_name, e))
            # if response is already loaded, delete it to save memory as forward manager would load again
            if response is not None:
                del response
                gc.collect()
            maybe_exist_flag = False
    if not maybe_exist_flag:
        response = manager.forward(prompts, file_name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
        torch.save(response, file_name)
    # step-2: get full spectrum of probability
    manager.fillin_logits_routine(response, file_name, max_num_seqs=4, gpu_memory_utilization=0.5)
