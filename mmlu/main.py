import argparse
import os

import torch

from mmlu_prompt_utils import create_mmlu_data_constraints
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from loguru import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMLUArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0,
                                temperature=1.0, output_root_dir="response_storywriting")
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_small.pt", help="task file")
    parser.add_argument("--expand_options", action="store_true",
                        help="expand options into prompts (so single prompt will be copied into multiple prompts, each with a different option)")
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
    assert constraint_level >= 0, "constraint level should be non-negative"
    tasks, task_to_ids = torch.load(args.task_selection_filename)
    args.enable_chunked_prefill = True
    args.max_num_batched_tokens = 256


    # for model, chat_template_path in models:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        '_task_selection_file_{}'.format(
            os.path.basename(args.task_selection_filename)) if args.task_selection_filename is not None else "",
        '_nudging_{}_{}_{}'.format(os.path.basename(args.nudging_model), args.nudging_max_prefix_length,
                                   args.nudging_freq_threshold) if args.nudging else "")
    file_name = os.path.join(output_root_dir, file_name)
    logger.add(file_name.replace(".pt", ".log"), rotation="10 MB")
    manager = ForwardManager(args, ckpt_freq=args.ckpt_freq, accumulation_batch_size=args.ckpt_freq * 3)
    # step-1: get original generated story
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, roles, answers, options, all_constrained_prompts, args = torch.load(metadata_filename)
    else:
        all_constrained_prompts, roles, answers, options = create_mmlu_data_constraints(
            tasks, task_to_ids,
            constraint_level,
            expand_options=args.expand_options,
            nudging=args.nudging,
            nudging_kwargs={
                "ckpt_path": args.nudging_ckpt_path,
                "model": args.nudging_model,
                "nudging_max_prefix_length": args.nudging_max_prefix_length,
                "nudging_freq_threshold": args.nudging_freq_threshold,
                "logger": logger
            }
        )
        prompts = []
        # sys msg from: https://github.com/FranxYao/chain-of-thought-hub/blob/main/MMLU/run_mmlu_gpt_3.5_turbo.py
        system_message = "Follow the given examples and answer the question."
        for constrained_prompt in all_constrained_prompts:
            prompt = format_prompt(model, constrained_prompt, tokenizer, system_message)
            prompts.append(prompt)
        torch.save((prompts, roles, answers, options, all_constrained_prompts, args), metadata_filename)

    token_counts = []
    for prompt in prompts:
        token_counts.append(len(tokenizer.encode(prompt)))
    logger.info("max tokens: {}, min tokens: {}, avg tokens: {}".format(max(token_counts), min(token_counts),
                                                                  sum(token_counts) / len(token_counts)))

    maybe_exist_flag = False
    response = None
    if not maybe_exist_flag:
        gpu_memory_utilization = compute_gpu_memory_utilization(model)
        # response = manager.forward(prompts, file_name, max_num_seqs=4 if "llama-3" in model.lower() else 8, gpu_memory_utilization=gpu_memory_utilization)
        max_num_seqs = 16
        # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
        if "70b" in model.lower() and "llama-3" in model.lower():
            gpu_memory_utilization = 0.55
            max_num_seqs = 128
        if "8b" in model.lower():
            gpu_memory_utilization = 0.3
            max_num_seqs = 128  # a100
        response = manager.forward(prompts, file_name, max_num_seqs=max_num_seqs,
                                   gpu_memory_utilization=gpu_memory_utilization)
        # oom as well (10 logprobs, a40 * 4)
        # response = manager.forward(prompts, file_name, max_num_seqs=256 if "llama-3" in model.lower() else 8, gpu_memory_utilization=0.98)
        # debugging llama-3-8b -- below oom (10 logprobs, a40 * 4)
        # response = manager.forward(prompts, file_name, gpu_memory_utilization=0.98)
        torch.save(response, file_name)
    # step-2: get full spectrum of probability
    # if not args.expand_option:
    #     manager.fillin_logits_routine(response, file_name, max_num_seqs=4, gpu_memory_utilization=0.8)
    # no, actually as fillin_logits_routine by default will check whether there the obtained logits are none,
    # and if so, it would just save the response as-is, so we can just keep this part to conform with the codes in other tasks
    # however, we do need to check whether prompt probs are available
    if not args.expand_options:
        manager.fillin_logits_routine(response, file_name, max_num_seqs=8, gpu_memory_utilization=0.98)
    else:
        manager.fillin_logits_routine(response, file_name, max_num_seqs=8, gpu_memory_utilization=0.98,
                                      check_prompt_probs=True)
