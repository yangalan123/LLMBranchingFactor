# adapted from storytelling/main.py
import argparse
import os
import torch
import psutil

from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NudgingArgsParsing.')
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_storywriting")
    # mmlu-specific setup
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_small.pt", help="task file")
    # nudging-specific setup
    parser.add_argument("--eval_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",)
    parser.add_argument("--eval_log_probs", type=int, default=50, help="log probs")
    parser.add_argument("--eval_output_dir", type=str, default="nudging_experiments", help="log probs")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    assert constraint_level >= 0, "constraint level should be non-negative"
    tasks, task_to_ids = torch.load(args.task_selection_filename)

    # Get memory usage of the current process
    process = psutil.Process()
    memory_info = process.memory_info()

    # Print memory usage in bytes
    print(memory_info.rss)

    # Print memory usage in MB
    print("Memory usage in MB:", memory_info.rss / (1024 * 1024))

    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        '_task_selection_file_{}'.format(os.path.basename(args.task_selection_filename)) if args.task_selection_filename is not None else "")
    nudging_output_root_dir = args.eval_output_dir
    os.makedirs(nudging_output_root_dir, exist_ok=True)
    nudging_output_filename = file_name + ".{}_top_{}.nudging".format(os.path.basename(args.eval_model), args.eval_log_probs)
    nudging_output_filename = os.path.join(nudging_output_root_dir, nudging_output_filename)
    file_name = os.path.join(output_root_dir, file_name)
    metadata_filename = file_name.replace(".pt", ".metadata")
    assert os.path.exists(metadata_filename), f"metadata file ({metadata_filename}) not found"
    prompts, roles, answers, options, all_constrained_prompts, metadata_args = torch.load(metadata_filename)
    # nudging part tricks
    # trick: update the log_probs to the eval_log_probs
    args.log_probs = args.eval_log_probs
    args.model = args.eval_model
    eval_model = args.eval_model
    # trick #2: update min_p and top_p -- both params are only used in the sampling process, so we can safely ignore them
    args.min_p = 0
    args.top_p = 1.0
    manager = ForwardManager(args, ckpt_freq=args.ckpt_freq)
    gpu_memory_utilization = compute_gpu_memory_utilization(model)
    max_num_seqs = 16
    if "70b" in eval_model.lower() and "llama-3" in eval_model.lower():
        gpu_memory_utilization = 0.6
        max_num_seqs = 4
    if "8b" in eval_model.lower():
        gpu_memory_utilization = 0.3
        max_num_seqs = 4
    assert os.path.exists(file_name), f"file ({file_name}) not found"
    manager.setup_model(max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
    response = torch.load(file_name)
    print("Memory usage in MB:", memory_info.rss / (1024 * 1024))

    manager.fillin_logits_routine(response, file_name, max_num_seqs=max_num_seqs,
                                  gpu_memory_utilization=gpu_memory_utilization,
                                  try_reuse_existing_logprobs=False,
                                  update_filename=nudging_output_filename + ".spectrum",
                                  patch_filename=nudging_output_filename + ".patch",
                                  not_remove_patch_after_spectrum=True)
