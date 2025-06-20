import argparse
import os
import torch

from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.nudging_top_k_computation import compare_original_response_and_patch_response
from loguru import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NudgingArgsParsing.')
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_storywriting")
    # mmlu-specific setup
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--max_constraint_level", type=int, default=10,)
    parser.add_argument("--dataset_path", type=str, default="Salesforce/wikitext", help="task/dataset path, first argument of datasets.load_dataset")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-v1", help="task/dataset name, second argument of datasets.load_dataset")
    parser.add_argument("--dataset_sample_counts", type=int, default=50, help="sample counts for dataset")
    parser.add_argument("--min_word_count", type=int, default=50, help="minimum word count per instance for dataset")
    parser.add_argument("--word_level_constraint", action="store_true", help="constraint level in word level")
    parser.add_argument("--word_level_constraint_multiplier", type=int, default=10, help="constraint level multiplier in word level")
    parser.add_argument("--input_file", type=str, help='input file', default=None)
    # nudging-specific setup
    parser.add_argument("--eval_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",)
    parser.add_argument("--eval_log_probs", type=int, default=50, help="log probs")
    parser.add_argument("--eval_output_dir", type=str, default="nudging_experiments", help="log probs")
    parser.add_argument("--force_recompute", action="store_true", help="force recompute")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    # assert constraint_level >= 0, "constraint level should be non-negative"
    # tasks, task_to_ids = torch.load(args.task_selection_filename)

    # Get memory usage of the current process
    # process = psutil.Process()
    # memory_info = process.memory_info()
    #
    # # Print memory usage in bytes
    # print(memory_info.rss)
    #
    # # Print memory usage in MB
    # print("Memory usage in MB:", memory_info.rss / (1024 * 1024))
    #
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        "_word_level_constraint_multiplier_{}".format(args.word_level_constraint_multiplier) if args.word_level_constraint else "",
        '_input_file_{}'.format(os.path.basename(args.input_file)) if args.input_file is not None else "")
    nudging_output_root_dir = args.eval_output_dir
    os.makedirs(nudging_output_root_dir, exist_ok=True)
    nudging_output_filename = file_name + ".{}_top_{}.nudging".format(os.path.basename(args.eval_model), args.eval_log_probs)
    logger.add(os.path.join(nudging_output_root_dir, "{}.log_topK".format(nudging_output_filename)), rotation="10 MB")
    logger.info("Running Args: {}".format(args))
    nudging_output_filename = os.path.join(nudging_output_root_dir, nudging_output_filename)
    file_name = os.path.join(output_root_dir, file_name)
    metadata_filename = file_name.replace(".pt", ".metadata")
    assert os.path.exists(metadata_filename), f"metadata file ({metadata_filename}) not found"
    prompts, roles, all_task_prompts, all_original_prompts, metadata_args = torch.load(metadata_filename)
    logger.info("Metadata Args: {}".format(metadata_args))
    args.log_probs = args.eval_log_probs
    eval_model = args.eval_model
    assert os.path.exists(file_name), f"file ({file_name}) not found"
    original_responses = torch.load(file_name)
    patch_filename = nudging_output_filename + ".patch"
    if os.path.exists(patch_filename):
        logger.info(f"patch file ({patch_filename}) found")
        patch_exist_flag = True
        patch_data = torch.load(patch_filename)
    else:
        logger.info(f"patch file ({patch_filename}) not found, try to load the spectrum file")
        patch_exist_flag = False
        spectrum_filename = nudging_output_filename + ".spectrum"
        assert os.path.exists(spectrum_filename), f"spectrum file ({spectrum_filename}) not found"
        patch_data = torch.load(spectrum_filename)
    # assert os.path.exists(patch_filename), f"patch file ({patch_filename}) not found"

    compare_original_response_and_patch_response(nudging_output_root_dir, original_responses, patch_data, patch_exist_flag, logger, args)



