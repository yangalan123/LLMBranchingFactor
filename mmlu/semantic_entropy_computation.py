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
from mmlu_prompt_utils import create_mmlu_data_constraints
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMLUArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=100, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_storywriting")
    parser.add_argument("--constraint_level", type=int, default=0,
                        help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_small.pt", help="task file")
    parser.add_argument("--expand_options", action="store_true", help="expand options into prompts (so single prompt will be copied into multiple prompts, each with a different option)")

    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
    assert constraint_level >= 0, "constraint level should be non-negative"
    tasks, task_to_ids = torch.load(args.task_selection_filename)

    # for model, chat_template_path in models:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed,
        '_task_selection_file_{}'.format(os.path.basename(args.task_selection_filename)) if args.task_selection_filename is not None else "")
    file_name = os.path.join(output_root_dir, file_name)
    manager = ForwardManager(args, ckpt_freq=100)
    spectrum_filename = manager.get_spectrum_filename(file_name)
    assert os.path.exists(spectrum_filename), "spectrum file does not exist: {}".format(spectrum_filename)
    responses = torch.load(spectrum_filename)
    # compute semantic_entropy
    semantic_entropy_filename = manager.get_semantic_entropy_filename(spectrum_filename)
    if os.path.exists(semantic_entropy_filename):
        print("File exists: {}".format(semantic_entropy_filename))
        semantic_entropy = torch.load(semantic_entropy_filename)
        assert len(semantic_entropy) == len(responses), "length mismatch"
    else:
        semantic_entropy = manager.semantic_uncertainty_computation(responses, semantic_entropy_filename)
        torch.save(semantic_entropy, semantic_entropy_filename)

    # manager.fillin_logits_routine(response, file_name, max_num_seqs=4, gpu_memory_utilization=0.8)
