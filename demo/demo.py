# adapted from storytelling/main.py
# implemented the whole pipeline from sampling generation based on prompt, compute entropy + logliks, and then compute BF
import argparse
import gc
import os
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.loglik_computation import (
    get_tokenwise_entropy_from_vllm_outputs, compute_loglik, get_tokenwise_logprob_from_vllm_outputs)
from uncertainty_quantification.uncertainty_computation import compute_bf_values
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from transformers import AutoTokenizer
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
import numpy as np
from uncertainty_quantification.common_utils import condense_arrays_with_inconsistent_lengths

def get_data_from_huggingface(args):
    # default dataset is wikitext-103-v1, as shown below in main argument parser
    if "wikitext" in args.dataset_path.lower():
        # normal wikipedia task
        ds = load_dataset(args.dataset_path, args.dataset_name)
        test_ds = ds['test'].filter(lambda x: len(word_tokenize(x['text'])) > args.min_word_count)
        sample_counts = min(args.dataset_sample_counts, len(test_ds))
        sampled_ds = test_ds.shuffle(seed=args.seed).select(range(sample_counts))
        return [x['text'] for x in sampled_ds]

# from language_modeling/main.py
def step1_run_llm(args):
    model = args.model
    chat_template_path = args.chat_template_path
    constraint_level = args.constraint_level
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
    # setting up the logger
    manager = ForwardManager(args, ckpt_freq=args.ckpt_freq, accumulation_batch_size=args.ckpt_freq * 1)
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
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
        prompts = []
        for _task_prompt in all_task_prompts:
            # if you are not sure what is a system message, just leave it empty
            system_message = ""
            # if you do not want to do teacher forcing, just leave it empty
            generated_prefix = ""
            prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message) + generated_prefix)
        torch.save((prompts, all_task_prompts, all_original_prompts, args), metadata_filename)
    maybe_exist_flag = False
    response = None
    print("Total prompts: {}".format(len(prompts)))
    # gpu_memory_utilization = compute_gpu_memory_utilization(model)
    # after vLLM v1 with better memory management, we can use 0.9 for gpu memory utilization
    gpu_memory_utilization = 0.9
    # max_num_seqs is the maximum number of sequences to generate in parallel, if you come across OOM, you can decrease this number
    max_num_seqs = 256
    # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
    # if "70b" in model.lower() and "llama-3" in model.lower():
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
    return [prompts, all_task_prompts, all_original_prompts, args], response, file_name

# from uncertainty_quantification.loglik_analysis.py
def step2_compute_loglik_and_entropy(database, metadata, args):
    distribution_profile = {"prompt": [], "output": [], "prompt_per_token_logprob": [],
                        "output_per_token_logprob": [], "output_per_token_logprob_truncated": [],
                        "entropy": [],
                        "metadata": metadata}
    prompts = metadata[0]
    for idx in tqdm(range(len(database)), desc=f"Processing database",
                    leave=False):
        data = database[idx]
        prompt = prompts[idx]
        all_outputs = data.outputs
        all_outputs_texts = [x.text.strip() for x in all_outputs]
        if data.prompt_logprobs is None:
            prompt_loglik = None
            prompt_per_token_loglik = None
        else:
            prompt_loglik = compute_loglik(data.prompt_token_ids, data.prompt_logprobs)
            prompt_per_token_loglik = get_tokenwise_logprob_from_vllm_outputs(data.prompt_token_ids,
                                                                                data.prompt_logprobs)
        prompt_loglik_profile = [prompt_loglik, len(data.prompt_token_ids), prompt]
        output_loglik_profiles = [[x.cumulative_logprob, len(x.token_ids), x.text, idx] for x in
                                    all_outputs]
        output_per_token_loglik_profiles = [get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs)
                                            for x in all_outputs]
        output_per_token_loglik_profiles_truncated = [
            get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs, top_p=args.top_p) for x in
            all_outputs]
        entropies = get_tokenwise_entropy_from_vllm_outputs(all_outputs, args.top_p, top_p_mode=True)
        entropies = [x[0] for x in entropies]
        distribution_profile["prompt"].append(prompt_loglik_profile)
        distribution_profile["output"].extend(output_loglik_profiles)
        distribution_profile["prompt_per_token_logprob"].append(prompt_per_token_loglik)
        distribution_profile["output_per_token_logprob"].append(output_per_token_loglik_profiles)
        distribution_profile['output_per_token_logprob_truncated'].append(
            output_per_token_loglik_profiles_truncated)
        distribution_profile['entropy'].append(entropies)
    return distribution_profile

def step3_compute_bf_values(distribution_profile, asymptotic_limit=50):
    output_token_entropies = []
    output_token_logprobs = []
    # for instance in entropy_dict:
    #     entropies = instance[-1]
    #     new_entropies = np.array(condense_arrays_with_inconsistent_lengths(entropies))
    #     output_token_entropies.append(new_entropies.cumsum())

    # # Process loglik values
    # for instance in output_per_token_logprob:
    #     new_logliks = np.array(
    #         condense_arrays_with_inconsistent_lengths(instance, prefix_mode=True, divide_by_length=True))
    #     output_token_logprobs.append(new_logliks)
    for instance in distribution_profile['entropy']:
        new_entropies = np.array(condense_arrays_with_inconsistent_lengths(instance))
        output_token_entropies.append(new_entropies.cumsum())
    for instance in distribution_profile['output_per_token_logprob_truncated']:
        new_logliks = np.array(condense_arrays_with_inconsistent_lengths(instance, prefix_mode=True, divide_by_length=True))
        output_token_logprobs.append(new_logliks)

    # Compute BF values
    bf_values = compute_bf_values(output_token_entropies, output_token_logprobs, asymptotic_limit=asymptotic_limit)
    # return bf_values per prompt, a list of floats, take care of the exponential of the bf values -- numerical overflow might happen
    # you can use np.mean to get overall dataset-level bf value across all prompts
    bf_values = np.exp(bf_values)
    return bf_values, np.mean(bf_values)

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
    logger.add("experiment.log", rotation="10 MB")
    # log all arguments for easy reproduction and experiment tracking
    logger.info("Arguments: {}".format(args))
    logger.info("Starting Step 1: Running LLM")
    metadata, response, file_name = step1_run_llm(args)
    prompts, _, _, args = metadata
    logger.info("Starting Step 2: Computing Loglik and Entropy")
    distribution_profile = step2_compute_loglik_and_entropy(response, metadata, args)
    logger.info("Starting Step 3: Computing BF Values")
    bf_values_per_prompt, overall_bf_value = step3_compute_bf_values(distribution_profile)
    logger.info(f"Overall BF value: {overall_bf_value}")

