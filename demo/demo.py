# adapted from storytelling/main.py
# implemented the whole pipeline from sampling generation based on prompt, compute entropy + logliks, and then compute BF
import argparse
import gc
import os
import torch
import re
import glob
from collections import defaultdict

from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.loglik_computation import (
    get_tokenwise_entropy_from_vllm_outputs, compute_loglik, get_tokenwise_logprob_from_vllm_outputs)
from uncertainty_quantification.uncertainty_computation import compute_bf_values
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm
import numpy as np
from uncertainty_quantification.common_utils import condense_arrays_with_inconsistent_lengths
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from uncertainty_quantification.visualization_utils import (
    matplotlib_plot, 
    model_name_visualization_name_mapping, 
    ebf_name_visualization_name_mapping
)
from data import (
    get_data, get_mmlu_data, get_storytelling_data, get_cognac_data_wrapper,
    apply_constraints_language_modeling_storytelling, process_prompts_with_roles
)

def extract_arg_value(arg_string, arg_name):
    pattern = rf'{arg_name}_(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, arg_string)
    if match:
        return match.group(1)
    return None

def run_plotting_mode(args):
    logger.info("Running in plotting mode...")
    output_root_dir = args.output_root_dir
    visualization_dir = os.path.join(output_root_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Pattern to find BF files
    # We look for files matching the general pattern but with varying constraint levels
    # Pattern: {model}_response_n_{sample_counts}_max_tokens_{max_tokens}_log_probs_{log_probs}_min_p_{min_p}_top_p_{top_p}_seed{seed}*_bf.pt
    
    file_pattern_parts = [
        os.path.basename(args.model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed
    ]
    base_pattern = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}".format(*file_pattern_parts)
    
    # Search recursively for files
    search_pattern = os.path.join(output_root_dir, "**", f"{base_pattern}*_bf.pt")
    
    logger.info(f"Searching for files with pattern: {search_pattern}")
    files = glob.glob(search_pattern, recursive=True)
    logger.info(f"Found {len(files)} files.")
    
    if not files:
        logger.warning("No files found. Exiting plotting mode.")
        return

    # Structure: yvalues_records[model_name][tag] = [val1, val2, ...] corresponding to sorted constraints
    
    constraint_value_pairs = []
    
    for file_path in files:
        # Extract constraint level from directory name
        # Expected structure: .../application_ctrlgen_multi_constraints_{level}/...
        dir_name = os.path.dirname(file_path)
        match = re.search(r'application_ctrlgen_multi_constraints_(\d+)', dir_name)
        
        if match:
            constraint_level = int(match.group(1))
        else:
            # Fallback: try to extract from filename if present (legacy support)
            file_name = os.path.basename(file_path)
            constraint_level_str = extract_arg_value(file_name, "constraint_level")
            if constraint_level_str is None:
                constraint_level_str = extract_arg_value(file_name, "word_level_constraint_multiplier")
            
            if constraint_level_str:
                constraint_level = int(float(constraint_level_str))
            else:
                logger.warning(f"Could not extract constraint level from path: {file_path}")
                continue
        
        try:
            # Load the BF file: [bf_values_per_prompt, overall_bf_value]
            data = torch.load(file_path)
            overall_bf = data[1] # Mean BF
            constraint_value_pairs.append((constraint_level, overall_bf))
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
            
    if not constraint_value_pairs:
        logger.warning("No valid data extracted.")
        return

    # Sort by constraint level
    constraint_value_pairs.sort(key=lambda x: x[0])
    constraints = [x[0] for x in constraint_value_pairs]
    bf_values = [x[1] for x in constraint_value_pairs]
    
    logger.info(f"Constraints: {constraints}")
    logger.info(f"BF Values: {bf_values}")
    
    model_name = os.path.basename(args.model)
    vis_model_name = model_name_visualization_name_mapping(model_name)
    
    # We'll use 'ebf_perplexity' tag which maps to 'BF' in ebf_name_visualization_name_mapping
    tag = "ebf_perplexity" 
    
    yvalues_records = {
        vis_model_name: {
            tag: bf_values
        }
    }
    
    save_path = os.path.join(visualization_dir, f"bf_vs_constraint_{model_name}.pdf")
    
    try:
        matplotlib_plot(
            constraints, 
            yvalues_records, 
            save_path, 
            tag=tag,
            y_label=ebf_name_visualization_name_mapping(tag),
            fontsize=50, 
            linewidth=5,
            # base_only=True # To simplify plot if only one model type
        )
        logger.info(f"Saved plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot: {e}")




# from language_modeling/main.py, storytelling/main.py, mmlu/main.py, cognac/main.py
def step1_run_llm(args):
    """
    Unified step1_run_llm that handles all task types: mmlu, language_modeling, storytelling, cognac
    """
    model = args.model
    chat_template_path = getattr(args, 'chat_template_path', None)
    task_type = getattr(args, 'task_type', 'language_modeling')  # default to language_modeling
    
    # Build file name based on task type
    file_name_parts = [
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed
    ]
    
    if task_type == 'language_modeling' or task_type == 'storytelling':
        if getattr(args, 'word_level_constraint', False):
            file_name_parts.append("_word_level_constraint_multiplier_{}".format(getattr(args, 'word_level_constraint_multiplier', 10)))
        if task_type == 'storytelling' and getattr(args, 'input_file', None) is not None:
            file_name_parts.append('_input_file_{}'.format(os.path.basename(args.input_file)))
    elif task_type == 'mmlu':
        if getattr(args, 'task_selection_filename', None) is not None:
            file_name_parts.append('_task_selection_file_{}'.format(os.path.basename(args.task_selection_filename)))
        if getattr(args, 'expand_options', False):
            file_name_parts.append('_expand_options')
        if getattr(args, 'nudging', False):
            file_name_parts.append('_nudging_{}_{}_{}'.format(
                os.path.basename(getattr(args, 'nudging_model', '')), 
                getattr(args, 'nudging_max_prefix_length', 5),
                getattr(args, 'nudging_freq_threshold', 50)))
    if len(file_name_parts) < 8:
        file_name_parts.append("")
    
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}.pt".format(*file_name_parts)
    file_name = os.path.join(args.output_root_dir, file_name)
    
    # setting up the logger
    ckpt_freq = getattr(args, 'ckpt_freq', 128)
    accumulation_batch_size = getattr(args, 'ckpt_freq', 128) * (1 if task_type in ['language_modeling', 'storytelling'] else 3)
    manager = ForwardManager(args, ckpt_freq=ckpt_freq, accumulation_batch_size=accumulation_batch_size)
    tokenizer = setup_tokenizer(model, chat_template_path)
    metadata_filename = file_name.replace(".pt", ".metadata")
    
    if os.path.exists(metadata_filename):
        # Load from metadata
        if task_type == 'mmlu':
            prompts, roles, answers, options, all_constrained_prompts, args = torch.load(metadata_filename, weights_only=False)
            all_task_prompts = all_constrained_prompts
            all_original_prompts = None  # MMLU doesn't store original prompts separately
        elif task_type == 'cognac':
            prompts, answers, sources, (hierarchy, updated_args) = torch.load(metadata_filename, weights_only=False)
            all_task_prompts = prompts
            all_original_prompts = None
            roles = None
        else:
            prompts, roles, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename, weights_only=False)
    else:
        # Generate prompts based on task type
        if task_type == 'mmlu':
            # MMLU constraint processing is done inside create_mmlu_data_constraints
            all_constrained_prompts, roles, answers, options = get_mmlu_data(args)
            all_task_prompts = all_constrained_prompts
            all_original_prompts = None
            
            prompts = []
            # sys msg from: https://github.com/FranxYao/chain-of-thought-hub/blob/main/MMLU/run_mmlu_gpt_3.5_turbo.py
            system_message = "Follow the given examples and answer the question."
            for constrained_prompt in all_constrained_prompts:
                prompt = format_prompt(model, constrained_prompt, tokenizer, system_message)
                prompts.append(prompt)
            torch.save((prompts, roles, answers, options, all_constrained_prompts, args), metadata_filename)
            
        elif task_type == 'cognac':
            prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data_wrapper(args)
            all_task_prompts = prompts
            all_original_prompts = None
            roles = None
            torch.save([prompts, answers, sources, (hierarchy, updated_args)], metadata_filename)
            
        elif task_type == 'storytelling':
            all_original_prompts = get_storytelling_data(args)
            all_task_prompts = apply_constraints_language_modeling_storytelling(all_original_prompts, args)
            
            # Handle nudging for storytelling
            generated_prefix = ""
            if getattr(args, 'nudging', False):
                from uncertainty_quantification.nudging_find_common_prefix import search_for_common_prefix
                ckpt_path = getattr(args, 'nudging_ckpt_path', None)
                nudging_model = getattr(args, 'nudging_model', None)
                nudging_dict_tree = torch.load(ckpt_path)
                candidate_prefixes = search_for_common_prefix(
                    nudging_dict_tree, model=nudging_model,
                    max_prefix_length=getattr(args, 'nudging_max_prefix_length', 5),
                    freq_threshold=getattr(args, 'nudging_freq_threshold', 50)
                )
                nudging_tokenizer = AutoTokenizer.from_pretrained(nudging_model)
                chosen_idx = 0
                if "8B" in nudging_model:
                    chosen_idx = 2
                elif "70B" in nudging_model:
                    chosen_idx = 1
                logger.info("candidate_prefixes: {}".format(candidate_prefixes[chosen_idx]))
                generated_prefix = nudging_tokenizer.decode(candidate_prefixes[chosen_idx][0])
                logger.info(f"Generated prefix: {generated_prefix}, original prefix: {candidate_prefixes[chosen_idx]}")
            
            # Get roles for storytelling
            try:
                from storytelling.consts import roles as storytelling_roles
                roles = storytelling_roles
            except ImportError:
                roles = [""]
            
            # Process prompts with roles
            system_message_template = "Now let's play a role-playing game. Please never share your identity in your output. For example, it is forbidden to say 'As a ...' or 'I am a ...'. Now Pretending "
            prompts = process_prompts_with_roles(all_task_prompts, model, tokenizer, roles, system_message_template, generated_prefix)
            torch.save((prompts, roles, all_task_prompts, all_original_prompts, args), metadata_filename)
            
        else:  # language_modeling (default)
            all_original_prompts = get_data(args)
            all_task_prompts = apply_constraints_language_modeling_storytelling(all_original_prompts, args)
            
            # Handle nudging for language modeling
            generated_prefix = ""
            if getattr(args, 'nudging', False):
                from uncertainty_quantification.nudging_find_common_prefix import search_for_common_prefix
                ckpt_path = getattr(args, 'nudging_ckpt_path', None)
                nudging_model = getattr(args, 'nudging_model', None)
                nudging_dict_tree = torch.load(ckpt_path)
                candidate_prefixes = search_for_common_prefix(
                    nudging_dict_tree, model=nudging_model,
                    max_prefix_length=getattr(args, 'nudging_max_prefix_length', 5),
                    freq_threshold=getattr(args, 'nudging_freq_threshold', 50)
                )
                nudging_tokenizer = AutoTokenizer.from_pretrained(nudging_model)
                chosen_idx = 0
                if "8B" in nudging_model:
                    chosen_idx = 2
                elif "70B" in nudging_model:
                    chosen_idx = 1
                logger.info("candidate_prefixes: {}".format(candidate_prefixes[chosen_idx]))
                generated_prefix = nudging_tokenizer.decode(candidate_prefixes[chosen_idx][0])
                logger.info(f"Generated prefix: {generated_prefix}, original prefix: {candidate_prefixes[chosen_idx]}")
            
            prompts = []
            for _task_prompt in all_task_prompts:
                system_message = ""
                prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message) + generated_prefix)
            
            roles = [""]  # language modeling uses empty roles
            torch.save((prompts, roles, all_task_prompts, all_original_prompts, args), metadata_filename)
    maybe_exist_flag = False
    response = None
    print("Total prompts: {}".format(len(prompts)))
    
    # Configure GPU memory and batch size based on model and task type
    gpu_memory_utilization = compute_gpu_memory_utilization(model)
    max_num_seqs = 16  # default
    
    # Model-specific adjustments
    if "70b" in model.lower() and "llama-3" in model.lower():
        gpu_memory_utilization = 0.55
        max_num_seqs = 128
    elif "70b" in model.lower() and "llama-2" in model.lower():
        gpu_memory_utilization = 0.8
        max_num_seqs = 64
    elif "8b" in model.lower():
        gpu_memory_utilization = 0.3
        max_num_seqs = 128
    elif task_type == 'language_modeling':
        # For language modeling, use more aggressive settings
        gpu_memory_utilization = 0.9
        max_num_seqs = 256
    
    # Task-specific adjustments
    if task_type == 'mmlu':
        args.enable_chunked_prefill = True
        args.max_num_batched_tokens = 256
        if "8b" in model.lower():
            gpu_memory_utilization = 0.3
            max_num_seqs = 128
    
    manager.setup_model(max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
    
    # step-1: get original generated story
    if os.path.exists(file_name):
        print("File exists: {}".format(file_name))
        try:
            response = torch.load(file_name, weights_only=False)
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
    
    # Return metadata in consistent format
    if task_type == 'mmlu':
        return [prompts, roles, answers, options, all_task_prompts, args], response, file_name
    elif task_type == 'cognac':
        return [prompts, answers, sources, (hierarchy, updated_args)], response, file_name
    else:
        return [prompts, roles, all_task_prompts, all_original_prompts, args], response, file_name

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
    parser = argparse.ArgumentParser(description='Unified Demo Args Parsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=50, min_p=0, top_p=0.9, temperature=1.0, output_root_dir="response_wikitext")
    
    # Task type selection
    parser.add_argument("--task_type", type=str, default="language_modeling", 
                        choices=["mmlu", "language_modeling", "storytelling", "cognac"],
                        help="Type of task to run")
    
    # Constraint arguments (used by language_modeling and storytelling)
    parser.add_argument("--constraint_level", type=int, default=5,
                        help="constraint level (in #(words) * multiplier/#(sentences) for language_modeling/storytelling, or #(\\n\\n splits) for mmlu)")
    parser.add_argument("--max_constraint_level", type=int, default=10,
                        help="maximum constraint level (for language_modeling/storytelling)")
    parser.add_argument("--word_level_constraint", action="store_true", help="constraint level in word level")
    parser.add_argument("--word_level_constraint_multiplier", type=int, default=10, help="constraint level multiplier in word level")
    
    # Dataset arguments (used by language_modeling)
    parser.add_argument("--dataset_path", type=str, default="Salesforce/wikitext", help="task/dataset path, first argument of datasets.load_dataset")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-v1", help="task/dataset name, second argument of datasets.load_dataset")
    parser.add_argument("--dataset_sample_counts", type=int, default=50, help="sample counts for dataset")
    parser.add_argument("--min_word_count", type=int, default=50, help="minimum word count per instance for dataset")
    parser.add_argument("--min_tokens", type=int, default=50, help="minimum token number per instance")
    
    # MMLU-specific arguments
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_small.pt", help="task file for MMLU")
    parser.add_argument("--expand_options", action="store_true", help="expand options into prompts (MMLU only)")
    
    # Storytelling-specific arguments
    parser.add_argument("--input_file", type=str, help='input file for storytelling', default=None)
    
    # Cognac-specific arguments
    parser.add_argument("--multi_constraints", type=int, default=1, help="multi constraints (cognac only)")
    
    # Nudging arguments (used by mmlu, language_modeling, storytelling)
    parser.add_argument("--nudging", action="store_true", help="nudging mode")
    parser.add_argument("--nudging_ckpt_path", type=str, default=None, help="nudging ckpt path")
    parser.add_argument("--nudging_model", type=str, default=None, help="nudging model")
    parser.add_argument("--nudging_max_prefix_length", type=int, default=5, help="nudging max prefix length")
    parser.add_argument("--nudging_freq_threshold", type=int, default=50, help="nudging freq threshold")
    
    parser.add_argument("--plotting_mode", action="store_true", help="Only run visualization on existing files")

    args = parser.parse_args()
    
    if args.plotting_mode:
        run_plotting_mode(args)
        exit(0)

    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)
    logger.add(os.path.join(output_root_dir, "experiment_{}.log".format(os.path.basename(args.model))), rotation="10 MB")
    # log all arguments for easy reproduction and experiment tracking
    logger.info("Arguments: {}".format(args))
    logger.info("Task type: {}".format(args.task_type))
    logger.info("Starting Step 1: Running LLM")
    metadata, response, file_name = step1_run_llm(args)
    
    # Extract prompts from metadata (format varies by task type)
    if args.task_type == 'mmlu':
        prompts, roles, answers, options, all_task_prompts, args = metadata
    elif args.task_type == 'cognac':
        prompts, answers, sources, (hierarchy, updated_args) = metadata
    else:
        prompts, roles, all_task_prompts, all_original_prompts, args = metadata
    
    logger.info("Starting Step 2: Computing Loglik and Entropy")
    distribution_profile = step2_compute_loglik_and_entropy(response, metadata, args)
    logger.info("Starting Step 3: Computing BF Values")
    bf_values_per_prompt, overall_bf_value = step3_compute_bf_values(distribution_profile)
    logger.info(f"Overall BF value: {overall_bf_value}")
    
    # # Build BF file name
    # bf_file_name_parts = [
    #     os.path.basename(args.model),
    #     args.sample_counts,
    #     args.max_tokens, args.log_probs,
    #     args.min_p, args.top_p, args.seed
    # ]
    # if args.task_type in ['language_modeling', 'storytelling']:
    #     if getattr(args, 'word_level_constraint', False):
    #         bf_file_name_parts.append("_word_level_constraint_multiplier_{}".format(getattr(args, 'word_level_constraint_multiplier', 10)))
    # if len(bf_file_name_parts) < 8:
    #     bf_file_name_parts.append("")
    # bf_file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}{}_bf.pt".format(*bf_file_name_parts)
    bf_file_name = file_name.replace(".pt", "_bf.pt")
    torch.save([bf_values_per_prompt, overall_bf_value], os.path.join(output_root_dir, bf_file_name))

