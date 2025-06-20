import random
import traceback
import concurrent.futures
from functools import partial

from uncertainty_quantification.visualization_utils import (
    DEFAULT_FIG_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_WIDTH, DEFAULT_VISUALIZATION_DIR,
    matplotlib_plot, matplotlib_plot_piecewise,
    model_name_visualization_name_mapping,
    axis_standardize
)
from uncertainty_quantification.uncertainty_computation import compute_bf_values
import argparse
import torch
import glob
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from uncertainty_quantification.consts import root_path
from loguru import logger
from tqdm import tqdm


def condense_arrays_with_inconsistent_lengths(array: list, reduce_fn=np.mean, q=0.95, prefix_mode=False,
                                              divide_by_length=False):
    array_lengths = [len(x) for x in array]
    quantile_length = np.quantile(array_lengths, q)
    array_cumsums = [np.cumsum(x) for x in array]

    # Define worker function for parallel processing
    def process_position(i):
        if prefix_mode:
            elements_at_this_position = np.array([array_cumsums[x_i][i] for x_i, x in enumerate(array) if len(x) >= i + 1])
        else:
            elements_at_this_position = np.array([x[i] for x in array if len(x) >= i + 1])
        if divide_by_length:
            elements_at_this_position = elements_at_this_position / (i + 1)
        if len(elements_at_this_position) <= 3:
            return None
        return reduce_fn(elements_at_this_position)

    # Parallel execution
    final_outputs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in executor.map(process_position, range(int(quantile_length))):
            if result is None:
                break
            final_outputs.append(result)

    return final_outputs


def load_loglik_data(filepath: str):
    # I will implement this by myself, no need to worry
    # it will return the x-axis (output position), and y-axis (length-averaged logliks up to the current position)
    # as two arrays
    logger.info(f"Now loading {filepath}")
    loglik_dict_all = torch.load(filepath)
    logger.info(f"Loaded {len(loglik_dict_all)} logliks")
    # randomly pick one level for now for simplicity
    constraint_levels = list(loglik_dict_all.keys())
    # constraint_levels.sort()
    # constraint_level = constraint_levels[0]
    ret = {}

    # Define worker function for processing each constraint level and instance
    def process_instance(args):
        constraint_level, instance_idx, instance, instance_non_trunc, entropy = args
        model_names = list(loglik_dict_all[constraint_level].keys())
        assert len(
            model_names) == 1, f"multiple or zero model_names? {model_names} at {filepath}, constraint = {constraint_level}, constraint_levels = {constraint_levels}"
        model_name = model_names[0]
        loglik_dict = loglik_dict_all[constraint_level][model_name]
        if "output_per_token_logprob_truncated" not in loglik_dict:
            logger.warning("truncated logprob not found", filepath, loglik_dict.keys())
            return constraint_level, instance_idx, None, None

        new_mean_logliks = np.array(
            condense_arrays_with_inconsistent_lengths(instance, prefix_mode=True, divide_by_length=True))
        new_mean_logliks_non_trunc = np.array(
            condense_arrays_with_inconsistent_lengths(instance_non_trunc, prefix_mode=True, divide_by_length=True))
        new_std_logliks = np.array(
            condense_arrays_with_inconsistent_lengths(instance, prefix_mode=True, divide_by_length=True,
                                                      reduce_fn=np.std))
        new_std_logliks_non_trunc = np.array(
            condense_arrays_with_inconsistent_lengths(instance_non_trunc, prefix_mode=True, divide_by_length=True,
                                                      reduce_fn=np.std))
        new_mean_entropies = np.array(
            condense_arrays_with_inconsistent_lengths(entropy, prefix_mode=True, divide_by_length=True)
        )
        new_std_entropies = np.array(
            condense_arrays_with_inconsistent_lengths(entropy, prefix_mode=True, divide_by_length=True, reduce_fn=np.std)
        )
        return constraint_level, instance_idx, new_mean_logliks, new_std_logliks, new_mean_logliks_non_trunc, new_std_logliks_non_trunc, new_mean_entropies, new_std_entropies

    # Prepare all tasks for parallel execution
    all_tasks = []
    for constraint_level in constraint_levels:
        model_names = list(loglik_dict_all[constraint_level].keys())
        assert len(
            model_names) == 1, f"multiple or zero model_names? {model_names} at {filepath}, constraint = {constraint_level}, constraint_levels = {constraint_levels}"
        model_name = model_names[0]
        loglik_dict = loglik_dict_all[constraint_level][model_name]
        if "output_per_token_logprob_truncated" not in loglik_dict:
            logger.warning("truncated logprob not found", filepath, loglik_dict.keys())
            exit()
        output_per_token_logprob = loglik_dict['output_per_token_logprob_truncated']
        output_per_token_logprob_non_trunc = loglik_dict['output_per_token_logprob']
        entropies = loglik_dict['entropy']
        assert len(output_per_token_logprob_non_trunc) == len(output_per_token_logprob), "inconsistent lengths: {} vs. {}".format(len(output_per_token_logprob), len(output_per_token_logprob_non_trunc))
        for idx, (instance, instance_non_trunc, entropy) in enumerate(zip(output_per_token_logprob, output_per_token_logprob_non_trunc, entropies)):
            all_tasks.append((constraint_level, idx, instance, instance_non_trunc, entropy))

    # Execute tasks in parallel with progress bar
    constraint_level_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(
                executor.map(process_instance, all_tasks),
                total=len(all_tasks),
                desc="Processing Loglik Instances",
                leave=False,
                position=1
        ):
            constraint_level, instance_idx, mean_logliks, std_logliks, mean_logliks_non_trunc, std_logliks_non_trunc, mean_entropy, std_entropy = result
            if mean_logliks is None:
                continue

            if constraint_level not in constraint_level_results:
                constraint_level_results[constraint_level] = {}

            constraint_level_results[constraint_level][instance_idx] = (mean_logliks, std_logliks, mean_logliks_non_trunc, std_logliks_non_trunc, mean_entropy, std_entropy)

    # Organize results by constraint level
    for constraint_level in tqdm(constraint_levels, desc=f'Processing Constraints at {filepath}', leave=False,
                                 position=0):
        if constraint_level not in constraint_level_results:
            continue

        output_token_logprobs_mean = []
        output_token_logprobs_std = []
        output_token_logprobs_mean_non_trunc = []
        output_token_logprobs_std_non_trunc = []
        output_token_entropies_mean = []
        output_token_entropies_std = []


        # Sort by instance index to maintain original order
        indices = sorted(constraint_level_results[constraint_level].keys())
        for idx in indices:
            mean_logliks, std_logliks, mean_logliks_non_trunc, std_logliks_non_trunc, mean_entropy, std_entropy = constraint_level_results[constraint_level][idx]
            output_token_logprobs_mean.append(mean_logliks)
            output_token_logprobs_std.append(std_logliks)
            output_token_logprobs_mean_non_trunc.append(mean_logliks_non_trunc)
            output_token_logprobs_std_non_trunc.append(std_logliks_non_trunc)
            output_token_entropies_mean.append(mean_entropy)
            output_token_entropies_std.append(std_entropy)

        mean_logliks = condense_arrays_with_inconsistent_lengths(output_token_logprobs_mean)
        std_logliks = condense_arrays_with_inconsistent_lengths(output_token_logprobs_std)
        mean_logliks_non_trunc = condense_arrays_with_inconsistent_lengths(output_token_logprobs_mean_non_trunc)
        std_logliks_non_trunc = condense_arrays_with_inconsistent_lengths(output_token_logprobs_std_non_trunc)
        mean_entropies = condense_arrays_with_inconsistent_lengths(output_token_entropies_mean)
        std_entropies = condense_arrays_with_inconsistent_lengths(output_token_entropies_std)
        assert len(mean_logliks) == len(
            std_logliks), f"inconsistent length of mean and std found when computing loglik: {len(mean_logliks), len(std_logliks)}"
        ret[constraint_level] = [
            np.arange(len(mean_logliks)) + 1,
            np.array(mean_logliks),
            np.array(std_logliks),
            np.array(mean_logliks_non_trunc),
            np.array(std_logliks_non_trunc),
            np.array(mean_entropies),
            np.array(std_entropies)
        ]
    return ret


def load_avg_entropy_data(filepath: str):
    # similar as above, but return avg entropy as y-axis
    entropy_dict_all = torch.load(filepath)
    constraint_levels = list(entropy_dict_all.keys())
    ret = {}

    # Define worker function for processing each constraint level and instance
    def process_instance(args):
        constraint_level, instance_idx, instance = args
        model_names = list(entropy_dict_all[constraint_level].keys())
        assert len(model_names) == 1, f"multiple model_names: {model_names}"
        model_name = model_names[0]
        entropy_dict = entropy_dict_all[constraint_level][model_name]
        ps = list(entropy_dict.keys())
        assert len(ps) == 1, f"multiple ps: {ps}"

        entropies = instance[-1]
        # new_entropies = np.array(condense_arrays_with_inconsistent_lengths(entropies, prefix_mode=True, divide_by_length=True))
        new_entropies = np.array(condense_arrays_with_inconsistent_lengths(entropies, prefix_mode=True, divide_by_length=True))
        # cumsum_entropies = new_entropies.cumsum()
        # return constraint_level, instance_idx, cumsum_entropies
        return constraint_level, instance_idx, new_entropies

    # Prepare all tasks for parallel execution
    all_tasks = []
    for constraint_level in constraint_levels:
        model_names = list(entropy_dict_all[constraint_level].keys())
        assert len(model_names) == 1, f"multiple model_names: {model_names}"
        model_name = model_names[0]
        entropy_dict = entropy_dict_all[constraint_level][model_name]
        ps = list(entropy_dict.keys())
        assert len(ps) == 1, f"multiple ps: {ps}"
        entropy_dict = entropy_dict[ps[0]]
        for idx, instance in enumerate(entropy_dict):
            all_tasks.append((constraint_level, idx, instance))

    # Execute tasks in parallel with progress bar
    constraint_level_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(
                executor.map(process_instance, all_tasks),
                total=len(all_tasks),
                desc="Processing Entropy Instances",
                leave=False,
                position=1
        ):
            constraint_level, instance_idx, cumsum_entropies = result
            if constraint_level not in constraint_level_results:
                constraint_level_results[constraint_level] = {}

            constraint_level_results[constraint_level][instance_idx] = cumsum_entropies

    # Organize results by constraint level
    for constraint_level in tqdm(constraint_levels, desc=f'Processing Constraints for Entropy', leave=False,
                                 position=0):
        output_token_entropies = []

        # Sort by instance index to maintain original order
        indices = sorted(constraint_level_results[constraint_level].keys())
        for idx in indices:
            output_token_entropies.append(constraint_level_results[constraint_level][idx])

        truncated_entropies = condense_arrays_with_inconsistent_lengths(output_token_entropies)
        ret[constraint_level] = [
            np.arange(len(truncated_entropies)) + 1,
            np.array(truncated_entropies)
        ]
    return ret


def process_loglik_paths(paths: list[str], source_dir: str):
    result = []
    paths = [x for x in paths if "enforce_min_p" not in x]

    # Define regex pattern to extract metadata
    # Pattern for: {task}/output_loglik_{source_dir}_max_tokens_{max_tokens}_min_p_{min_p}_top_p_{top_p}*/loglik_analysis_{model_name}.pt
    pattern = fr'output_loglik_{source_dir}_max_tokens_(\d+)_min_p_([\d\.]+)_top_p_([\d\.]+)[^/]*?/loglik_analysis_([^\.]+)\.pt'

    for path in paths:
        match = re.search(pattern, path)
        if match:
            max_tokens = match.group(1)
            min_p = match.group(2)
            top_p = match.group(3)
            model_name = match.group(4)

            # Create tuple with path and metadata
            result.append((path, source_dir, max_tokens, min_p, top_p, model_name))
        else:
            logger.warning(f"Warning: Could not extract metadata from loglik path: {path}")
    logger.info("loglik filepath parsing success rate: {}".format(len(result) / len(paths)))

    return result


def process_entropy_paths(paths: list[str], source_dir: str):
    result = []
    paths = [x for x in paths if "enforce_min_p" not in x]

    # Define regex pattern to extract metadata
    # Pattern for: {task}/output_manual_check_{source_dir}_max_tokens_{max_tokens}_min_p_{min_p}_top_p_{top_p}*/ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt
    pattern = fr'output_manual_check_{source_dir}.*?_max_tokens_(\d+)_min_p_([\d\.]+)_top_p_([\d\.]+)[^/]*?/ctrlgen_multi_constraints_investigation_increasing_PPL_([^\.]+)\.pt'

    for path in paths:
        match = re.search(pattern, path)
        if match:
            # source_dir = match.group(1)
            max_tokens = match.group(1)
            min_p = match.group(2)
            top_p = match.group(3)
            model_name = match.group(4)

            # Create tuple with path and metadata
            result.append((path, source_dir, max_tokens, min_p, top_p, model_name))
        else:
            logger.warning(f"Warning: Could not extract metadata from entropy path: {path}")
    logger.info("entropy filepath parsing success rate: {}".format(len(result) / len(paths)))

    return result


def compute_bf_values_dict(entropy_filepath: str, loglik_filepath: str, model_name: str, yvalues_record: dict,
                           asymptotic_limit=50):

    entropy_dict_all = torch.load(entropy_filepath)
    entropy_constraint_levels = list(entropy_dict_all.keys())
    loglik_dict_all = torch.load(loglik_filepath)
    # randomly pick one level for now for simplicity
    loglik_constraint_levels = list(loglik_dict_all.keys())
    shared_constraint_levels = list(set(entropy_constraint_levels).intersection(loglik_constraint_levels))
    shared_constraint_levels.sort()
    model_name_in_vis = model_name_visualization_name_mapping(model_name)
    assert model_name_in_vis not in yvalues_record, "{model_name} is already in yvalues record".format(
        model_name=model_name)
    yvalues_record[model_name_in_vis] = dict()

    # Define worker function to process each constraint level
    def process_constraint_level(constraint_level):
        model_names = list(entropy_dict_all[constraint_level].keys())
        assert len(model_names) == 1, f"multiple model_names: {model_names}"
        model_name = model_names[0]
        entropy_dict = entropy_dict_all[constraint_level][model_name]
        ps = list(entropy_dict.keys())
        assert len(ps) == 1, f"multiple ps: {ps}"
        entropy_dict = entropy_dict[ps[0]]
        loglik_dict = loglik_dict_all[constraint_level][model_name]
        output_per_token_logprob = loglik_dict['output_per_token_logprob_truncated']

        # Process entropy values
        output_token_entropies = []
        for instance in entropy_dict:
            entropies = instance[-1]
            new_entropies = np.array(condense_arrays_with_inconsistent_lengths(entropies))
            output_token_entropies.append(new_entropies.cumsum())

        # Process loglik values
        output_token_logprobs = []
        for instance in output_per_token_logprob:
            new_logliks = np.array(
                condense_arrays_with_inconsistent_lengths(instance, prefix_mode=True, divide_by_length=True))
            output_token_logprobs.append(new_logliks)

        # Compute BF values
        bf_values = compute_bf_values(output_token_entropies, output_token_logprobs, asymptotic_limit=asymptotic_limit)
        return constraint_level, np.exp(np.mean(bf_values))

    # Execute tasks in parallel with progress bar
    all_bf_values = []
    constraint_level_to_bf = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(
                executor.map(process_constraint_level, shared_constraint_levels),
                total=len(shared_constraint_levels),
                desc=f'Processing BF Values for {model_name}',
                leave=False
        ):
            constraint_level, bf_value = result
            constraint_level_to_bf[constraint_level] = bf_value

    # Maintain order of constraint levels
    for constraint_level in shared_constraint_levels:
        all_bf_values.append(constraint_level_to_bf[constraint_level])

    yvalues_record[model_name_in_vis]["BF"] = all_bf_values
    yvalues_record[model_name_in_vis]["constraints"] = shared_constraint_levels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("--output_pkl_dir", type=str, default='./output')
    parser.add_argument("--task", type=str, default='cognac')
    parser.add_argument("--source_dir", type=str, default='cognac_responses_200')
    parser.add_argument("--entropy_file_additional_output_dir_pattern", type=str, default="",
                        help="additional output dir pattern")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_pkl_dir = args.output_pkl_dir
    source_dir = args.source_dir
    entropy_file_additional_output_dir_pattern = args.entropy_file_additional_output_dir_pattern
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_pkl_dir, exist_ok=True)

    task = args.task
    loglik_path_pattern = f"{root_path}/{task}/output_loglik_{source_dir}_max_tokens_*_min_p_*_top_p_*/loglik_analysis_*.pt"
    loglik_paths = glob.glob(loglik_path_pattern)
    loglik_paths = process_loglik_paths(loglik_paths, source_dir=source_dir)

    entropy_path_pattern = f"{root_path}/{task}/output_manual_check_{source_dir}_*_max_tokens_*_min_p_*_top_p_*{entropy_file_additional_output_dir_pattern}/ctrlgen_multi_constraints_investigation_increasing_PPL_*.pt"
    entropy_paths = glob.glob(entropy_path_pattern)
    entropy_paths = process_entropy_paths(entropy_paths, source_dir=source_dir)

    # Task 1: Implement grouping
    # Group loglik_paths and entropy_paths according to exact match of [meta1, meta2, ...]
    grouped_paths = []

    # Create dictionaries to index paths by their metadata
    loglik_dict = {}
    for loglik_item in loglik_paths:
        meta_key = tuple(loglik_item[1:])  # Create a tuple from metadata for dictionary key
        loglik_dict[meta_key] = loglik_item

    # Find matching entropy items for each loglik metadata key
    for entropy_item in entropy_paths:
        meta_key = tuple(entropy_item[1:])
        if meta_key in loglik_dict:
            # Found a match, add to grouped_paths
            grouped_paths.append((entropy_item, loglik_dict[meta_key]))

    # Task 2: Complete plotting
    for entropy_item, loglik_item in grouped_paths:
        _entropy_path = entropy_item[0]
        _loglik_path = loglik_item[0]
        assert entropy_item[1:] == loglik_item[1:]
        meta_items = entropy_item[1:]

        # Load data
        entropy_data_dict = load_avg_entropy_data(_entropy_path)
        loglik_data_dict = load_loglik_data(_loglik_path)
        entropy_constraint_levels = set(list(entropy_data_dict.keys()))
        loglik_constraint_levels = set(list(loglik_data_dict.keys()))
        shared_constraint_levels = entropy_constraint_levels.intersection(loglik_constraint_levels)
        for sampled_constraint_level in shared_constraint_levels:
            if str(sampled_constraint_level) == "0":
                continue
            x_pos, entropy_values = entropy_data_dict[sampled_constraint_level]
            x_pos_loglik, loglik_values, loglik_values_std, loglik_values_non_trunc, loglik_values_std_non_trunc, mean_entropies, std_entropies = loglik_data_dict[sampled_constraint_level]
            for split_idx in [-1]:
                if split_idx > 0:
                    x_pos_split = x_pos[:split_idx]
                    entropy_values_split = entropy_values[:split_idx]
                    x_pos_loglik_split = x_pos_loglik[:split_idx]
                    loglik_values_split = loglik_values[:split_idx]
                    loglik_values_non_trunc_split = loglik_values_non_trunc[:split_idx]
                    mean_entropies_split = mean_entropies[:split_idx]
                    std_entropies_split = std_entropies[:split_idx]
                else:
                    x_pos_split = x_pos
                    entropy_values_split = entropy_values
                    x_pos_loglik_split = x_pos_loglik
                    loglik_values_split = loglik_values
                    loglik_values_non_trunc_split = loglik_values_non_trunc
                    mean_entropies_split = mean_entropies
                    std_entropies_split = std_entropies
                # Create figure with two y-axes (one for entropy, one for loglik)
                plt.rc('font', size=DEFAULT_FONT_SIZE)  # Controls default text sizes
                fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

                # Plot entropy on the first y-axis
                ax1.set_xlabel('Output Position', fontsize=DEFAULT_FONT_SIZE)
                ax1.set_ylabel('Bits', fontsize=DEFAULT_FONT_SIZE)
                ax1.plot(x_pos_split[1:], entropy_values_split[1:], color='tab:blue',
                         linewidth=DEFAULT_LINE_WIDTH, label='Length-Averaged Entropy')
                ax1.plot(x_pos_loglik_split[1:], -loglik_values_split[1:], color='tab:red',
                         linewidth=DEFAULT_LINE_WIDTH, label='Length-Averaged NLL')
                axis_standardize(ax1, ylabel_pad=5, xlabel_pad=5)
                ax1.legend(loc='best', fontsize=DEFAULT_FONT_SIZE - 2)

                # Adjust layout
                plt.tight_layout()

                # Save figure
                output_filename = f"{'_'.join(meta_items)}_split_{split_idx}_constraint_{sampled_constraint_level}.pdf"
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                plt.clf()
                pkl_output_filename = f"{'_'.join(meta_items)}_split_{split_idx}_constraint_{sampled_constraint_level}.pkl"
                pkl_output_path = os.path.join(output_pkl_dir, pkl_output_filename)
                torch.save([
                   x_pos_split, entropy_values_split, x_pos_loglik_split, loglik_values_split,
                ], pkl_output_path)

                logger.info(f"Saved plot to {output_path}")
            # additionally, plot a new figure with loglik_values_std
            split_idx = 100
            fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
            plt.rc('font', size=DEFAULT_FONT_SIZE)  # Controls default text sizes
            # starting from 3rd -- as the first few tokens are usually placeholder (e.g., <bos>)
            ax.plot(x_pos[3:split_idx], loglik_values_std[3:split_idx], color='tab:blue', linewidth=DEFAULT_LINE_WIDTH,
                    label="Loglik Std (truncated)")
            ax.set_xlabel('Output Position', fontsize=DEFAULT_FONT_SIZE)
            ax.set_ylabel('Length-Averaged NLL Std', fontsize=DEFAULT_FONT_SIZE)
            plt.subplots_adjust(bottom=0.05)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "{}_loglik_std.pdf".format(output_filename.replace(".pdf", "")))
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved plot to {output_path}")
            plt.close()
            fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
            plt.rc('font', size=DEFAULT_FONT_SIZE)  # Controls default text sizes
            ax.plot(x_pos[3:split_idx], loglik_values_std[3:split_idx], color='tab:blue', linewidth=DEFAULT_LINE_WIDTH,
                    label="Length-Averaged NLL Std")
            ax.plot(x_pos[3:split_idx], std_entropies[3:split_idx], color='tab:red', linewidth=DEFAULT_LINE_WIDTH,
                    label="Length-Averaged Entropy Std")
            ax.set_xlabel('Output Position', fontsize=DEFAULT_FONT_SIZE)
            axis_standardize(ax, ylabel_pad=5, xlabel_pad=5)
            ax.legend(loc='best', fontsize=DEFAULT_FONT_SIZE - 2)
            plt.subplots_adjust(bottom=0.05)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "{}_loglik_entropy_std.pdf".format(output_filename.replace(".pdf", "")))
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved plot to {output_path}")
            plt.close()

    # Task 3: compute and plot BF
    all_bf_values_dict = dict()
    deduplication_paths = dict()
    for entropy_item, loglik_item in grouped_paths:
        entropy_path = entropy_item[0]
        loglik_path = loglik_item[0]
        source_dir, max_tokens, min_p, top_p, model_name = entropy_item[1:]
        all_values_dict_key = (source_dir, max_tokens, min_p, top_p)
        if all_values_dict_key not in all_bf_values_dict:
            all_bf_values_dict[all_values_dict_key] = dict()
            deduplication_paths[all_values_dict_key] = dict()
        if model_name not in deduplication_paths[all_values_dict_key]:
            deduplication_paths[all_values_dict_key][model_name] = [entropy_path, loglik_path]
            assert model_name not in all_bf_values_dict[
                all_values_dict_key], f"{model_name} should not exist in yvalues dict!"
        else:
            logger.warning(
                "duplicated model_name: {} found for meta-key: {}\nExisting item: {}\nCurrent Item: {}".format(
                    model_name, all_values_dict_key, deduplication_paths[all_values_dict_key][model_name],
                    (entropy_path, loglik_path)))
        compute_bf_values_dict(entropy_path, loglik_path, model_name, all_bf_values_dict[all_values_dict_key])
    for key in all_bf_values_dict:
        source_dir, max_tokens, min_p, top_p = key
        if len(all_bf_values_dict[key]) > 1:
            try:
                model_names = list(all_bf_values_dict[key].keys())
                shared_constraint_levels = all_bf_values_dict[key][model_names[0]]['constraints']
                for model_name in model_names:
                    assert all_bf_values_dict[key][model_name][
                               'constraints'] == shared_constraint_levels, f"inconsistent constraint levels: {shared_constraint_levels} vs. {all_bf_values_dict[key][model_name]['constraints']}"
                output_path = os.path.join(DEFAULT_VISUALIZATION_DIR, "BF_values",
                                           "{}_{}_minp_{}_topp_{}".format(source_dir, max_tokens, min_p, top_p))
                os.makedirs(output_path, exist_ok=True)
                img_path = os.path.join(output_path, "model_wise_comparison_BF.pdf")
                matplotlib_plot(shared_constraint_levels, all_bf_values_dict[key], img_path, tag="BF",
                                y_label="Branching Factor")
                logger.info("Saved to {}".format(img_path))
            except AssertionError as e:
                logger.warning(e)
                traceback.print_stack()