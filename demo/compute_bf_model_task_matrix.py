import glob
import os
import traceback
import re
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from uncertainty_quantification.visualization_utils import (
    matplotlib_plot, 
    matplotlib_plot_piecewise,
    model_name_visualization_name_mapping, 
    ebf_name_visualization_name_mapping,
    loglik_type_visualization_name_mapping,
    base_aligned_model_name_mapping,
    DEFAULT_FIG_SIZE,
    DEFAULT_FONT_SIZE
)
from uncertainty_quantification.io_utils import StoreManager
from uncertainty_quantification.consts import root_path
from demo import step2_compute_loglik_and_entropy, step3_compute_bf_values
import argparse
import loguru

def extract_arg_value(arg_string, arg_name):
    pattern = rf'{arg_name}_(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, arg_string)
    if match:
        return match.group(1)
    return None

def load_new_format_bf_values(root_dir, bf_values=None):
    """
    Load new format BF values (*_bf.pt) from the directory recursively.
    """
    if bf_values is None:
        bf_values = defaultdict(dict) # bf_values[vis_model_name][constraint_level] = val
    
    # # Clean root_dir if it contains glob patterns
    # if "*" in root_dir:
    #     root_dir = os.path.dirname(root_dir)
        
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return bf_values

    # Search for _bf.pt files
    search_pattern = os.path.join(root_dir, "**", "*_bf.pt")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} _bf.pt files in {root_dir}")
    debug_search_pattern = os.path.join(root_dir, "application_ctrlgen_multi_constraints_1", "*_bf.pt")
    debug_files = glob.glob(debug_search_pattern, recursive=True)
    print(f"Found {len(debug_files)} _bf.pt files in {root_dir}/application_ctrlgen_multi_constraints_1")
    # extract model name from debug_files
    debug_model_names = list()
    for file in debug_files:
        model_name = os.path.basename(file).split("_")[0]
        debug_model_names.append(model_name)
    print(f"Debug model names: {debug_model_names}")
    
    for file_path in files:
        try:
            # Extract constraint level
            dir_name = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            print(f"File name: {file_name}")
            
            # Try extracting from directory name first
            match = re.search(r'application_ctrlgen_multi_constraints_(\d+)', dir_name)
            if match:
                constraint_level = int(match.group(1))
            else:
                # Fallback to filename extraction
                constraint_level_str = extract_arg_value(file_name, "constraint_level")
                if constraint_level_str is None:
                    constraint_level_str = extract_arg_value(file_name, "word_level_constraint_multiplier")
                
                if constraint_level_str:
                    constraint_level = int(float(constraint_level_str))
                else:
                    print(f"Could not extract constraint level from: {file_path}")
                    continue

            # Extract model name
            # Assuming model name is the first part of filename before _response_
            model_name = file_name.split("_response_")[0]
            vis_model_name = model_name_visualization_name_mapping(model_name)
            if vis_model_name in bf_values and constraint_level in bf_values[vis_model_name] and len(bf_values[vis_model_name][constraint_level]) > 0:
                print(f"Model {vis_model_name} with constraint level {constraint_level} already exists in bf_values")
                continue
            
            # Load data
            # with StoreManager(temp_dir="./temp") as store:
            #     data = store.load(file_path)
            data = torch.load(file_path, weights_only=False)
            overall_bf = None
            
            if isinstance(data, list) or isinstance(data, tuple):
                if len(data) == 3:
                    bf_values_per_prompt, overall_bf, distribution_profile = data
                elif len(data) == 2:
                    bf_values_per_prompt, overall_bf = data
                    distribution_profile = None
            
            if overall_bf is not None:
                # Use visualization name mapping for consistency
                bf_values[vis_model_name][constraint_level] = [overall_bf, bf_values_per_prompt]
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
    return bf_values

def plot_heatmap(data_dict, output_filename, title):
    if not data_dict:
        print(f"No data found for heatmap: {output_filename}")
        return

    df = pd.DataFrame(data_dict)
    print(f"Heatmap Data for {output_filename}:")
    print(df)
    
    # Heatmap Data for X
    # It looks like:
    # Task, Model Family, Ratio
    # mmlu, Llama-2-13B, 1.2
    
    # Custom sorting for the x-axis (Model Family)
    # We want: llama-2, llama-3, olmo2-series, qwen3
    # Within each series: sort by size (e.g., 8B -> 70B)
    
    def get_model_sort_key(model_name):
        model_name_lower = model_name.lower()
        
        # Family ordering: 0: llama-2, 1: llama-3, 2: olmo2-series, 3: qwen3
        family_idx = 99
        if 'llama-2' in model_name_lower:
            family_idx = 0
        elif 'llama-3' in model_name_lower:
            family_idx = 1
        elif 'olmo' in model_name_lower:
            family_idx = 2
        elif 'qwen' in model_name_lower:
            family_idx = 3
            
        # Size extracting
        size_val = 0.0
        match = re.search(r'(\d+(?:\.\d+)?)[bx]', model_name_lower)
        if match:
            size_val = float(match.group(1))
        # Handle cases like 8x7B (Mixtral)
        if '8x7b' in model_name_lower:
            size_val = 8 * 7.0
            
        return (family_idx, size_val, model_name)

    df['sort_key'] = df['Model Family'].apply(get_model_sort_key)
    sorted_models = df[['Model Family', 'sort_key']].drop_duplicates().sort_values('sort_key')['Model Family'].tolist()
    
    # Pivot for heatmap
    pivot_table = df.pivot(index="Task", columns="Model Family", values="Ratio")
    
    # Reorder columns based on custom sort
    # Filter sorted_models to only those present in the pivot_table
    ordered_cols = [m for m in sorted_models if m in pivot_table.columns]
    pivot_table = pivot_table[ordered_cols]
    
    # Plot
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    sns.set(font_scale=2.5)
    # Using mask for NaNs is automatic in seaborn, but we can make it explicit if needed.
    ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                     cbar_kws={'label': 'BF Ratio (Base/Aligned)'},
                     annot_kws={"size": 35})
    # plt.title(title, fontsize=DEFAULT_FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=35)
    plt.yticks(fontsize=35)
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"Saved heatmap to {output_filename}")

if __name__ == '__main__':
    # add command line arguments for whether resume from checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--ckpt_file", type=str, default="./bf_model_task_matrix.pt", help="checkpoint file")
    args = parser.parse_args()
    no_resume = args.no_resume
    ckpt_file = args.ckpt_file
    if not no_resume:
        print(f"Resuming from checkpoint: {ckpt_file}")
    else:
        print(f"Not resuming from checkpoint: {ckpt_file}")
    # e.g., model_wise_comparison_ebf_cond_ppl_prod.pdf
    # common_bf_filename_pattern = "model_wise_comparison_ebf*pkl"
    # e.g., xx_loglik_profile.pdf
    # common_loglik_filename_pattern = "*loglik_profile.pkl"
    common_loglik_filename_pattern = "loglik_analysis*"
    bf_smoothing_factor = 0.1
    # for old step2 and step3 shell scripts output
    # bf_dirs = {
    #     "cognac": f"{root_path}/cognac/stat_cognac_app_ctrlgen_multi_constraints/output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "cnn_dm": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_cnn_dm_news_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "random_strings": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_random_strings_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "bbcnews": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_news_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "mmlu": f"{root_path}/mmlu/stat_mmlu_app_ctrlgen_multi_constraints/output_manual_check_response_mmlu_256_app_ctrlgen_multi_constraints_max_tokens_256_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "storytelling": f"{root_path}/storytelling/stat_storytelling_app_ctrlgen_multi_constraints/output_manual_check_response_storywriting_local_story_gen_full_word_level_constraint_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_*/*plot*pkl",
    #     "cognac_plan": f"{root_path}/cognac/stat_cognac_app_ctrlgen_multi_constraints_keywords_mode_2/output_manual_check_cognac_responses_keywords_mode_2_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
    #     "wikitext": f"{root_path}/language_modeling/stat_lm_app_ctrlgen_multi_constraints/output_manual_check_response_lm_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}"
    # }
    loglik_dirs = {
        "cognac": f"{root_path}/cognac/output_loglik_cognac_responses_200_max_tokens_512_min_p_0_top_p_0.9",
        "cnn_dm": f"{root_path}/language_modeling/output_loglik_response_cnn_dm_news_max_tokens_512_min_p_0_top_p_0.9",
        "random_strings": f"{root_path}/language_modeling/output_loglik_response_random_strings_max_tokens_512_min_p_0_top_p_0.9",
        "bbcnews": f"{root_path}/language_modeling/output_loglik_response_news_max_tokens_512_min_p_0_top_p_0.9",
        "mmlu": f"{root_path}/mmlu/output_loglik_response_mmlu_256_max_tokens_256_min_p_0_top_p_0.9",
        # "storytelling": f"{root_path}/storytelling/output_loglik_response_storywriting_local_story_gen_full_word_level_constraint_max_tokens_1024_min_p_0_top_p_0.9/*plot*pkl",
        "storytelling": f"{root_path}/storytelling/output_loglik_response_storywriting_local_story_gen_full_word_level_constraint_max_tokens_1024_min_p_0_top_p_0.9",
        "cognac_plan": f"{root_path}/cognac/output_loglik_cognac_responses_keywords_mode_2_max_tokens_512_min_p_0_top_p_0.9",
        "wikitext": f"{root_path}/language_modeling/output_loglik_response_lm_max_tokens_1024_min_p_0_top_p_0.9"
    }
    new_root_dir = "/net/scratch2/chenghao/bf_formal_codebase/demo"
    new_format_loglik_dirs = {
        "mmlu": f"{new_root_dir}/response_mmlu",
        "storytelling": f"{new_root_dir}/response_storywriting",
        "bbcnews": f"{new_root_dir}/response_news",
        "random_strings": f"{new_root_dir}/response_random_strings",
    }
    
    # Define task sets
    subset_tasks = ['mmlu', 'storytelling', "random_strings", "bbcnews"]
    all_tasks = list(loglik_dirs.keys())
    
    aligned_to_base_mapping_dict, base_to_aligned_mapping_dict = base_aligned_model_name_mapping()
    
    # Data storage
    # all_distribution_profiles_old_output = {}
    # all_bf_values_merged = {}
    if os.path.exists(ckpt_file) and not no_resume:
        all_bf_values_merged = torch.load(ckpt_file, weights_only=False)
    else:
        # all_distribution_profiles_old_output = {}
        all_bf_values_merged = {}
    
    print(base_to_aligned_mapping_dict)
    print(aligned_to_base_mapping_dict)
    # 1. Load Data for ALL tasks (Old Format)
    file_verification_errors = 0
    for task in all_tasks:
        if task not in all_bf_values_merged:
            all_bf_values_merged[task] = dict()
            # all_distribution_profiles_old_output[task] = dict()
            
        # --- Old Format Loading ---
        file_pattern = os.path.join(loglik_dirs[task], common_loglik_filename_pattern)
        files = glob.glob(file_pattern)
        print(f"[Old Format] Found {len(files)} files for task: {task}: {files}")
        # if task not in all_distribution_profiles_old_output:
        #     all_distribution_profiles_old_output[task] = dict()
        #     all_bf_values_merged[task] = dict()
        if task in all_bf_values_merged and len(all_bf_values_merged[task]) > 0:
            print(f"Task {task} already exists in all_bf_values_merged with models: {list(all_bf_values_merged[task].keys())}")
            continue
        all_bf_values_merged[task] = dict()
        for file in files:
            try:
                distribution_profile = torch.load(file, weights_only=False)
                constraint_levels = list(distribution_profile.keys())
                models = list(distribution_profile[constraint_levels[0]].keys())
                assert len(models) == 1, f"multiple models: {models}, task: {task}, file: {file}"
                assert "entropy" in distribution_profile[constraint_levels[0]][models[0]], f"entropy not found, task: {task}, file: {file}"
                model = model_name_visualization_name_mapping(models[0])
                # keep comments below -- distribution profile could be useful for future analysis
                # assert model not in all_distribution_profiles_old_output[task], f"model: {model} already exists, task: {task}, file: {file}"
                # all_distribution_profiles_old_output[task][model] = dict() 
                all_bf_values_merged[task][model] = dict()
                # assert "metadata" in distribution_profile[constraint_levels[0]][model], f"metadata not found, task: {task}, file: {file}"
                # args = distribution_profile[constraint_levels[0]][model]["metadata"][-1]
                for constraint_level in constraint_levels:
                    # all_distribution_profiles_old_output[task][model][constraint_level] = distribution_profile[constraint_level][models[0]]
                    bf_values_per_prompt, overall_bf_value = step3_compute_bf_values(distribution_profile[constraint_level][models[0]])
                    # Store only the scalar for heatmap
                    all_bf_values_merged[task][model][constraint_level] = [overall_bf_value, bf_values_per_prompt]
                
            except Exception as e:
                file_verification_errors += 1
                print(f"Error loading old file: {file}, error: {e}")
                print(traceback.format_exc())
        with StoreManager(temp_dir="./temp") as store:
            store.save(all_bf_values_merged, ckpt_file, async_write=False)
        

    # 2. Load Data for Specific tasks (New Format)
    # Only load for tasks defined in new_format_loglik_dirs
    # path_recorded = dict()
    new_format_ckpt_file = "./new_format_bf_values.pt"
    if os.path.exists(new_format_ckpt_file) and not no_resume:
        new_bf_values = torch.load(new_format_ckpt_file, weights_only=False)
    else:
        new_bf_values = dict()
    for task, path in new_format_loglik_dirs.items():
        if task not in all_bf_values_merged:
            all_bf_values_merged[task] = dict()
            
        print(f"Loading new format BF values for task: {task}")
        if task not in new_bf_values:
            new_bf_values[task] = None
        _new_bf_values = load_new_format_bf_values(path, new_bf_values[task])
        new_bf_values[task] = _new_bf_values
        print("Models: ", list(new_bf_values[task].keys()))
        
        for model, constraints in new_bf_values[task].items():
            all_bf_values_merged[task][model] = dict()
            
            for constraint, val in constraints.items():
                # assert model not in all_bf_values_merged[task], f"model: {model} already exists, task: {task}"
                all_bf_values_merged[task][model][constraint] = val
        with StoreManager(temp_dir="./temp") as store:
            store.save(new_bf_values, new_format_ckpt_file, async_write=False)

    # 3. Compute Ratios
    # We will generate two datasets: one for subset, one for all
    
    def compute_heatmap_data(task_list):
        heatmap_data = [] 
        
        task_name_mapping = {
            "random_strings": "Random Strings",
            "bbcnews": "BBCNewsLatest",
            "storytelling": "Creative StoryGen",
            "mmlu": "MMLU"
        }
        
        for task in task_list:
            if task not in all_bf_values_merged:
                print(f"Task {task} not found in all_bf_values_merged")
                continue
                
            display_task_name = task_name_mapping.get(task, task)
                
            for aligned_model, base_model in aligned_to_base_mapping_dict.items():
                vis_aligned = model_name_visualization_name_mapping(aligned_model)
                vis_base = model_name_visualization_name_mapping(base_model)
                
                # Check availability in the specific task
                if (vis_aligned in all_bf_values_merged[task] and 
                    vis_base in all_bf_values_merged[task]):
                    
                    # Find common constraint levels
                    aligned_constraints = set(all_bf_values_merged[task][vis_aligned].keys())
                    base_constraints = set(all_bf_values_merged[task][vis_base].keys())
                    common_constraints = aligned_constraints.intersection(base_constraints)
                    
                    if not common_constraints:
                        print(f"No common constraints for {task}: {vis_aligned} vs {vis_base}")
                        continue
                        
                    ratios = []
                    for c in common_constraints:
                        val_aligned = all_bf_values_merged[task][vis_aligned][c][0]
                        val_base = all_bf_values_merged[task][vis_base][c][0]
                        # Compute Base / Aligned
                        if val_aligned != 0:
                            ratios.append(val_base / val_aligned)
                    
                    if ratios:
                        avg_ratio = np.mean(ratios)
                        
                        # Use a simplified model family name for the plot
                        # model_family = vis_aligned.replace("-chat", "").replace("Instruct", "").strip()
                        # if model_family.endswith("-"): model_family = model_family[:-1]
                        model_family = model_name_visualization_name_mapping(vis_base)
                        if "-Base" in model_family:
                            model_family = model_family.replace("-Base", "")
                        
                        heatmap_data.append({
                            "Task": display_task_name,
                            "Model Family": model_family,
                            "Ratio": avg_ratio
                        })
        return heatmap_data

    # Generate Subset Heatmap
    data_subset = compute_heatmap_data(subset_tasks)
    plot_heatmap(data_subset, "bf_ratio_heatmap_selected.pdf", 
                "Branching Factor Ratio: Base / Aligned (Selected Tasks)")

    # Generate All Tasks Heatmap
    data_all = compute_heatmap_data(all_tasks)
    plot_heatmap(data_all, "bf_ratio_heatmap_all.pdf", 
                "Branching Factor Ratio: Base / Aligned (All Tasks)")
