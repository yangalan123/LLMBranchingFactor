import glob
import os
import pickle
from uncertainty_quantification.visualization_utils import ebf_name_visualization_name_mapping, loglik_type_visualization_name_mapping
from uncertainty_quantification.consts import root_path
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
# Create a color map for constraints
cmap = plt.get_cmap('viridis')

results = {}


def create_heatmap(data, save_path, title):
    plt.figure(figsize=(30, 20))

    # Remove unwanted columns if they exist
    data = data.drop(['r_squared', 'correlation_sign'], axis=1, errors='ignore')
    # Create a custom colormap
    colors = ['#FF9999', '#FFFFFF', '#99FF99']  # Light red, white, light green
    n_bins = 100  # Number of color gradations
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Determine the maximum absolute R² value for symmetrical color scaling
    vmax = np.abs(data).max().max()

    # Transpose the dataframe to flip x and y axes
    # data_transposed = data.T

    ax = sns.heatmap(data, annot=True, cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
                     fmt='.2f', cbar_kws={'label': 'Signed R² Value'})

    plt.title(title, fontsize=fontsize)
    plt.xlabel('Models', fontsize=fontsize - 10)
    plt.ylabel('Datasets', fontsize=fontsize - 10)

    # Increase font size for colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize - 20)
    cbar.set_label(cbar.ax.get_ylabel(), size=fontsize - 15)

    # Increase font size for annotations and tick labels
    for text in ax.texts:
        text.set_fontsize(fontsize - 20)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize - 20)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right', fontsize=fontsize - 20)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust these values as needed
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # e.g., model_wise_comparison_ebf_cond_ppl_prod.pdf
    common_bf_filename_pattern = "model_wise_comparison_ebf*pkl"
    # e.g., xx_loglik_profile.pdf
    common_loglik_filename_pattern = "*loglik_profile.pkl"
    bf_smoothing_factor = 0.1
    bf_dirs = {
        "cognac": f"{root_path}/cognac/stat_cognac_app_ctrlgen_multi_constraints/output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "cnn_dm": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_cnn_dm_news_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "random_strings": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_random_strings_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "bbcnews": f"{root_path}/language_modeling/stat_news_app_ctrlgen_multi_constraints/output_manual_check_response_news_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "mmlu": f"{root_path}/mmlu/stat_mmlu_app_ctrlgen_multi_constraints/output_manual_check_response_mmlu_256_app_ctrlgen_multi_constraints_max_tokens_256_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "storytelling": f"{root_path}/storytelling/stat_storytelling_app_ctrlgen_multi_constraints/output_manual_check_response_storywriting_local_story_gen_full_word_level_constraint_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_*/*plot*pkl",
        "cognac_plan": f"{root_path}/cognac/stat_cognac_app_ctrlgen_multi_constraints_keywords_mode_2/output_manual_check_cognac_responses_keywords_mode_2_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}",
        "wikitext": f"{root_path}/language_modeling/stat_lm_app_ctrlgen_multi_constraints/output_manual_check_response_lm_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9/visualization_promptwise_0_smoothing_{bf_smoothing_factor}"
    }
    loglik_dirs = {
        "cognac": f"{root_path}/cognac/output_loglik_cognac_responses_200_max_tokens_512_min_p_0_top_p_0.9",
        "cnn_dm": f"{root_path}/language_modeling/output_loglik_response_cnn_dm_news_max_tokens_512_min_p_0_top_p_0.9",
        "random_strings": f"{root_path}/language_modeling/output_loglik_response_random_strings_max_tokens_512_min_p_0_top_p_0.9",
        "bbcnews": f"{root_path}/language_modeling/output_loglik_response_news_max_tokens_512_min_p_0_top_p_0.9",
        "mmlu": f"{root_path}/mmlu/output_loglik_response_mmlu_256_max_tokens_256_min_p_0_top_p_0.9",
        "storytelling": f"{root_path}/storytelling/output_loglik_response_storywriting_local_story_gen_full_word_level_constraint_max_tokens_1024_min_p_0_top_p_0.9/*plot*pkl",
        "cognac_plan": f"{root_path}/cognac/output_loglik_cognac_responses_keywords_mode_2_max_tokens_512_min_p_0_top_p_0.9",
        "wikitext": f"{root_path}/language_modeling/output_loglik_response_lm_max_tokens_1024_min_p_0_top_p_0.9"
    }
    visualization_root = "bf_loglik_correlation"
    os.makedirs(visualization_root, exist_ok=True)
    figsize = (20, 15)
    fontsize = 50
    linewidth = 5
    n_col = 2
    for dataset_name in bf_dirs:
        if dataset_name in ["mmlu", "cnn_dm"]:
            continue
        print(f"Processing {dataset_name}")
        bf_path_pattern = os.path.join(bf_dirs[dataset_name], common_bf_filename_pattern) if "pkl" not in bf_dirs[dataset_name] else bf_dirs[dataset_name]
        loglik_path_pattern = os.path.join(loglik_dirs[dataset_name], common_loglik_filename_pattern) if "pkl" not in loglik_dirs[dataset_name] else loglik_dirs[dataset_name]
        bf_paths = glob.glob(bf_path_pattern)
        loglik_paths = glob.glob(loglik_path_pattern)
        assert len(bf_paths) > 0, f"No bf paths found for {bf_path_pattern}"
        assert len(loglik_paths) > 0, f"No loglik paths found for {loglik_path_pattern}"
        for bf_path in bf_paths:
            if "ppl" in bf_path:
                print(f"Skipping {bf_path}")
                continue
            for loglik_path in loglik_paths:
                if "output_loglik_profile.pkl" in loglik_path:
                    print(f"Skipping {loglik_path}")
                    continue
                bf_type = os.path.basename(bf_path).split("ebf_")[-1].split(".")[0]
                bf_label_name = ebf_name_visualization_name_mapping(bf_type)
                if "loglik_profile" in loglik_path:
                    loglik_type = os.path.basename(loglik_path).split("_loglik")[0]
                else:
                    loglik_type = os.path.basename(loglik_path).split("_wise")[0]
                loglik_label_name = loglik_type_visualization_name_mapping(loglik_type)

                pair_key = f"{bf_label_name}_{loglik_label_name}"
                if pair_key not in results:
                    results[pair_key] = {'r_squared': {}, 'correlation_sign': {}}

                visualization_dir = os.path.join(visualization_root, dataset_name, "{}_{}".format(bf_label_name, loglik_label_name))
                visualization_dir = visualization_dir.replace(" ", "_")
                os.makedirs(visualization_dir, exist_ok=True)

                bf_data = pickle.load(open(bf_path, "rb"))
                loglik_data = pickle.load(open(loglik_path, "rb"))
                assert bf_data[0] == loglik_data[0], "Constraint mismatch: {} vs {}".format(bf_data[0], loglik_data[0])
                constraints_level = bf_data[0]
                bfvalue_records = bf_data[1]
                loglik_records = loglik_data[1]
                models = list(bfvalue_records.keys())

                for model in models:
                    save_path = os.path.join(visualization_dir, "{}.pdf".format(model))
                    model_bf_values = bfvalue_records[model]["ebf_" + bf_type]
                    model_loglik_values = loglik_records[model][loglik_type]

                    # Create the scatter plot and fit the line
                    fig, ax = plt.subplots(figsize=figsize)

                    # Normalize constraints for color mapping
                    unique_constraints = sorted(set(constraints_level))
                    norm = mcolors.Normalize(vmin=min(unique_constraints), vmax=max(unique_constraints))

                    # Plot points with color based on constraint level
                    for bf, loglik, constraint in zip(model_bf_values, model_loglik_values, constraints_level):
                        ax.scatter(bf, loglik, c=[cmap(norm(constraint))], alpha=0.7, s=100)

                    # Calculate the line of best fit
                    slope, intercept, r_value, p_value, std_err = stats.linregress(model_bf_values, model_loglik_values)
                    line = slope * np.array(model_bf_values) + intercept

                    # Store R² value and correlation sign
                    if model not in results[pair_key]:
                        results[pair_key][model] = {}
                    results[pair_key][model][dataset_name] = np.sign(r_value) * r_value ** 2

                    # Plot the line of best fit
                    ax.plot(model_bf_values, line, color='r', linewidth=linewidth)

                    # Set labels and title with the requested changes
                    ax.set_xlabel(f"{bf_label_name}", fontsize=fontsize)
                    ax.set_ylabel(f"{loglik_label_name}", fontsize=fontsize)
                    ax.set_title(f"{model}", fontsize=fontsize)

                    # Add correlation coefficient to the plot
                    ax.text(0.05, 0.95, f'R² = {r_value ** 2:.2f}', transform=ax.transAxes,
                            fontsize=fontsize - 10, verticalalignment='top')

                    # Create a new axes for the colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)

                    # Add colorbar to show constraint levels
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax)
                    cbar.set_label('Constraint Level', fontsize=fontsize - 10)
                    cbar.ax.tick_params(labelsize=fontsize - 15)

                    # Adjust tick label font size
                    ax.tick_params(axis='both', which='major', labelsize=fontsize - 10)

                    # Tight layout and save
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close(fig)

    for pair_key, pair_data in results.items():
        # Combined signed R² heatmap
        signed_r_squared_df = pd.DataFrame(pair_data)
        heatmap_save_path = os.path.join(visualization_root, f"{pair_key}_signed_r_squared_heatmap.pdf")
        create_heatmap(signed_r_squared_df, heatmap_save_path, f"Signed R² Values Heatmap - {pair_key}")

    print("Visualization complete. Results saved in:", visualization_root)
    print("Heatmaps for each BF-loglik pair saved in the visualization root directory.")
