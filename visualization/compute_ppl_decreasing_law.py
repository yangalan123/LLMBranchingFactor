from multiprocessing.managers import Value
from uncertainty_quantification.consts import root_path as REMOTE_BASE


import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Dict, Set, Tuple, Any

import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
# from typing import Dict, List, Tuple
import os
from matplotlib.colors import LinearSegmentedColormap
import traceback
import argparse
from uncertainty_quantification.consts import ALL_MODELS
from uncertainty_quantification.visualization_utils import model_name_visualization_name_mapping
# Task configurations from previous script
TASK_CONFIGS = {
    "cognac": {
        "path": "cognac/stat_cognac_app_ctrlgen_multi_constraints",
        "output_prefix": "cognac_responses_200",
        "max_tokens": "512"
    },
    "cognac_extended": {
        "path": "cognac/stat_cognac_app_ctrlgen_multi_constraints_extended",
        "output_prefix": "cognac_responses_200",
        "max_tokens": "512"
    },
    "mmlu": {
        "path": "mmlu/stat_mmlu_app_ctrlgen_multi_constraints",
        "output_prefix": "response_mmlu_256",
        "max_tokens": "256"
    },
    "mmlu_wa_filler": {
        "path": "mmlu/stat_mmlu_app_ctrlgen_multi_constraints",
        "output_prefix": "response_mmlu_256_with_filler_wa",
        "max_tokens": "256"
    },
    "cnn_dm": {
        "path": "language_modeling/stat_news_app_ctrlgen_multi_constraints",
        "output_prefix": "response_cnn_dm_news",
        "max_tokens": "512"
    },
    "bbcnews": {
        "path": "language_modeling/stat_news_app_ctrlgen_multi_constraints",
        "output_prefix": "response_news",
        "max_tokens": "512"
    },
    "storytelling": {
        "path": "storytelling/stat_storytelling_app_ctrlgen_multi_constraints",
        "output_prefix": "response_storywriting_local_story_gen_full_word_level_constraint",
        "max_tokens": "1024"
    }
}
LLAMA_MODELS = [model for model in ALL_MODELS if "llama" in model.lower()]
def parse_args():
    parser = argparse.ArgumentParser(description='Analysis of language model metrics across tasks')

    # Data loading parameters
    parser.add_argument('--maxlen', type=int, default=5,
                        help='Maximum length parameter for data loading, -1 for no maximum length')
    parser.add_argument('--smoothing', type=float, default=1.0,
                        help='Smoothing factor for data processing')
    parser.add_argument('--min_samples', type=int, default=1,
                        help='Minimum number of samples required for model-wise analysis')

    # Analysis parameters
    parser.add_argument('--fontsize', type=int, default=40,
                        help='Font size for plots')
    parser.add_argument('--linewidth', type=float, default=5,
                        help='Line width for plots')
    parser.add_argument('--hist_bins', type=int, default=20,
                        help='Number of bins for histogram analysis')

    # Visualization parameters
    parser.add_argument('--fig_width', type=int, default=20,
                        help='Width of output figures')
    parser.add_argument('--fig_height', type=int, default=16,
                        help='Height of output figures')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')

    parser.add_argument('--output_dir', type=str, default='ppl_decreasing_law_analysis',)

    return parser.parse_args()

def get_pkl_path(task: str, model_name: str, maxlen: int = 5, smoothing: float = 1.0, metric_name: str="perplexity") -> str:
    """Constructs the path to the perplexity PKL file for a given task."""
    config = TASK_CONFIGS[task]
    return os.path.join(
        REMOTE_BASE,
        f"{config['path']}_maxlen_{maxlen}" if maxlen > 0 else f"{config['path']}",
        f"output_manual_check_{config['output_prefix']}_app_ctrlgen_multi_constraints_max_tokens_{config['max_tokens']}_min_p_0_top_p_0.9",
        f"visualization_promptwise_0_maxlen_{maxlen}_smoothing_{smoothing}" if maxlen > 0 else f"visualization_promptwise_0_smoothing_{smoothing}",
        f"piecewise_ebf_{model_name}_{metric_name}.pkl"
    )


def load_task_data(task: str, model_name: str, maxlen: int = 5, smoothing: float = 1.0, metric_name: str = "perplexity") -> List[Tuple]:
    """Loads and processes data for a single task."""
    dataset = []
    # set_per_dimensions = [set() for _ in range(len(dimension_names))]

    pkl_path = get_pkl_path(task, model_name, maxlen, smoothing, metric_name)
    try:
        with open(pkl_path, "rb") as f:
            x_values, y_value_dict = pickle.load(f)
            model_with_constraints = list(y_value_dict.keys())
            model_with_constraints = [[x.split("_constraint_")[0], int(x.split("_constraint_")[1]), x] for x in
                                      model_with_constraints]
            model_with_constraints.sort(key=lambda x: x[1])
            model_names = set([x[0] for x in model_with_constraints])
            assert len(model_names) == 1, "Expecting only one model, got {}".format(model_names)

            for idx, (model_name, constraint_level, _model_name_with_constraint) in enumerate(model_with_constraints):
                dataset.append((model_name, constraint_level, [x + 5 for x in x_values[_model_name_with_constraint]],
                                y_value_dict[_model_name_with_constraint]["perplexity"]))
    except FileNotFoundError:
        print(f"Warning: Could not find PKL file for task {task} at {pkl_path}")
    except Exception as e:
        print(f"Error processing task {task}: {str(e)}")

    # print(dataset)
    # exit()

    return dataset

def create_custom_colormap():
    """Create a custom blue-white-red colormap similar to seaborn's RdBu"""
    colors = ['#2166ac', '#f7f7f7', '#b2182b']  # Blue, White, Red
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

# conduct regression analysis to fit models: log(y) = a + b * log(x) for each (task, model, constraint) pair
# plot the fitted models and the scatter plot of the data
# additionally, plot the heatmap for signed R^2 values
# save all figures to pdf

def perform_regression_analysis(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform log-log regression analysis.
    Returns: slope, intercept, and signed R² value
    """
    # Convert to log scale, handling potential negative or zero values
    mask = (x_values > 0) & (y_values > 0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan

    log_x = np.log(x_values[mask])
    # log_x = x_values[mask]
    log_y = np.log(y_values[mask])

    # Perform linear regression
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    r_squared = r_value ** 2
    # Sign the R² based on the slope
    signed_r_squared = r_squared if slope > 0 else -r_squared

    return slope, intercept, signed_r_squared


def plot_regression_results(args, task_results: Dict[str, Dict[str, List]], output_dir: str) -> Dict[
    str, Dict[str, float]]:
    """
    Plot regression results for each task-model combination
    Returns: Dictionary containing regression results for each task-model pair
    """
    os.makedirs(output_dir, exist_ok=True)

    # Store regression results
    regression_results = {}

    # Prepare figure for regression plots
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plt.rcParams.update({'font.size': args.fontsize})

    for task in task_results:
        regression_results[task] = {}
        for model in task_results[task]:
            data = task_results[task][model]
            regression_results[task][model] = {}
            constraints = [point[1] for point in data]
            for idx, constraint in enumerate(constraints):
                # try:
                    # x_values = np.array([point[2] for point in data])
                x_values = np.array(data[idx][2])
                    # y_values = np.array([point[3] for point in data])
                y_values = np.array(data[idx][3])
                # except ValueError:
                #     print(f"Error processing task {task} and model {model}")
                #     print(data)
                #     traceback.print_exc()
                #     exit()

                slope, intercept, r_squared = perform_regression_analysis(x_values, y_values)
                regression_results[task][model][constraint] = {'slope': slope, 'intercept': intercept, 'r_squared': r_squared}

                if not np.isnan(slope):
                    # Plot scatter and regression line
                    plt.scatter(x_values, y_values, alpha=0.5, label=f'C={constraint}')
                    x_range = np.linspace(min(x_values), max(x_values), 100)
                    y_pred = np.exp(intercept + slope * np.log(x_range))
                    # y_pred = np.exp(intercept + slope * x_range)
                    plt.plot(x_range, y_pred, 'r-', linewidth=args.linewidth,
                             # label=f'Fit: log(y) = {intercept:.2f} + {slope:.2f}log(x)\nR² = {abs(r_squared):.2f}')
                            label = f'C={constraint}, Signed R^2 = {r_squared:.2f}')

            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{task} - {model}')
            plt.xlabel('Position')
            plt.ylabel('Perplexity')
            plt.legend()
            # plt.grid(True)

            # Save individual plot
            plt.savefig(os.path.join(output_dir, f'regression_{task}_{model}.pdf'),
                        bbox_inches='tight', dpi=args.dpi)
            plt.clf()

    return regression_results


def create_heatmap(regression_results: Dict[str, Dict[str, float]], output_dir: str, metric_key: str, args):
    """
    Create a heatmap of signed R² values for all task-model combinations using pre-computed results
    """
    # Extract tasks and models
    tasks = list(regression_results.keys())
    models = list(set([model for task in regression_results.values() for model in task.keys()]))

    # Create matrix of R² values
    r_squared_matrix = np.zeros((len(tasks), len(models)))

    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            if model in regression_results[task]:
                _value = np.mean([abs(regression_results[task][model][constraint][metric_key]) for constraint in regression_results[task][model]])
                r_squared_matrix[i, j] = _value

    # Create heatmap
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plt.rcParams.update({'font.size': args.fontsize})

    colormap = create_custom_colormap()
    plt.imshow(r_squared_matrix, cmap=colormap, aspect='auto', vmin=-1, vmax=1)

    # Add colorbar and labels
    plt.colorbar(label=metric_key)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.yticks(range(len(tasks)), tasks)

    plt.title(f'{metric_key} Values Across Tasks and Models')
    plt.tight_layout()

    # Save heatmap
    plt.savefig(os.path.join(output_dir, f'{metric_key}_heatmap.pdf'),
                bbox_inches='tight', dpi=args.dpi)
    plt.close()

def create_output_directory(args: argparse.Namespace) -> str:
    """
    Create a structured output directory based on hyperparameter combinations.
    Returns the path to the created directory.
    """
    # Create base timestamp for this run
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure based on hyperparameters
    output_dir = os.path.join(
        args.output_dir,
        # f"maxlen_{args.maxlen}_smoothing_{args.smoothing}_min_samples_{args.min_samples}",
        f"maxlen_{args.maxlen}_smoothing_{args.smoothing}",
    )

    # Combine with timestamp to ensure uniqueness
    # output_dir = os.path.join(args.base_output_dir, f"{param_dir}")

    # Create the directory
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration for reproducibility
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Analysis Configuration:\n")
        f.write("=====================\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    return output_dir

def main():
    args = parse_args()
    output_dir = create_output_directory(args)

    # Dictionary to store results for each task and model
    task_results = {}

    # Process each task and model combination
    for task in TASK_CONFIGS.keys():
        try:
            task_results[task] = {}
            for model in LLAMA_MODELS:
                model_name = model_name_visualization_name_mapping(os.path.basename(model))
                dataset = load_task_data(task, model_name, args.maxlen, args.smoothing)
                task_results[task][model_name] = dataset
        except Exception as e:
            print(f"Error processing task {task}: {str(e)}")
            traceback.print_exc()
            # if len(dataset) >= args.min_samples:
            #     task_results[task][model] = dataset

    # Generate plots and analysis
    regression_results = plot_regression_results(args, task_results, output_dir)
    create_heatmap(regression_results, output_dir, "r_squared", args)
    create_heatmap(regression_results, output_dir, "slope", args)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()