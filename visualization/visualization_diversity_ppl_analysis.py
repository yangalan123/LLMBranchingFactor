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

# Task configurations from previous script
TASK_CONFIGS = {
    "Cognac": {
        "path": "cognac/stat_cognac_app_ctrlgen_multi_constraints",
        "output_prefix": "cognac_responses_200",
        "max_tokens": "512"
    },
    "MMLU": {
        "path": "mmlu/stat_mmlu_app_ctrlgen_multi_constraints",
        "output_prefix": "response_mmlu_256",
        "max_tokens": "256"
    },
    "BBCNewsLatest": {
        "path": "language_modeling/stat_news_app_ctrlgen_multi_constraints",
        "output_prefix": "response_news",
        "max_tokens": "512"
    },
    "CreativeStoryGen": {
        "path": "storytelling/stat_storytelling_app_ctrlgen_multi_constraints",
        "output_prefix": "response_storywriting_local_story_gen_full_word_level_constraint",
        "max_tokens": "1024"
    }
}

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
    parser.add_argument('--base_output_dir', type=str, default='analysis_results',
                        help='Base directory for all output files')
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

    parser.add_argument('--output_dir', type=str, default='diversity_perplexity_regressional_analysis',)

    return parser.parse_args()

def get_pkl_path(task: str, maxlen: int = 5, smoothing: float = 1.0, metric_name: str="perplexity") -> str:
    """Constructs the path to the perplexity PKL file for a given task."""
    config = TASK_CONFIGS[task]
    maxlen_suffix = f"_maxlen_{maxlen}" if maxlen > 0 else ""
    return os.path.join(
        REMOTE_BASE,
        f"{config['path']}_maxlen_{maxlen}" if maxlen > 0 else f"{config['path']}",
        f"output_manual_check_{config['output_prefix']}_app_ctrlgen_multi_constraints_max_tokens_{config['max_tokens']}_min_p_0_top_p_0.9",
        f"visualization_promptwise_0{maxlen_suffix}_smoothing_{smoothing}" if maxlen > 0 else f"visualization_promptwise_0_smoothing_{smoothing}",
        f"model_wise_comparison_ebf_{metric_name}.pkl" if task != "storytelling" else f"model_wise_comparison_plot_ebf_{metric_name}.pkl"
    )


def load_task_data(task: str, maxlen: int = 5, smoothing: float = 1.0, metric_name: str = "perplexity") -> List[Tuple]:
    """Loads and processes data for a single task."""
    dataset = []
    # set_per_dimensions = [set() for _ in range(len(dimension_names))]

    pkl_path = get_pkl_path(task, maxlen, smoothing, metric_name)
    try:
        with open(pkl_path, "rb") as f:
            x_values, y_value_dict = pickle.load(f)

            for model, y_values in y_value_dict.items():
                is_instruct_model = "instruct" in model.lower() or "chat" in model.lower()
                model_parts = model.split("-")
                if len(model_parts) >= 3:
                    model_size = model_parts[2]
                    model_gen = model_parts[1]
                else:
                    # Handle cases where model name doesn't follow the expected format
                    print("Warning: Model name doesn't follow the expected format:", model)
                    model_size = "unknown"
                    model_gen = "unknown"

                for x, y in zip(x_values, y_values[f'ebf_{metric_name}']):
                    data_point = (is_instruct_model, model_size, model_gen, model, x, y)
                    dataset.append(data_point)
                    # for i, value in enumerate([is_instruct_model, model_size, model_gen, x]):
                    #     set_per_dimensions[i].add(value)
    except FileNotFoundError:
        print(f"Warning: Could not find PKL file for task {task} at {pkl_path}")
    except Exception as e:
        print(f"Error processing task {task}: {str(e)}")

    return dataset



def create_custom_colormap():
    """Create a custom blue-white-red colormap similar to seaborn's RdBu"""
    colors = ['#2166ac', '#f7f7f7', '#b2182b']  # Blue, White, Red
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

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
        f"maxlen_{args.maxlen}_smoothing_{args.smoothing}_min_samples_{args.min_samples}",
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


def create_custom_colormap():
    """Create a custom blue-white-red colormap similar to seaborn's RdBu"""
    colors = ['#2166ac', '#f7f7f7', '#b2182b']  # Blue, White, Red
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

def model_wise_regression_analysis(datasets: Dict, task: str, args: argparse.Namespace, output_dir: str) -> None:
    """Perform model-wise regression analysis to compare with global correlations."""
    metric_pairs = []
    for metric1 in ['perplexity']:
        for metric2 in ['distinct_1', 'distinct_2', 'distinct_3', 'distinct_4']:
            metric_pairs.append((metric1, metric2))

    args.metrics = ['perplexity', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4']
    diversity_metrics = args.metrics[1:]

    # Group data by model
    model_data = {}
    for metric in datasets:
        for point in datasets[metric]:
            model = point[3]
            if model not in model_data:
                model_data[model] = {m: [] for m in args.metrics}
            model_data[model][metric].append(point[5])

    # Analyze each model separately
    fig, axes = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height))
    fig.suptitle(f'{task.upper()} Task - Model-wise Correlation Analysis', fontsize=args.fontsize + 4)
    model_wise_r_values = dict()

    for idx, (metric1, metric2) in enumerate(metric_pairs):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # Calculate global correlation for reference
        global_data1 = np.array([p[5] for p in datasets[metric1]])
        global_data2 = np.array([p[5] for p in datasets[metric2]])
        global_corr, _ = stats.spearmanr(global_data1, global_data2)

        # Analyze each model
        model_corrs = []
        for model, data in model_data.items():
            if len(data[metric1]) >= args.min_samples:  # Only analyze models with sufficient data
                model_corr, _ = stats.spearmanr(data[metric1], data[metric2])
                model_corrs.append((model, model_corr))
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[metric1], data[metric2])
            signed_r_squared = r_value ** 2 if slope > 0 else -r_value ** 2
            if model not in model_wise_r_values:
                model_wise_r_values[model] = {m: [] for m in diversity_metrics}
            model_wise_r_values[model][metric2] = signed_r_squared


        # Plot distribution of model-wise correlations
        if model_corrs:
            corrs = [c for _, c in model_corrs]
            ax.hist(corrs, bins=args.hist_bins, alpha=0.6)
            ax.axvline(global_corr, color='r', linestyle='--',
                       label=f'Global correlation: {global_corr:.3f}')
            ax.set_xlabel('Correlation coefficient')
            ax.set_ylabel('Number of models')
            ax.set_title(f'{metric1} vs {metric2}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_model_wise_correlations.pdf'),
                dpi=args.dpi, bbox_inches='tight')
    plt.close()
    plt.clf()
    models = list(model_wise_r_values.keys())
    heatmap_data = np.zeros((len(models), len(diversity_metrics)))
    for i, model in enumerate(models):
        for j, metric in enumerate(diversity_metrics):
            heatmap_data[i, j] = model_wise_r_values[model][metric]
    # normalize heatmap data
    # heatmap_data = (heatmap_data - heatmap_data.mean(axis=0)) / heatmap_data.std(axis=0)
    # plot heatmap
    fig_heatmap = plt.figure(figsize=(args.fig_width, args.fig_height))
    ax_heatmap = fig_heatmap.add_subplot(111)
    # Create custom colormap
    cmap = create_custom_colormap()

    # Plot heatmap
    im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap=cmap)

    # Add colorbar
    plt.colorbar(im)
    # Configure axes
    ax_heatmap.set_xticks(np.arange(len(diversity_metrics)))
    ax_heatmap.set_yticks(np.arange(len(models)))
    ax_heatmap.set_xticklabels(diversity_metrics, rotation=45, ha='right')
    ax_heatmap.set_yticklabels(models)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(diversity_metrics)):
            text = ax_heatmap.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                   ha='center', va='center')

    # plt.title(f'{task.upper()} Task - Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_modelwise_diversity_signed_r2.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()



def regressional_analysis(datasets: Dict, task: str, args: argparse.Namespace, output_dir: str) -> None:
    """Perform regressional analysis on the given datasets."""
    # Set up visualization parameters
    plt.style.use('default')
    plt.rc('font', size=args.fontsize)
    plt.rc('axes', titlesize=args.fontsize)
    plt.rc('axes', labelsize=args.fontsize)
    plt.rc('xtick', labelsize=args.fontsize - 2)
    plt.rc('ytick', labelsize=args.fontsize - 2)

    # Define colors
    main_color = '#1f77b4'  # Nice blue color
    regression_color = '#d62728'  # Nice red color

    # Prepare for analysis across different metric combinations
    metric_pairs = []
    for i, metric1 in enumerate(['perplexity']):
        for metric2 in ['distinct_1', 'distinct_2', 'distinct_3', 'distinct_4']:
            metric_pairs.append((metric1, metric2))

    # Create figures for R² analysis and correlation
    fig_r2, axes_r2 = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height))
    fig_corr, axes_corr = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height))

    for idx, (metric1, metric2) in enumerate(metric_pairs):
        row, col = idx // 2, idx % 2

        # Extract data for the metric pair
        data1 = np.array([point[5] for point in datasets[metric1]])
        data2 = np.array([point[5] for point in datasets[metric2]])
        assert len(data1) == len(data2), "Data lengths do not match, data1: {}, data2: {}".format(len(data1),
                                                                                                  len(data2))

        # Remove any NaN values
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
        data1 = data1[valid_mask]
        data2 = data2[valid_mask]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
        r_squared = r_value ** 2
        signed_r_squared = r_squared if slope > 0 else -r_squared

        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(data1, data2)

        # Plot R² analysis
        ax_r2 = axes_r2[row, col]
        ax_r2.scatter(data1, data2, alpha=0.6, color=main_color, edgecolor='white', s=100)
        x_line = np.linspace(min(data1), max(data1), 100)
        y_line = slope * x_line + intercept
        ax_r2.plot(x_line, y_line, color=regression_color, linewidth=args.linewidth)
        ax_r2.set_xlabel(f'{metric1}', fontsize=args.fontsize)
        ax_r2.set_ylabel(f'{metric2}', fontsize=args.fontsize)
        ax_r2.set_title(f'R² = {signed_r_squared:.3f}\np = {p_value:.3e}')
        ax_r2.grid(True, linestyle='--', alpha=0.7)

        # Plot correlation analysis
        ax_corr = axes_corr[row, col]
        # Create density plot using 2D histogram
        h, xedges, yedges = np.histogram2d(data1, data2, bins=20)
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot density and scatter
        ax_corr.contourf(X, Y, h.T, levels=10, cmap='Blues', alpha=0.5)
        ax_corr.scatter(data1, data2, alpha=0.4, color=main_color, edgecolor='white', s=80)
        ax_corr.plot(x_line, y_line, color=regression_color, linewidth=args.linewidth)
        ax_corr.set_xlabel(f'{metric1}', fontsize=args.fontsize)
        ax_corr.set_ylabel(f'{metric2}', fontsize=args.fontsize)
        ax_corr.set_title(f'Spearman ρ = {spearman_corr:.3f}\np = {spearman_p:.3e}')
        ax_corr.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save figures
    for fig, name in [(fig_r2, 'r2'), (fig_corr, 'correlation')]:
        plt.figure(fig.number)
        plt.suptitle(f'{task.upper()} Task - {"R² Analysis" if name == "r2" else "Correlation Analysis"}',
                     fontsize=args.fontsize + 4, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task}_{name}_analysis.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Add model-wise analysis
    model_wise_regression_analysis(datasets, task, args, output_dir)


def collect_cross_task_correlations(all_task_datasets: Dict[str, Dict[str, List]], args: argparse.Namespace) -> Dict[
    str, Dict[str, Dict[str, float]]]:
    """
    Collect correlation metrics across all tasks for each model and metric pair.

    Returns:
        Dict with structure: {
            'signed_r2': {
                'distinct_1': {model: {task: value}},
                'distinct_2': {model: {task: value}},
                ...
            },
            'spearman': {
                'distinct_1': {model: {task: value}},
                'distinct_2': {model: {task: value}},
                ...
            }
        }
    """
    diversity_metrics = ['distinct_1', 'distinct_2', 'distinct_3', 'distinct_4']
    correlations = {
        'signed_r2': {metric: {} for metric in diversity_metrics},
        'spearman': {metric: {} for metric in diversity_metrics}
    }

    for task, datasets in all_task_datasets.items():
        # Group data by model
        model_data = {}
        for metric in datasets:
            for point in datasets[metric]:
                model = point[3]  # model name is at index 3
                if model not in model_data:
                    model_data[model] = {m: [] for m in ['perplexity'] + diversity_metrics}
                model_data[model][metric].append(point[5])  # metric value is at index 5

        # Calculate correlations for each model and metric pair
        for model, data in model_data.items():
            if len(data['perplexity']) >= args.min_samples:
                perplexity_data = np.array(data['perplexity'])

                for div_metric in diversity_metrics:
                    div_data = np.array(data[div_metric])

                    # Calculate signed R²
                    slope, _, r_value, _, _ = stats.linregress(perplexity_data, div_data)
                    r_squared = r_value ** 2
                    signed_r_squared = r_squared if slope > 0 else -r_squared

                    # Calculate Spearman correlation
                    spearman_corr, _ = stats.spearmanr(perplexity_data, div_data)

                    # Store results
                    if model not in correlations['signed_r2'][div_metric]:
                        correlations['signed_r2'][div_metric][model] = {}
                    if model not in correlations['spearman'][div_metric]:
                        correlations['spearman'][div_metric][model] = {}

                    correlations['signed_r2'][div_metric][model][task] = signed_r_squared
                    correlations['spearman'][div_metric][model][task] = spearman_corr

    return correlations


def plot_cross_task_heatmaps(correlations: Dict, tasks: List[str], args: argparse.Namespace, output_dir: str) -> None:
    """Plot heatmaps for cross-task correlations."""
    plt.style.use('default')
    plt.rc('font', size=args.fontsize)

    # Create custom colormap
    cmap = create_custom_colormap()

    for corr_type in ['signed_r2', 'spearman']:
        for div_metric in correlations[corr_type].keys():
            # Prepare data for heatmap
            models = sorted(correlations[corr_type][div_metric].keys())
            heatmap_data = np.zeros((len(models), len(tasks)))

            for i, model in enumerate(models):
                for j, task in enumerate(tasks):
                    value = correlations[corr_type][div_metric][model].get(task, np.nan)
                    heatmap_data[i, j] = value

            # Create figure
            fig = plt.figure(figsize=(args.fig_width, args.fig_height))
            ax = fig.add_subplot(111)

            # Plot heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap)

            # Add colorbar
            plt.colorbar(im)

            # Configure axes
            ax.set_xticks(np.arange(len(tasks)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels([t.upper() for t in tasks], rotation=45, ha='right')
            ax.set_yticklabels(models)

            # Add text annotations
            for i in range(len(models)):
                for j in range(len(tasks)):
                    if not np.isnan(heatmap_data[i, j]):
                        text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                       ha='center', va='center')

            # Set title
            metric_name = div_metric.replace('_', '-')
            # plt.title(f'Cross-task {corr_type.replace("_", " ").title()} between Perplexity and {metric_name}')

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cross_task_{corr_type}_{div_metric}.pdf'),
                        dpi=args.dpi, bbox_inches='tight')
            plt.close()

def main():
    args = parse_args()
    output_dir = create_output_directory(args)

    # Create structured output directory

    metric_names = ['perplexity', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4']

    all_task_datasets = {}
    # Analyze each task individually
    for task in TASK_CONFIGS.keys():
        task_datasets = dict()
        try:
            print(f"Output directory created at: {output_dir}")
            task_dir = os.path.join(output_dir, task)
            os.makedirs(task_dir, exist_ok=True)

            for metric_name in metric_names:
                print(f"\nProcessing task: {task}, metric: {metric_name}")
                dataset = load_task_data(task, maxlen=args.maxlen,
                                         smoothing=args.smoothing,
                                         metric_name=metric_name)
                task_datasets[metric_name] = dataset

            # Perform analysis and get results
            regressional_analysis(task_datasets, task, args, task_dir)
            # only collect dataset when regressional analysis is successful
            all_task_datasets[task] = task_datasets

            # # Save results with proper organization
            # save_analysis_results(args, results, task_dir, task)

        except Exception as e:
            print(f"Error processing task {task}: {str(e)}")
            # error_log_path = os.path.join(output_dir, f"{task}_error_log.txt")
            # with open(error_log_path, "w") as f:
            #     traceback.print_exc(file=f)
            traceback.print_exc()
    # Collect cross-task correlations
    correlations = collect_cross_task_correlations(all_task_datasets, args)

    # Plot cross-task heatmaps
    plot_cross_task_heatmaps(correlations, list(all_task_datasets.keys()), args, output_dir)


if __name__ == "__main__":
    main()