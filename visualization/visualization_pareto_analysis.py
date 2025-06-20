import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Dict, Set, Tuple, Any
from uncertainty_quantification.visualization_utils import DEFAULT_FONT_SIZE, DEFAULT_FIG_SIZE, DEFAULT_LINE_WIDTH
from uncertainty_quantification.consts import root_path as REMOTE_BASE

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


def get_pkl_path(task: str, maxlen: int = 5, smoothing: float = 1.0) -> str:
    """Constructs the path to the perplexity PKL file for a given task."""
    config = TASK_CONFIGS[task]
    maxlen_suffix = f"_maxlen_{maxlen}" if maxlen > 0 else ""
    return os.path.join(
        REMOTE_BASE,
        f"{config['path']}{maxlen_suffix}",
        f"output_manual_check_{config['output_prefix']}_app_ctrlgen_multi_constraints_max_tokens_{config['max_tokens']}_min_p_0_top_p_0.9",
        f"visualization_promptwise_0{maxlen_suffix}_smoothing_{smoothing}",
        "model_wise_comparison_ebf_perplexity.pkl" if task != "storytelling" else "model_wise_comparison_plot_ebf_perplexity.pkl"
    )


def load_task_data(task: str, dimension_names: List[str], smoothing: float = 1.0) -> Tuple[
    List[Tuple], List[Set]]:
    """Loads and processes data for a single task."""
    datasets = []
    set_per_dimensions = [set() for _ in range(len(dimension_names))]

    for maxlen in [-1]:
    # for maxlen in [5, -1]:
        pkl_path = get_pkl_path(task, maxlen, smoothing)
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
                        model_size = "unknown"
                        model_gen = "unknown"
                    if "13b" in model_size.lower():
                        # ignore llama-2-13b, as we do not have llama-3-13b
                        # continue
                        model_size = model_size.replace("13", "8")

                    for x, y in zip(x_values, y_values['ebf_perplexity']):
                        data_point = (is_instruct_model, maxlen, model_size, model_gen, x, y)
                        datasets.append(data_point)
                        for i, value in enumerate([is_instruct_model, maxlen, model_size, model_gen, x]):
                            set_per_dimensions[i].add(value)
        except FileNotFoundError:
            print(f"Warning: Could not find PKL file for task {task} at {pkl_path}")
        except Exception as e:
            print(f"Error processing task {task}: {str(e)}")

    return datasets, set_per_dimensions


def pareto_analysis(dimension_names: List[str], datasets: List[Tuple], method: str = "range", task: str = "all", output_dir: str="./") -> None:
    """
    Perform Pareto analysis.
    Args remain the same as your original function, added task name for output file
    """
    if not datasets:
        print(f"No data available for analysis for task: {task}")
        return

    df = pd.DataFrame(datasets, columns=dimension_names)

    # Calculate impacts (rest of your original pareto_analysis implementation)
    impacts = {}
    for dimension in dimension_names:
        if dimension == "Branching Factor":
            continue
        if dimension == "Len(Y)":
            # confounder, also influenced by other impact factors. Stratifying by this will lead to a collider bias
            continue
        grouped = df.groupby(dimension)["Branching Factor"].mean()

        if method == "range":
            impact = grouped.max() - grouped.min()
        elif method == "mean":
            levels = grouped.values
            if len(levels) == 1:
                print(f"dimension {dimension} only has one levels {levels}")
                continue
            impact = sum(abs(levels[i] - levels[j])
                         for i in range(len(levels))
                         for j in range(i + 1, len(levels))) / (len(levels) * (len(levels) - 1) / 2)
        else:
            raise ValueError("Invalid method. Choose either 'range' or 'mean'.")

        impacts[dimension] = impact

    # Normalize and create chart (your original implementation)
    total_impact = sum(impacts.values())
    normalized_impacts = {k: (v / total_impact) * 100 for k, v in impacts.items()}
    sorted_impacts = dict(sorted(normalized_impacts.items(), key=lambda item: item[1], reverse=True))

    dimensions = list(sorted_impacts.keys())
    individual_impacts = list(sorted_impacts.values())
    cumulative_impacts = [sum(individual_impacts[:i + 1]) for i in range(len(individual_impacts))]

    fontsize = DEFAULT_FONT_SIZE
    linewidth = DEFAULT_LINE_WIDTH
    plt.rc('font', size=fontsize)
    # Plotting
    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax1.bar(dimensions, individual_impacts, color='skyblue', label="Individual Impact")
    # ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Impact (%)", fontsize=fontsize)
    # ax1.set_title(f"Pareto Analysis - {task}")
    # ax1.tick_params(axis='x', rotation=45)
    # ax1.tick_params(axis='x')
    # ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(dimensions, cumulative_impacts, color='orange', marker='o', label="Cumulative Impact", linewidth=linewidth)
    ax2.axhline(80, color='red', linestyle='--', linewidth=linewidth, label="80% Threshold")
    ax2.set_ylabel("Cumulative Impact (%)")
    # ax2.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.05)
    ax2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pareto_chart_{task}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # dimension_names = ["Aligned?", "Len(Y)", "M-Size (S)", "M-Gen (G)", "C", "Branching Factor"]
    dimension_names = ["AT", "Len(Y)", "S", "G", "C", "Branching Factor"]
    # dimension_names = ["Aligned?", "M-Size (S)", "M-Gen (G)", "C", "Branching Factor"]
    # dimension_names = ["Aligned?", "M-Size", "M-Gen", "#(ICC)", "Branching Factor"]
    output_dir = "pareto_analysis_max_factor"
    # output_dir = "pareto_analysis_max_length_max_impact"
    if "max_impact" in output_dir:
        method="range"
    else:
        method="mean"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze each task individually
    all_datasets = []
    for task in TASK_CONFIGS.keys():
        print(f"\nProcessing task: {task}")
        datasets, set_per_dimensions = load_task_data(task, dimension_names)
        if datasets:
            print(f"Found {len(datasets)} data points")
            print(f"Dimension values found: {[s for s in set_per_dimensions]}")
            pareto_analysis(dimension_names, datasets, method=method, task=task, output_dir=output_dir)
            all_datasets.extend(datasets)
        else:
            print(f"No data found for task {task}")

    # Perform analysis on combined data
    if all_datasets:
        print("\nPerforming analysis on combined data...")
        # pareto_analysis(dimension_names, all_datasets, method="mean", task="combined", output_dir=output_dir)
        pareto_analysis(dimension_names, all_datasets, method=method, task="combined", output_dir=output_dir)


if __name__ == "__main__":
    main()