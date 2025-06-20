import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np
from uncertainty_quantification.consts import root_path

DEFAULT_FIG_SIZE=(20, 15)
DEFAULT_FONT_SIZE=50
DEFAULT_LINE_WIDTH=5
DEFAULT_VISUALIZATION_DIR=f"{root_path}/visualization"

def axis_standardize(ax, simple_adjust=False, xlabel_pad=10, ylabel_pad=10):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # Format tick labels
    if not simple_adjust:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val: >8.2f}'))
        # Standardize label padding
        ax.xaxis.labelpad = xlabel_pad
        ax.yaxis.labelpad = ylabel_pad
    else:
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val: >.2f}'))
        pass


def loglik_type_visualization_name_mapping(loglik_type):
    loglik_type_mapping = {
        "output": "Loglik (output)",
        "prompt": "Loglik (prompt)",
        "prompt_min_k": "Min-K"
    }
    return loglik_type_mapping.get(loglik_type, loglik_type)


def model_name_visualization_name_mapping(model_name):
    if "Meta-" in model_name:
        new_model_name = model_name.replace("Meta-", "")
    else:
        new_model_name = model_name
    if "chat" not in new_model_name:
        return new_model_name.replace("b-hf", "B")
    else:
        return new_model_name.replace("b-chat-hf", "B-chat")


def ebf_name_visualization_name_mapping(ebf_name):
    mapping_dict = {
        "ebf_mc_ppl": r"$\log(\text{MC-PPL})$",
        "ebf_mean_seq_entropy": r"$\text{Mean Seq Entropy}$",
        "ebf_ht_ppl": r"$\log(\text{HT-PPL})$",
        "ebf_cond_ppl_prod": r"$\log(\text{Cond-PPL-Prod})$",
        # "ebf_perplexity": r"$\text{Perplexity}$",
        "ebf_perplexity": r"$\text{BF}$",
    }
    return mapping_dict.get(ebf_name, ebf_name)


def matplotlib_plot(constraints_level, yvalues_records, save_path, tag="prompt", y_label="Loglik", fontsize=DEFAULT_FONT_SIZE,
                    linewidth=DEFAULT_LINE_WIDTH, n_col=2, figsize=DEFAULT_FIG_SIZE, base_only=False, save_cache=True, simple_axis_adjust=False,
                    std_dict=None, error_style='band', alpha=0.2, legend_params=None):
    """
    Enhanced plotting function with error bands/bars support
    Args:
        ... (existing args) ...
        std_dict: Dictionary matching structure of yvalues_records containing standard deviation values
        error_style: 'band' or 'bar' for error visualization style
        alpha: Transparency for error bands (only used if error_style='band')
    """
    plt.rc('font', size=fontsize)
    fig, ax1 = plt.subplots(figsize=figsize)

    # Prepare model lists and palettes (same as before)
    models = list(yvalues_records.keys())
    chat_models = [x for x in models if "chat" in x.lower() or "instruct" in x.lower()]
    chat_models.sort()
    non_chat_models = [x for x in models if x not in chat_models]
    non_chat_models.sort()

    _chat_models_palette = ['tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray']
    _non_chat_models_palette = ['tab:green', 'tab:blue', 'tab:purple', 'tab:cyan', 'tab:olive']
    chat_models_palette = _chat_models_palette + _non_chat_models_palette
    non_chat_models_palette = _non_chat_models_palette + _chat_models_palette

    if not base_only:
        chat_lines = []
        # Plot chat models on primary y-axis
        for chat_idx, chat_model in enumerate(chat_models):
            y_values = yvalues_records[chat_model][tag]
            line, = ax1.plot(constraints_level, y_values,
                             marker='^', label=chat_model, linestyle='-',
                             linewidth=linewidth, color=chat_models_palette[chat_idx])
            chat_lines.append(line)

        # Customize primary y-axis
        ax1.set_xlabel('Prompt Complexity (C)', fontsize=fontsize)
        ax1.set_ylabel(y_label + " (Instruct)", fontsize=fontsize, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=fontsize)
        axis_standardize(ax1, simple_axis_adjust)

        # Create secondary y-axis for non-chat models
        ax2 = ax1.twinx()
        non_chat_lines = []

        # Plot non-chat models on secondary y-axis
        for non_chat_idx, non_chat_model in enumerate(non_chat_models):
            y_values = yvalues_records[non_chat_model][tag]
            line, = ax2.plot(constraints_level, y_values,
                             marker='o', label=non_chat_model, linestyle='-.',
                             linewidth=linewidth, color=non_chat_models_palette[non_chat_idx])
            non_chat_lines.append(line)

        ax2.set_ylabel(y_label + " (Base) ", fontsize=fontsize, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fontsize)
        axis_standardize(ax2, simple_axis_adjust)
        all_lines = chat_lines + non_chat_lines

    else:
        all_lines = []
        for non_chat_idx, non_chat_model in enumerate(non_chat_models):
            y_values = yvalues_records[non_chat_model][tag]
            line, = ax1.plot(constraints_level, y_values,
                             marker='o', label=non_chat_model, linestyle='-.',
                             linewidth=linewidth, color=non_chat_models_palette[non_chat_idx])
            all_lines.append(line)

            # Add error visualization if std is provided
            if std_dict is not None and non_chat_model in std_dict:
                std_values = std_dict[non_chat_model][tag]
                if error_style == 'band':
                    ax1.fill_between(constraints_level,
                                     np.array(y_values) - np.array(std_values),
                                     np.array(y_values) + np.array(std_values),
                                     color=non_chat_models_palette[non_chat_idx],
                                     alpha=alpha)
                else:  # error bars
                    ax1.errorbar(constraints_level, y_values,
                                 yerr=std_values,
                                 color=non_chat_models_palette[non_chat_idx],
                                 fmt='none',
                                 capsize=5)

        ax1.set_xlabel('Prompt Complexity (C)', fontsize=fontsize)
        ax1.set_ylabel(y_label, fontsize=fontsize, color='tab:blue')
        axis_standardize(ax1)

    # Legend and final adjustments
    labels = [line.get_label() for line in all_lines]
    plt.subplots_adjust(bottom=0.05)
    if legend_params is not None:
        bbox_to_anchor = legend_params['bbox_to_anchor']
        loc = legend_params['loc']
        no_legend = legend_params['no_legend']
    else:
        bbox_to_anchor = (0.5, -0.01)
        loc="upper center"
        no_legend = False
    if not no_legend:
        fig.legend(all_lines, labels, bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=n_col)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if not no_legend:
        plt.clf()
        plt.close()
    else:
        plt.close(fig)
        fig_legend = plt.figure(figsize=(2, 1), dpi=300)
        fig_legend.legend(all_lines, labels, bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=n_col)
        fig_legend.canvas.draw()
        fig_legend.savefig(save_path.replace(".pdf", "_legend_only.pdf"), bbox_inches='tight', dpi=300)
        plt.close(fig_legend)

    if save_cache:
        if std_dict is None:
            pickle.dump([constraints_level, yvalues_records], open(save_path.replace("pdf", "pkl"), "wb"))
        else:
            pickle.dump([constraints_level, yvalues_records, std_dict], open(save_path.replace("pdf", "pkl"), "wb"))




def matplotlib_plot_piecewise(x_values, yvalues_records, save_path, tag="prompt", y_label="Loglik", fontsize=DEFAULT_FONT_SIZE,
                              linewidth=DEFAULT_LINE_WIDTH, n_col=2, figsize=DEFAULT_FIG_SIZE, save_cache=True, legend_params=None):
    plt.rc('font', size=fontsize)  # Controls default text sizes
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax1 = plt.subplots(figsize=figsize)
    # Plot chat-tuned models on primary y-axis
    # prepare palette for chat-models and non-chat models separately
    model_with_constraints = list(yvalues_records.keys())
    model_with_constraints = [[x.split("_constraint_")[0], int(x.split("_constraint_")[1]), x] for x in
                              model_with_constraints]
    model_with_constraints.sort(key=lambda x: x[1])
    model_names = set([x[0] for x in model_with_constraints])
    assert len(model_names) == 1, "Expecting only one model, got {}".format(model_names)
    model_name = model_with_constraints[0][0]
    # give me 10 different colors with high contrast
    constraint_palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:pink', 'tab:gray',
                          'tab:brown', 'tab:cyan', 'tab:olive']

    all_lines = []
    for idx, (model_name, constraint_level, _model_name_with_constraint) in enumerate(model_with_constraints):
        line, = ax1.plot(x_values[_model_name_with_constraint], yvalues_records[_model_name_with_constraint][tag],
                         marker='^', label="C={}".format(constraint_level), linestyle='-',
                         linewidth=linewidth, color=constraint_palette[idx])
        all_lines.append(line)

    # Customize primary y-axis
    ax1.set_xlabel('Output Position', fontsize=fontsize)
    ax1.set_ylabel(y_label, fontsize=fontsize)
    axis_standardize(ax1)

    # Combine legends, put the legend outside the plot
    # Create a combined legend from both axes
    # lines = [line1, line2, line3, line4]
    # all_lines = chat_lines + non_chat_lines
    labels = [line.get_label() for line in all_lines]

    # Adjust subplot to make room for the legend
    plt.subplots_adjust(bottom=0.05)
    if legend_params is not None:
        bbox_to_anchor = legend_params['bbox_to_anchor']
        loc = legend_params['loc']
        no_legend = legend_params['no_legend']
    else:
        bbox_to_anchor = (0.5, -0.01)
        loc="upper center"
        no_legend = False
    if not no_legend:
        fig.legend(all_lines, labels, title="Prompt Complexity (C)", bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=n_col)
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if not no_legend:
        plt.clf()
        plt.close()
    else:
        plt.close(fig)
        fig_legend = plt.figure(figsize=(2, 1), dpi=300)
        fig_legend.legend(all_lines, labels, title="Prompt Complexity (C)", bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=n_col)
        fig_legend.canvas.draw()
        fig_legend.savefig(save_path.replace(".pdf", "_legend_only.pdf"), bbox_inches='tight', dpi=300)
        plt.close(fig_legend)
    if save_cache:
        pickle.dump([x_values, yvalues_records], open(save_path.replace("pdf", "pkl"), "wb"))
