import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def axis_standardize(ax, simple_adjust=False):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # Format tick labels
    if not simple_adjust:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val: >8.2f}'))
        # Standardize label padding
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
    else:
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val: >.2f}'))
        pass
data = pickle.load(open("loglik_delta.pkl", "rb"))
constraints, local_dict = data
save_path = "log_delta_local.pdf"
def matplotlib_plot(constraints_level, yvalues_records, save_path, tag="prompt", y_label="Loglik", fontsize=50,
                    linewidth=5, n_col=2, figsize=(20, 15), base_only=False, save_cache=True, simple_axis_adjust=False):
    fig, ax1 = plt.subplots(figsize=figsize)
    # Plot chat-tuned models on primary y-axis
    # prepare palette for chat-models and non-chat models separately
    models = list(yvalues_records.keys())
    chat_models = [x for x in models if "chat" in x.lower() or "instruct" in x.lower()]
    chat_models.sort()
    non_chat_models = [x for x in models if x not in chat_models]
    non_chat_models.sort()
    # chat models color should be a bit darker, like red, orange, etc.
    # non-chat models color should be a bit lighter, like green, blue, etc.
    chat_models_palette = ['tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray']
    non_chat_models_palette = ['tab:green', 'tab:blue', 'tab:purple', 'tab:cyan', 'tab:olive']
    plt.rc('font', size=fontsize)  # Controls default text sizes

    if not base_only:
        chat_lines = []
        # line1, = ax1.plot(constraints_level, model_ebf_dict["Llama-2-13B-chat"],
        #                   marker='^', label='Llama-2-13B-chat', linestyle='-',
        #                   linewidth=linewidth, color='tab:red')
        # line2, = ax1.plot(constraints_level, model_ebf_dict["Llama-2-70B-chat"],
        #                   marker='d', label='Llama-2-70B-chat', linestyle='-',
        #                   linewidth=linewidth, color='tab:orange')
        for chat_idx, chat_model in enumerate(chat_models):
            line, = ax1.plot(constraints_level, yvalues_records[chat_model][tag],
                             marker='^', label=chat_model, linestyle='-',
                             linewidth=linewidth, color=chat_models_palette[chat_idx])
            chat_lines.append(line)
        # annotate the plot -- each line should attach the corresponding model name
        # ax1.annotate('Llama-2-13B-chat', xy=(constraints_level[-1], model_ebf_dict["Llama-2-13B-chat"][-1]), xytext=(0, 0), textcoords='offset points')
        # ax1.annotate('Llama-2-70B-chat', xy=(constraints_level[-1], model_ebf_dict["Llama-2-70B-chat"][-1]), xytext=(0, 0), textcoords='offset points')

        # Customize primary y-axis
        ax1.set_xlabel('Constraint Level', fontsize=fontsize)
        # y_label = y_label + f"({tag})"
        ax1.set_ylabel(y_label + " (Instruct)", fontsize=fontsize, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=fontsize)
        axis_standardize(ax1, simple_axis_adjust)

        # Create secondary y-axis for non-chat models
        ax2 = ax1.twinx()
        non_chat_lines = []
        # line3, = ax2.plot(constraints_level, model_ebf_dict["Llama-2-13B"],
        #                   marker='o', label='Llama-2-13B', linestyle='-.',
        #                   linewidth=linewidth, color='tab:green')
        # line4, = ax2.plot(constraints_level, model_ebf_dict["Llama-2-70B"],
        #                   marker='s', label='Llama-2-70B', linestyle='-.',
        #                   linewidth=linewidth, color='tab:blue')
        for non_chat_idx, non_chat_model in enumerate(non_chat_models):
            line, = ax2.plot(constraints_level, yvalues_records[non_chat_model][tag],
                             marker='o', label=non_chat_model, linestyle='-.',
                             linewidth=linewidth, color=non_chat_models_palette[non_chat_idx])
            non_chat_lines.append(line)
        # annotate the plot -- each line should attach the corresponding model name
        # ax2.annotate('Llama-2-13B', xy=(constraints_level[-1], model_ebf_dict["Llama-2-13B"][-1]), xytext=(0, 0), textcoords='offset points')
        # ax2.annotate('Llama-2-70B', xy=(constraints_level[-1], model_ebf_dict["Llama-2-70B"][-1]), xytext=(0, 0), textcoords='offset points')

        # Customize secondary y-axis
        # if "diff" not in flag:
        #     ax2.set_ylabel('Log(PPL) (Non-Chat)', fontsize=fontsize, color='tab:blue')
        # else:
        ax2.set_ylabel(y_label + " (Base) ", fontsize=fontsize, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fontsize)
        axis_standardize(ax2, simple_axis_adjust)
        all_lines = chat_lines + non_chat_lines

        # Combine legends, put the legend outside the plot
        # Create a combined legend from both axes
        # lines = [line1, line2, line3, line4]
    else:
        all_lines = []
        for non_chat_idx, non_chat_model in enumerate(non_chat_models):
            line, = ax1.plot(constraints_level, yvalues_records[non_chat_model][tag],
                             marker='o', label=non_chat_model, linestyle='-.',
                             linewidth=linewidth, color=non_chat_models_palette[non_chat_idx])
            all_lines.append(line)
        # Customize primary y-axis
        ax1.set_xlabel('Constraint Level', fontsize=fontsize)
        ax1.set_ylabel(y_label, fontsize=fontsize, color='tab:blue')
        axis_standardize(ax1)

    labels = [line.get_label() for line in all_lines]
    # Adjust subplot to make room for the legend
    # plt.subplots_adjust(bottom=0.05)
    fig.legend(all_lines, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=n_col)
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


matplotlib_plot(constraints, local_dict, save_path, tag="loglik_delta", y_label="LogLik Diff", fontsize=50, linewidth=5,
                figsize=(20, 15), simple_axis_adjust=True)
