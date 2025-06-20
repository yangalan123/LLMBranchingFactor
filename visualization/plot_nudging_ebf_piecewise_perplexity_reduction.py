import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fontsize = 50
    linewidth = 5
    figsize = (20, 15)
    # model_size = "8B"
    model_size = "70B"
    flags = ["Nudging", "Base (70B)", "Aligned (8B)"]
    constraints = ["-1", "5"]
    for root_dir, constraint in zip(["just_eval_instruct_nudging", "mmlu_wa_filler_nudging"], constraints):
        plt.rc('font', size=fontsize)
        plt.figure(figsize=figsize)
        key = f"Llama-3-{model_size}_constraint_{constraint}"
        filepaths = [
            f"{root_dir}/nudging_pkl/nudging_ebf_max_round_100.pkl",
            f"{root_dir}/normal_pkl/piecewise_ebf_Llama-3-{model_size}_perplexity.pkl",
            f"{root_dir}/normal_pkl/piecewise_ebf_Llama-3-8B-Instruct_perplexity.pkl"
        ]
        keys = [
            f"Llama-3-{model_size}_constraint_{constraint}",
            f"Llama-3-{model_size}_constraint_{constraint}",
            f"Llama-3-8B-Instruct_constraint_{constraint}",
        ]
        colors = ["red", "black", "blue", "green"]
        for flag, filepath, color, key in zip(flags, filepaths, colors, keys):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            x_values = data[0][key]
            x_values = [x for x in x_values if x < 150]
            y_values = data[1][key]["perplexity"][:len(x_values)]
            plt.plot(x_values, y_values, color=color, label=flag, linewidth=linewidth)
        plt.xlabel("Output Position")
        plt.ylabel("BF")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"{root_dir}/{flag.lower().replace(' ', '_')}_piecewise_ebf_Llama-3-{model_size}_perplexity.png")
        # plt.savefig(f"{root_dir}/all_priming_nudging_piecewise_ebf_Llama-3-{model_size}_perplexity.png")
        plt.savefig(f"{root_dir}/nudging_{root_dir.split('_nudging')[0]}_piecewise_ebf_Llama-3-{model_size}_perplexity.pdf", dpi=300)
        plt.clf()