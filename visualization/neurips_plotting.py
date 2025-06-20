import glob
import copy
import os
import pickle

from draw_synthetic import fig_size

from uncertainty_quantification.visualization_utils import matplotlib_plot, matplotlib_plot_piecewise, \
    ebf_name_visualization_name_mapping, DEFAULT_FIG_SIZE, DEFAULT_FONT_SIZE

if __name__ == '__main__':
    # to run this program, first download piecewise pkl files from the server
    root_dir = "figure_neurips_25"
    piecewise_pkl_filenames = glob.glob(f"{root_dir}/*/piecewise*pkl")
    ebf_key = "perplexity"
    renamed_ebf_key = f"ebf_{ebf_key}"
    output_dir = "figure_neurips_25/outputs"
    os.makedirs(output_dir, exist_ok=True)
    legend_space_inches = 3
    for subdir in os.listdir(root_dir):
        if "outputs" not in subdir:
            print(f"Creating {os.path.join(output_dir, subdir)}")
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    for piecewise_pkl_filename in piecewise_pkl_filenames:
        if "legend" in piecewise_pkl_filename:
            continue
        print(f"processing {piecewise_pkl_filename}")
        x_values, y_values = pickle.load(open(piecewise_pkl_filename, "rb"))
        y_label = ebf_name_visualization_name_mapping(renamed_ebf_key)
        legend_param = {
            "bbox_to_anchor": (1, 0.5),
            "loc": "center left",
            "no_legend": True
        }
        output_filename = piecewise_pkl_filename.replace(root_dir, output_dir)
        matplotlib_plot_piecewise(x_values, y_values, save_path=output_filename.replace(".pkl", "_no_legend.pdf"),
                                  tag=ebf_key, y_label=y_label, legend_params=legend_param, save_cache=False, n_col=1)
        # legend_param['no_legend'] = False
        # fig_size = copy.copy(DEFAULT_FIG_SIZE)
        # fig_size = list(fig_size)
        # fig_size[0] = fig_size[0] + legend_space_inches
        #
        # matplotlib_plot_piecewise(x_values, y_values, save_path=output_filename.replace(".pkl", "_right_legend.pdf"),
        #                           n_col=1, tag=ebf_key, y_label=y_label, figsize=fig_size, legend_params=legend_param, save_cache=False)

    tag = "ebf_perplexity"
    modelwise_pkl_filenames = glob.glob(f"{root_dir}/*/model_wise*pkl")
    for modelwise_pkl_filename in modelwise_pkl_filenames:
        if "legend" in modelwise_pkl_filename:
            continue
        print(f"processing {modelwise_pkl_filename}")
        x_values, y_values = pickle.load(open(modelwise_pkl_filename, "rb"))
        y_label = ebf_name_visualization_name_mapping(renamed_ebf_key)
        legend_param = {
            "bbox_to_anchor": (1, 0.5),
            "loc": "center left",
            "no_legend": True
        }
        output_filename = modelwise_pkl_filename.replace(root_dir, output_dir)
        matplotlib_plot(x_values, y_values, save_path=output_filename.replace(".pkl", "_no_legend.pdf"), tag=tag,
                        y_label=y_label, legend_params=legend_param, save_cache=False, n_col=1)
        # legend_param['no_legend'] = False
        # fig_size = copy.copy(DEFAULT_FIG_SIZE)
        # fig_size = list(fig_size)
        # fig_size[0] = fig_size[0] + legend_space_inches
        # matplotlib_plot(x_values, y_values, save_path=output_filename.replace(".pkl", "_right_legend.pdf"), n_col=1,
        #                 tag=tag, y_label=y_label, figsize=fig_size, legend_params=legend_param, save_cache=False)
