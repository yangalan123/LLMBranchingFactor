# adapted from normal_nlp_application/stat_cognac_increasing_ppl.py
import glob
import os
import re
import sys
import traceback
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import torch
from spacy.compat import pickle

from uncertainty_quantification.stat_computations_from_entropy_profile import parse_args, stat_dict_generation, \
    stat_visualization, ebf_name_visualization_name_mapping, model_name_visualization_name_mapping
from uncertainty_quantification.visualization_utils import matplotlib_plot


def extract_arg_value(arg_string, arg_name):
    pattern = rf'{arg_name}_(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, arg_string)
    if match:
        return match.group(1)
    return None


def process_and_plot_models(ebf_key, primary_families, secondary_families, family_wise_dict,
                            constraints, visualization_dir, mode="source"):
    """Helper function to process and plot model data
    Args:
        mode: "source" or "target" to determine aggregation direction
    """
    aggregation_path = os.path.join(visualization_dir, f"{mode}_model_aggregation")
    os.makedirs(aggregation_path, exist_ok=True)

    for primary_family in primary_families:
        model_values = {}
        model_stds = {}

        for secondary_family in secondary_families:
            # Get the correct key order based on mode
            family_pair = (primary_family, secondary_family) if mode == "source" else (secondary_family, primary_family)
            if family_pair not in family_wise_dict:
                continue

            data = family_wise_dict[family_pair]
            if len(data['y']) < len(constraints):
                print(f"Skipping {family_pair} under {mode}-mode due to insufficient data")
                continue
            model_name = model_name_visualization_name_mapping(secondary_family) if mode == "source" else secondary_family.capitalize()

            model_values[model_name] = {ebf_key: data["y"]}
            model_stds[model_name] = {ebf_key: data["std"]}

        save_path = os.path.join(aggregation_path, f"{mode}_as_{primary_family}.pdf")
        matplotlib_plot(constraints, model_values, save_path,
                        tag=ebf_key,
                        y_label=ebf_name_visualization_name_mapping(ebf_key),
                        fontsize=50, linewidth=5,
                        std_dict=model_stds)


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.output_root_dir, args.ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    # redirect stdout to a file
    # sys.stdout = open(os.path.join(str(output_dir),
    #                                f"stat_cognac_app_ctrlgen_multi_constraints_offset_{args.offset}_high_ent_threshold_{args.high_entropy_threshold}.log"),
    #                   "w")
    # sys.stdout = open(os.path.join(str(output_dir),
    #                                f"stat_cognac_app_ctrlgen_multi_constraints_offset_{args.offset}"
    #                                f"_high_ent_threshold_{args.high_entropy_threshold}"
    #                                f"{'' if args.maxlen is None else '_maxlen_' + str(args.maxlen)}.log"),
    #                   "w")
    constraints = [int(x) for x in args.constraints.split(",")]
    models, prompt_wise_dict = stat_dict_generation(args)

    # plot the prompt-wise stats using plotly
    # only do it with offset = 0
    if args.offset == 0:
        # visualization_dir = os.path.join(str(output_dir),
        #                                  f"visualization_promptwise_{args.offset}{'' if args.maxlen is None else f'_maxlen_{args.maxlen}'}")
        visualization_dir = os.path.join(str(output_dir),
                                         f"visualization_promptwise_{args.offset}{'' if args.maxlen is None else f'_maxlen_{args.maxlen}'}{'_smoothing_' + str(args.smoothing_factor)}")
        stat_visualization(visualization_dir, args, models, prompt_wise_dict, constraints)

        # please note: very early version of storytelling experiments just using one single story, so the codes below might break
        try:
            ckpt_dir = args.ckpt_dir
            # using regex to extract min_p, top_p, max_tokens from ckpt_dir
            # the format is like: output_manual_check_response_storywriting_local_story_gen_full_app_ctrlgen_multi_constraints_max_tokens_1024_min_p_0_top_p_0.9_enforce_min_p_0.1
            # first step: remove "enforce_min_p_0.1" from the end
            arg_str = ckpt_dir.split("_enforce_min_p")[0] if "_enforce_min_p" in ckpt_dir else ckpt_dir
            # min_p = re.search(r"min_p_(\d\.\d)", ckpt_dir).group(1)
            # top_p = re.search(r"top_p_(\d\.\d)", ckpt_dir).group(1)
            # max_tokens = re.search(r"max_tokens_(\d+)", ckpt_dir).group(1)
            min_p = float(extract_arg_value(arg_str, "min_p"))
            top_p = float(extract_arg_value(arg_str, "top_p"))
            max_tokens = int(extract_arg_value(arg_str, "max_tokens"))
            source_dir = args.source_dir
            # clean figure
            fig = go.Figure()

            # find the corresponding files in the source_dir
            x_values = constraints
            ebf_keys = None
            family_wise_dict = dict()
            for model_name in models:
                y_values = defaultdict(list)
                std_values = defaultdict(list)
                source_model_families = None
                for constraint in x_values:
                    if min_p > 0:
                        files = glob.glob(
                            os.path.join(source_dir, f"application_ctrlgen_multi_constraints_{constraint}",
                                         f"{model_name}_response*max_tokens_{max_tokens}*min_p_{min_p}_*.pt.update_full_spectrum"))
                    else:
                        files = glob.glob(
                            os.path.join(source_dir, f"application_ctrlgen_multi_constraints_{constraint}",
                                         f"{model_name}_response*max_tokens_{max_tokens}*top_p_{top_p}_*.pt.update_full_spectrum"))
                    assert len(files) == 1, f"Expect to find 1 file, got {len(files)}: {files}"
                    filename = files[0]
                    metadata_filename = filename.replace(".pt.update_full_spectrum", ".metadata")
                    prompts, roles, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
                    # make sure we are loading the correct metadata
                    assert top_p == args.top_p, f"top_p mismatch: {top_p} vs {args.top_p}"
                    assert max_tokens == args.max_tokens, f"max_tokens mismatch: {max_tokens} vs {args.max_tokens}"
                    # assert min_p == args.min_p, f"min_p mismatch: {min_p} vs {args.min_p}"
                    assert len(all_original_prompts) == len(
                        prompt_wise_dict), f"Prompt number mismatch: {len(all_original_prompts)} vs {len(prompt_wise_dict)}"
                    assert isinstance(all_original_prompts[0],
                                      tuple), f"Expecting tuple, got {type(all_original_prompts[0])}"
                    if source_model_families is None:
                        source_model_families = set([x[1] for x in all_original_prompts])
                    else:
                        assert source_model_families == set([x[1] for x in
                                                             all_original_prompts]), f"Source model families mismatch: {source_model_families} vs {set([x[1] for x in all_original_prompts])}"
                    for source_model_family in source_model_families:
                        if "newyorker" in source_model_family.lower():
                            continue
                        source_model_family_prompt_indices = [idx for idx, x in enumerate(all_original_prompts) if
                                                              x[1] == source_model_family]
                        # y_values[source_model_family].append(
                        #     np.mean([prompt_wise_dict[prompt_i][constraint][model_name]['ebf_mc_ppl'] for prompt_i in
                        #              source_model_family_prompt_indices]))
                        y_values[source_model_family].append(
                            np.mean(
                                [np.exp(prompt_wise_dict[prompt_i][constraint][model_name]['ebf_mean_seq_entropy']) for
                                 prompt_i in
                                 source_model_family_prompt_indices]))
                        std_values[source_model_family].append(
                            np.mean(
                                [prompt_wise_dict[prompt_i][constraint][model_name]['BF_std_perplexity'] for prompt_i in
                                 source_model_family_prompt_indices]))
                for source_model_family in source_model_families:
                    legend_group = f"{source_model_family}->{model_name}"

                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=[y + std for y, std in zip(y_values[source_model_family], std_values[source_model_family])],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        name=f"{legend_group} Upper",
                        legendgroup=legend_group,  # Link to the main trace
                        hoverinfo='skip'
                    ))

                    # Main line
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values[source_model_family],
                        mode='lines+markers',
                        name=legend_group,
                        legendgroup=legend_group,  # Same legend group as bounds
                        fill='tonexty',
                        fillcolor='rgba(68, 68, 68, 0.2)'
                    ))

                    # Lower bound
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=[y - std for y, std in zip(y_values[source_model_family], std_values[source_model_family])],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        name=f"{legend_group} Lower",
                        legendgroup=legend_group,  # Link to the main trace
                        fill='tonexty',
                        hoverinfo='skip'
                    ))
                    family_key = (source_model_family, model_name)
                    assert family_key not in family_wise_dict, f"{family_key} already exist in family_wise_dict!"
                    family_wise_dict[family_key] = {"x": x_values, "y": y_values[source_model_family],
                                                    "std": std_values[source_model_family]}
            fig.write_html(os.path.join(visualization_dir, f"prompt_wise_aggregation_source_model_family.html"))
            pickle.dump(family_wise_dict,
                        open(os.path.join(visualization_dir, f"prompt_wise_aggregation_source_model_family.pkl"), "wb"))
            # for ebf_key in ebf_keys:
            #     save_path = os.path.join(visualization_dir, f"model_wise_comparison_plot_{ebf_key}.pdf")
            #     matplotlib_plot(constraints, plot_y_values_record, save_path, tag=ebf_key,
            #                     y_label=ebf_name_visualization_name_mapping(ebf_key), fontsize=50, linewidth=5)
            source_model_families = set([x[0] for x in family_wise_dict.keys()])
            target_model_families = set([x[1] for x in family_wise_dict.keys()])
            # plot two figures:
            # 1. for given source_model_family, each line represents the performance of different target_model_family
            # 2. for given target_model_family, each line represents the performance of different source_model_family
            # all using matplotlib_plot, put under different subdirectories
            ebf_key = "ebf_perplexity"
            process_and_plot_models(ebf_key, source_model_families, target_model_families,
                                    family_wise_dict, constraints, visualization_dir, mode="source")
            process_and_plot_models(ebf_key, target_model_families, source_model_families,
                                    family_wise_dict, constraints, visualization_dir, mode="target")




        except Exception as e:
            print("Failed to extract metadata from the source_dir")
            print(e)
            # print stack trace
            traceback.print_exc()
