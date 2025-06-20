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

from mmlu_metrics import mmlu_metrics
from uncertainty_quantification.stat_computations_from_entropy_profile import parse_args, stat_dict_generation, \
    stat_visualization, ebf_name_visualization_name_mapping, model_name_visualization_name_mapping
from uncertainty_quantification.visualization_utils import matplotlib_plot


def extract_arg_value(arg_string, arg_name):
    pattern = rf'{arg_name}_(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, arg_string)
    if match:
        return match.group(1)
    return None


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.output_root_dir, args.ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    # redirect stdout to a file
    sys.stdout = open(os.path.join(str(output_dir),
                                   f"stat_cognac_app_ctrlgen_multi_constraints_offset_{args.offset}"
                                   f"_high_ent_threshold_{args.high_entropy_threshold}"
                                   f"{'' if args.maxlen is None else '_maxlen_' + str(args.maxlen)}.log"),
                      "w")
    constraints = [int(x) for x in args.constraints.split(",")]
    models, prompt_wise_dict = stat_dict_generation(args)
    print(f"Completed Models: {models}")

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
            min_p = float(extract_arg_value(arg_str, "min_p"))
            top_p = float(extract_arg_value(arg_str, "top_p"))
            max_tokens = int(extract_arg_value(arg_str, "max_tokens"))
            source_dir = args.source_dir
            # clean figure
            fig = go.Figure()

            # find the corresponding files in the source_dir
            x_values = constraints
            ebf_keys = None
            source_y_values_record = dict()
            source_performance_values_record = dict()
            source_answer_parse_success_rate = dict()
            for model_name in models:
                y_values = defaultdict(list)
                performance_values = defaultdict(list)
                answer_parse_success_rate = defaultdict(list)
                source_model_families = None
                for constraint in x_values:
                    if min_p > 0:
                        files = glob.glob(
                            os.path.join(source_dir, f"application_ctrlgen_multi_constraints_{constraint}",
                                         # f"{model_name}_response*max_tokens_{max_tokens}*min_p_{min_p}_*.pt.update_full_spectrum"))
                                         f"{model_name}_response*max_tokens_{max_tokens}*min_p_{min_p}"
                                         f"_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
                        )
                    else:
                        files = glob.glob(
                            os.path.join(source_dir, f"application_ctrlgen_multi_constraints_{constraint}",
                                         # f"{model_name}_response*max_tokens_{max_tokens}*top_p_{top_p}_*.pt.update_full_spectrum"))
                                         f"{model_name}_response*max_tokens_{max_tokens}*top_p_{top_p}"
                                         f"_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
                        )
                        assert len(files) == 1, f"Expect to find 1 file, got {len(files)}: {files}"
                    filename = files[0]
                    metadata_filename = filename.replace(".pt.update_full_spectrum", ".metadata")
                    prompts, roles, answers, options, all_constrained_prompts, metadata_args = torch.load(metadata_filename)
                    # make sure we are loading the correct metadata
                    assert top_p == metadata_args.top_p, f"top_p mismatch: {top_p} vs {metadata_args.top_p}"
                    assert max_tokens == metadata_args.max_tokens, f"max_tokens mismatch: {max_tokens} vs {metadata_args.max_tokens}"
                    # assert min_p == args.min_p, f"min_p mismatch: {min_p} vs {args.min_p}"
                    assert len(all_constrained_prompts) == len(
                        prompt_wise_dict), f"Prompt number mismatch: {len(all_constrained_prompts)} vs {len(prompt_wise_dict)}"
                    # assert isinstance(all_constrained_prompts[0],
                    #                   tuple), f"Expecting tuple, got {type(all_original_prompts[0])}"
                    _source_model_families = set([x.split("\t")[0] for x in roles])
                    if source_model_families is None:
                        # cot/standard/filler_cot/...
                        source_model_families = _source_model_families
                    else:
                        assert source_model_families == _source_model_families, f"Source model families mismatch: {source_model_families} vs {_source_model_families}"
                    for source_model_family in source_model_families:
                        source_model_family_prompt_indices = [idx for idx, x in enumerate(roles) if
                                                              x.split("\t")[0] == source_model_family]
                        y_values[source_model_family].append(
                            np.mean([prompt_wise_dict[prompt_i][constraint][model_name]['ebf_mc_ppl'] for prompt_i in
                                     source_model_family_prompt_indices]))
                        _mmlu_metrics = [
                            mmlu_metrics(
                                prompt_wise_dict[prompt_i]['outputs'][constraint][model_name],
                                answers[prompt_i], model_name, source_model_family,
                                self_consistency=False)
                            for prompt_i in source_model_family_prompt_indices]
                        performance_values[source_model_family].append(
                            np.mean(
                                [x[0] for x in _mmlu_metrics]
                            )
                        )
                        answer_parse_success_rate[source_model_family].append(
                            np.mean([x[1] for x in _mmlu_metrics])
                        )
                        if source_model_family not in source_y_values_record:
                            source_y_values_record[source_model_family] = dict()
                            source_performance_values_record[source_model_family] = dict()
                            source_answer_parse_success_rate[source_model_family] = dict()
                        ebf_keys = [x for x in prompt_wise_dict[source_model_family_prompt_indices[0]][constraint][
                            model_name].keys() if "ebf" in x]
                        _model_name = model_name_visualization_name_mapping(model_name)
                        if _model_name not in source_y_values_record[source_model_family]:
                            source_y_values_record[source_model_family][_model_name] = dict()
                            source_performance_values_record[source_model_family][_model_name] = {
                                "Accuracy": performance_values[source_model_family]
                            }
                            source_answer_parse_success_rate[source_model_family][_model_name] = {
                                "Parse Failure Rate": answer_parse_success_rate[source_model_family]
                            }
                        for ebf_key in ebf_keys:
                            if ebf_key not in source_y_values_record[source_model_family][_model_name]:
                                source_y_values_record[source_model_family][_model_name][ebf_key] = []
                            source_y_values_record[source_model_family][_model_name][ebf_key].append(
                                np.mean([prompt_wise_dict[prompt_i][constraint][model_name][ebf_key] for prompt_i in
                                         source_model_family_prompt_indices]))
                for source_model_family in source_model_families:
                    fig.add_trace(go.Scatter(x=x_values, y=y_values[source_model_family], mode='lines+markers',
                                             name=f"{source_model_family}->{model_name}"))
            fig.write_html(os.path.join(visualization_dir, f"prompt_wise_aggregation_source_model_family.html"))
            for source_model_family in source_model_families:
                for ebf_key in ebf_keys:
                    save_path = os.path.join(visualization_dir,
                                             f"model_wise_comparison_{source_model_family}_{ebf_key}.pdf")
                    matplotlib_plot(constraints, source_y_values_record[source_model_family], save_path, tag=ebf_key,
                                    y_label=ebf_name_visualization_name_mapping(ebf_key), fontsize=50, linewidth=5)
                    performance_fig_path = os.path.join(visualization_dir,
                                                        f"performance_{source_model_family}.pdf")
                    matplotlib_plot(constraints, source_performance_values_record[source_model_family], performance_fig_path, tag="Accuracy",
                                    y_label="Accuracy", fontsize=50, linewidth=5)
                    parse_rate_fig_path = os.path.join(visualization_dir,
                                                        f"parse_rate_{source_model_family}.pdf")
                    matplotlib_plot(constraints, source_answer_parse_success_rate[source_model_family], parse_rate_fig_path,
                                    tag="Parse Failure Rate",
                                    y_label="Parse Failure Rate", fontsize=50, linewidth=5)
            # exclude the chat/instruct models
            for source_model_family in source_model_families:
                for ebf_key in ebf_keys:
                    save_path = os.path.join(visualization_dir,
                                             f"model_wise_comparison_base_{source_model_family}_{ebf_key}.pdf")
                    matplotlib_plot(constraints, source_y_values_record[source_model_family], save_path, tag=ebf_key,
                                    y_label=ebf_name_visualization_name_mapping(ebf_key), fontsize=50, linewidth=5,
                                    base_only=True)
                    performance_fig_path = os.path.join(visualization_dir,
                                                        f"performance_base_{source_model_family}.pdf")
                    matplotlib_plot(constraints, source_performance_values_record[source_model_family], performance_fig_path, tag="Accuracy",
                                    y_label="Accuracy", fontsize=50, linewidth=5, base_only=True)
                    parse_rate_fig_path = os.path.join(visualization_dir,
                                                       f"parse_rate_base_{source_model_family}.pdf")
                    matplotlib_plot(constraints, source_answer_parse_success_rate[source_model_family], parse_rate_fig_path,
                                    tag="Parse Failure Rate",
                                    y_label="Parse Failure Rate", fontsize=50, linewidth=5, base_only=True)

        except Exception as e:
            print("Failed to extract metadata from the source_dir")
            print(e)
            # print stack trace
            traceback.print_exc()
