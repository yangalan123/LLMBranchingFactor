import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import torch

from uncertainty_quantification.diversity_metrics import compute_diversity_metrics
from uncertainty_quantification.uncertainty_computation import \
    from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies, \
    compute_ebf_from_length_wise_and_sample_wise_entropies
from uncertainty_quantification.visualization_utils import matplotlib_plot, matplotlib_plot_piecewise, \
    model_name_visualization_name_mapping, ebf_name_visualization_name_mapping


def parse_args():
    parser = argparse.ArgumentParser(description='StatParser.')
    parser.add_argument('--ckpt_dir', type=str, default="output_manual_check_cognac_app_ctrlgen_multi_constraints",
                        help='model name')
    parser.add_argument("--high_entropy_threshold", type=float, default=0.7, help="high entropy threshold")
    parser.add_argument("--output_root_dir", type=str, default="stat_cognac_app_ctrlgen_multi_constraints", )
    parser.add_argument("--offset", type=int, default=0, help="offset for the prompt")
    parser.add_argument("--maxlen", type=int, default=None, help="max length for the prompt")
    parser.add_argument("--constraints", type=str, default="1,2,3,4,5", help="constraints")
    parser.add_argument("--no_modelwise_plot", action="store_true", help="no modelwise plot")
    parser.add_argument("--piecewise_ebf", action="store_true", help="piecewise ebf computation")
    parser.add_argument("--piecewise_ebf_position", type=int, default=5, help="piecewise ebf position")
    parser.add_argument("--smoothing_factor", type=float, default=1, help="smoothing factor")
    parser.add_argument("--min_example_per_position", type=int, default=25, help="minimum example per position")
    # only useful for storytelling experiments
    parser.add_argument("--source_dir", type=str, default="", help="only useful for storytelling experiments")
    parser.add_argument("--additional_file_search_pattern", type=str, default="", help="additional file search pattern")
    args = parser.parse_args()
    return args


def ema_smooth(data, alpha=0.1):
    smoothed_data = []
    for i, x in enumerate(data):
        if i == 0:
            smoothed_data.append(x)
        else:
            smoothed_data.append(alpha * x + (1 - alpha) * smoothed_data[-1])
    return smoothed_data


def stat_dict_generation(args):
    ckpt_dir = args.ckpt_dir
    ckpt_names = glob.glob(os.path.join(ckpt_dir, f"ctrlgen_multi_constraints_investigation_increasing_PPL_*.pt"))
    # extract the model names from the ckpt_names
    models = [os.path.basename(x).split("PPL_")[-1].split(".")[0] for x in ckpt_names if "llama" in x.lower()]
    print("Models:", models)
    constraints = [int(x) for x in args.constraints.split(",")]
    ebf_types = None
    prompt_wise_dict = dict()
    completed_models = []
    for model_name in models:
        ckpt_name = os.path.join(ckpt_dir,
                                 f"ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt")
        if not os.path.exists(ckpt_name):
            print(f"File not found: {ckpt_name}")
            continue
        ckpt = torch.load(ckpt_name)
        # verify the dump file a bit
        _constraint_keys = list(ckpt.keys())
        if set(_constraint_keys).issubset(constraints) and len(_constraint_keys) < len(constraints):
            print(
                f"Model {model_name} has not completed all constraints, missing: {set(constraints) - set(_constraint_keys)}")
            continue
        else:
            completed_models.append(model_name)

        ps = list(ckpt[constraints[0]][model_name].keys())
        sample_p = ps[0]
        diagnostics = dict()
        for constraint in sorted(_constraint_keys):
            # for constraint in constraints:
            # task 1: count the output with too short length -- it may contain a bunch of <unk>, or simply just </s>
            entropy_profile = ckpt[constraint][model_name][sample_p]
            diagnostics[constraint] = dict()
            diagnostics[constraint]["prompt_num"] = len(entropy_profile)
            diagnostics[constraint]['output_num'] = sum(
                len(all_output_texts) for _, all_output_texts, _, _ in entropy_profile)
            diagnostics[constraint]['prompt_wise_stats'] = defaultdict(list)
            for idx, (prompt, all_output_texts, token_texts, entropies) in enumerate(entropy_profile):
                if idx not in prompt_wise_dict:
                    # whole_seqs_entropy is the entropy of the whole sequence per sample -- we will use it for piecewise ebf computation
                    # collect outputs for task-specific performance evaluation (e.g., mmlu)
                    prompt_wise_dict[idx] = {"prompts": [], "entropies": {}, "outputs": {}, "diversity": {}}
                if constraint not in prompt_wise_dict[idx]:
                    prompt_wise_dict[idx][constraint] = dict()
                    prompt_wise_dict[idx]['entropies'][constraint] = dict()
                    prompt_wise_dict[idx]['outputs'][constraint] = dict()
                    prompt_wise_dict[idx]['diversity'][constraint] = dict()
                if model_name not in prompt_wise_dict[idx][constraint]:
                    prompt_wise_dict[idx][constraint][model_name] = dict()
                prompt_wise_dict[idx]["prompts"].append((prompt, model_name))
                prompt_wise_dict[idx]['outputs'][constraint][model_name] = all_output_texts
                prompt_wise_dict[idx]['diversity'][constraint][model_name] = compute_diversity_metrics(token_texts)
                num_unique_outputs = len(set(all_output_texts))
                if args.offset >= max([len(x) for x in entropies]):
                    continue
                length_wise_entropies, sample_wise_seq_mean_entropies = from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies(
                    entropies, maxlen=args.maxlen)
                ebf = compute_ebf_from_length_wise_and_sample_wise_entropies(length_wise_entropies,
                                                                             sample_wise_seq_mean_entropies,
                                                                             offset=args.offset)
                prompt_wise_dict[idx]['entropies'][constraint][model_name] = entropies
                for ebf_key in ebf.keys():
                    diagnostics[constraint]['prompt_wise_stats'][f'ebf_{ebf_key}'].append(ebf[ebf_key])
                    prompt_wise_dict[idx][constraint][model_name][f'ebf_{ebf_key}'] = ebf[ebf_key]
                    if ebf_key == "mean_seq_entropy":
                        prompt_wise_dict[idx][constraint][model_name][f"BF_std_{ebf_key}"] = np.std(
                            sample_wise_seq_mean_entropies)
                    elif ebf_key == "perplexity":
                        prompt_wise_dict[idx][constraint][model_name][f"BF_std_{ebf_key}"] = np.std(
                            np.exp(sample_wise_seq_mean_entropies))
                for diversity_key in prompt_wise_dict[idx]['diversity'][constraint][model_name].keys():
                    prompt_wise_dict[idx][constraint][model_name][f'ebf_{diversity_key}'] = \
                    prompt_wise_dict[idx]['diversity'][constraint][model_name][diversity_key]

                if ebf_types is None:
                    ebf_types = [f'ebf_{ebf_key}' for ebf_key in ebf.keys()]
                # start counting too short outputs
                too_short_output_count = 0
                for token_text_i, token_text in enumerate(token_texts):
                    bool_too_short_flag = False
                    if len(token_text) <= 2 or len(token_text) <= args.offset:
                        # mostly just ['</s>']
                        bool_too_short_flag = True
                    for empty_token in ["<unk>", "O", '<0x0A>']:
                        if token_text.count(empty_token) / len(token_text) >= 0.8:
                            bool_too_short_flag = True
                            break
                    if bool_too_short_flag:
                        too_short_output_count += 1
                    else:
                        high_entropy_token_num = 0
                        non_zero_entropy_token_num = 0
                        avg_non_zero_entropy_position = None
                        _entropy = entropies[token_text_i]
                        non_zero_entropy_position = []
                        high_entropy_position = []
                        # for _entropy_i, _entropy_elem in enumerate(_entropy):
                        for _entropy_i in range(args.offset, len(_entropy)):
                            _entropy_elem = _entropy[_entropy_i]
                            if _entropy_elem > 0:
                                non_zero_entropy_token_num += 1
                                non_zero_entropy_position.append(_entropy_i)
                            if _entropy_elem >= args.high_entropy_threshold:
                                high_entropy_token_num += 1
                                high_entropy_position.append(_entropy_i)
                            # if _entropy_elem >= 0.7:
                        avg_non_zero_entropy_position = np.mean(
                            non_zero_entropy_position) if non_zero_entropy_position else None
                        std_non_zero_entropy_position = np.std(
                            non_zero_entropy_position) if non_zero_entropy_position else None
                        avg_high_entropy_position = np.mean(high_entropy_position) if high_entropy_position else None
                        std_high_entropy_position = np.std(high_entropy_position) if high_entropy_position else None
                        if avg_non_zero_entropy_position is not None:
                            diagnostics[constraint]['prompt_wise_stats']['mean_non_zero_entropy_position'].append(
                                avg_non_zero_entropy_position)
                            diagnostics[constraint]['prompt_wise_stats']['std_non_zero_entropy_position'].append(
                                std_non_zero_entropy_position)
                            diagnostics[constraint]['prompt_wise_stats'][
                                'mean_non_zero_entropy_position_relative'].append(
                                avg_non_zero_entropy_position / len(token_text))
                            diagnostics[constraint]['prompt_wise_stats'][
                                'std_non_zero_entropy_position_relative'].append(
                                std_non_zero_entropy_position / len(token_text))
                            diagnostics[constraint]['prompt_wise_stats']['high_entropy_token_num'].append(
                                high_entropy_token_num)
                            diagnostics[constraint]['prompt_wise_stats']['non_zero_entropy_token_num'].append(
                                non_zero_entropy_token_num)
                        if avg_high_entropy_position is not None:
                            diagnostics[constraint]['prompt_wise_stats']['avg_high_entropy_position'].append(
                                avg_high_entropy_position)
                            diagnostics[constraint]['prompt_wise_stats']['std_high_entropy_position'].append(
                                std_high_entropy_position)
                            diagnostics[constraint]['prompt_wise_stats']['avg_high_entropy_position_relative'].append(
                                avg_high_entropy_position / len(token_text))
                            diagnostics[constraint]['prompt_wise_stats']['std_high_entropy_position_relative'].append(
                                std_high_entropy_position / len(token_text))

                        # normal output, collect other stats
                diagnostics[constraint]['prompt_wise_stats']['num_unique_outputs'].append(num_unique_outputs)
                diagnostics[constraint]['prompt_wise_stats']['too_short_output_count'].append(too_short_output_count)
            diagnostics[constraint]['prompt_wise_stats']['avg_num_unique_outputs'] = np.mean(
                diagnostics[constraint]['prompt_wise_stats']['num_unique_outputs'])
            diagnostics[constraint]['prompt_wise_stats']['std_too_short_output_count'] = np.std(
                diagnostics[constraint]['prompt_wise_stats']['too_short_output_count'])
            diagnostics[constraint]['prompt_wise_stats']['avg_too_short_output_count'] = np.mean(
                diagnostics[constraint]['prompt_wise_stats']['too_short_output_count'])
            diagnostics[constraint]['prompt_wise_stats']['max_too_short_output_count'] = np.max(
                diagnostics[constraint]['prompt_wise_stats']['too_short_output_count'])
            del diagnostics[constraint]['prompt_wise_stats']['num_unique_outputs']
            del diagnostics[constraint]['prompt_wise_stats']['too_short_output_count']
            for metric in ["mean_non_zero_entropy_position", "mean_non_zero_entropy_position_relative",
                           "high_entropy_token_num", "non_zero_entropy_token_num",
                           "std_non_zero_entropy_position", "std_non_zero_entropy_position_relative",
                           "avg_high_entropy_position", "std_high_entropy_position",
                           "avg_high_entropy_position_relative", "std_high_entropy_position_relative"] + ebf_types:
                if metric in diagnostics[constraint]['prompt_wise_stats']:
                    diagnostics[constraint]['prompt_wise_stats'][f"avg_{metric}"] = np.mean(
                        diagnostics[constraint]['prompt_wise_stats'][metric])
                    del diagnostics[constraint]['prompt_wise_stats'][metric]

        print(f"Model: {model_name}")
        overall_stat = defaultdict(list)
        for constraint in sorted(_constraint_keys):
            overall_stat['prompt_num'].append(diagnostics[constraint]['prompt_num'])
            overall_stat['output_num'].append(diagnostics[constraint]['output_num'])
            for k, v in diagnostics[constraint]['prompt_wise_stats'].items():
                overall_stat[k].append(v)
        # print the overall stats
        print("Overall stats:")
        # give an ordering for the overall stats
        overall_stat_keys = [
            "prompt_num", "output_num", "avg_num_unique_outputs",
            "std_too_short_output_count", "avg_too_short_output_count", "max_too_short_output_count",
            "avg_mean_non_zero_entropy_position", "avg_std_non_zero_entropy_position",
            "avg_mean_non_zero_entropy_position_relative", "avg_std_non_zero_entropy_position_relative",
            "avg_high_entropy_token_num", "avg_non_zero_entropy_token_num",
            "avg_avg_high_entropy_position", "avg_std_high_entropy_position",
            "avg_avg_high_entropy_position_relative", "avg_std_high_entropy_position_relative",
            "avg_ebf_mc_ppl"
        ]
        for k in overall_stat_keys:
            v = overall_stat[k]
            # for each number in v, print with 2 decimal points
            # get rounded number first for each element in v
            rounded_v = [round(x, 2) for x in v]
            print(f"{k}: {np.mean(rounded_v):.2f} ({rounded_v})")
    return completed_models, prompt_wise_dict

def stat_piecewise(args, prompts, prompt_wise_dict, models, constraints, visualization_dir, smooth, fixed_start=False):
    # piecewise ebf computation
    # x-axis: piecewise position, e.g., [0, 15], [15, 30],....
    # y-axis: ebf value averaged in the piecewise position
    # label: model_name-constraint
    fig = go.Figure()
    model_constraint_piecewise_aggregated = dict()
    piecewise_ebf_keys = None
    if fixed_start:
        saved_filename_suffix = "_fixed_start"
    else:
        saved_filename_suffix = ""
    for model_name in models:
        model_constraint_piecewise_aggregated[model_name] = {x: dict() for x in constraints}
        for constraint in constraints:
            for prompt_i in prompts:
                # position_wise_entropies = prompt_wise_dict[prompt_i]['position_wise_entropy'][constraint][
                #     model_name]
                entropies = prompt_wise_dict[prompt_i]['entropies'][constraint][model_name]
                length_to_compute = min([len(x) for x in entropies])
                for position in range(0, length_to_compute, args.piecewise_ebf_position):
                    if position + args.piecewise_ebf_position >= length_to_compute:
                        continue
                    if not fixed_start:
                        offset = position
                    else:
                        offset = 0
                    piecewise_length_wise_entropies, piecewise_sample_wise_seq_mean_entropies = from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies(
                        entropies, offset=offset, maxlen=position + args.piecewise_ebf_position)
                    flag_check_min_example_per_position = False
                    for _position in piecewise_length_wise_entropies.keys():
                        if len(piecewise_length_wise_entropies[_position]) < args.min_example_per_position:
                            flag_check_min_example_per_position = True
                            print("Position {} has {} examples, less than the threshold {}, ignored this piece".format(
                                _position,
                                len(
                                    piecewise_length_wise_entropies[
                                        _position]),
                                args.min_example_per_position))
                            break
                    if flag_check_min_example_per_position:
                        continue

                    ebf = compute_ebf_from_length_wise_and_sample_wise_entropies(piecewise_length_wise_entropies,
                                                                                 piecewise_sample_wise_seq_mean_entropies,
                                                                                 offset=offset)
                    if position not in model_constraint_piecewise_aggregated[model_name][constraint]:
                        model_constraint_piecewise_aggregated[model_name][constraint][position] = dict()
                    if piecewise_ebf_keys is None:
                        piecewise_ebf_keys = list(ebf.keys())
                    for ebf_key in piecewise_ebf_keys:
                        if ebf_key not in model_constraint_piecewise_aggregated[model_name][constraint][position]:
                            model_constraint_piecewise_aggregated[model_name][constraint][position][ebf_key] = []
                        model_constraint_piecewise_aggregated[model_name][constraint][position][ebf_key].append(
                            ebf[ebf_key])
    constraint_dict = dict()
    x_values_dict = dict()
    for ebf_key in piecewise_ebf_keys:
        for model_name in models:
            for constraint in constraints:
                x_values = list(model_constraint_piecewise_aggregated[model_name][constraint].keys())
                x_values.sort()
                new_x_values = []
                new_y_values = []
                for position in x_values:
                    _y_value = np.nanmean(
                        model_constraint_piecewise_aggregated[model_name][constraint][position][ebf_key])
                    if np.isnan(_y_value) or np.isinf(_y_value):
                        continue
                    new_x_values.append(position)
                    new_y_values.append(_y_value)
                new_y_values = smooth(new_y_values)
                fig.add_trace(
                    go.Scatter(x=new_x_values, y=new_y_values, mode='lines+markers',
                               name=f"{model_name}_{constraint}"))
                renamed_model_name = model_name_visualization_name_mapping(model_name)
                new_key = f"{renamed_model_name}_constraint_{constraint}"
                if new_key not in constraint_dict:
                    constraint_dict[new_key] = dict()
                    x_values_dict[new_key] = new_x_values
                constraint_dict[new_key][ebf_key] = new_y_values
        # add x-axis label, y-axis label, title
        fig.update_layout(
            xaxis_title="Piecewise Position",
            yaxis_title=ebf_key,
            title=f"Piecewise EBF {ebf_key}"
        )
        fig.write_html(os.path.join(visualization_dir, f"piecewise_ebf_{ebf_key}{saved_filename_suffix}.html"))
    for model_name in models:
        for ebf_key in piecewise_ebf_keys:
            renamed_model_name = model_name_visualization_name_mapping(model_name)
            save_path = os.path.join(visualization_dir, f"piecewise_ebf_{renamed_model_name}_{ebf_key}{saved_filename_suffix}.pdf")
            local_dict = {k: v for k, v in constraint_dict.items() if f"{renamed_model_name}_constraint_" in k}
            renamed_ebf_key = f"ebf_{ebf_key}"
            matplotlib_plot_piecewise(x_values_dict, local_dict, save_path,
                                      y_label=ebf_name_visualization_name_mapping(renamed_ebf_key), tag=ebf_key,
                                      fontsize=50, linewidth=5, n_col=3)
            print("Saved to", save_path)


def stat_visualization(visualization_dir, args, models, prompt_wise_dict, constraints):
    os.makedirs(visualization_dir, exist_ok=True)
    prompts = list(prompt_wise_dict.keys())
    prompts.sort()
    all_prompts = []
    prompt_dict_keys = list(prompt_wise_dict[prompts[0]][constraints[0]][models[0]].keys())
    ebf_keys = [x for x in prompt_dict_keys if "ebf" in x]
    smooth = lambda x: ema_smooth(x, alpha=args.smoothing_factor)
    for ebf_key in ebf_keys:
        fig = go.Figure()
        # we have to use idx to refer to prompts, as different models have different prompts for the same content
        for prompt_i in prompts:
            all_model_prompts = prompt_wise_dict[prompt_i]["prompts"]
            all_model_prompts.sort(key=lambda x: len(x[0]))
            all_prompts.append(all_model_prompts[-1])
            x_values = constraints
            for model_name in models:
                if model_name not in prompt_wise_dict[prompt_i][constraints[0]]:
                    raise ValueError(
                        f"Model {model_name} not found in prompt {prompt_i}, constraints {constraints[0]}, current available models: {prompt_wise_dict[prompt_i][constraints[0]].keys()}")
                y_values = [prompt_wise_dict[prompt_i][constraint][model_name][ebf_key] for constraint in x_values]
                fig.add_trace(
                    go.Scatter(x=x_values, y=smooth(y_values), mode='lines+markers', name=f"{model_name}_{prompt_i}"))
        fig.write_html(os.path.join(visualization_dir, f"prompt_wise_aggregation_{ebf_key}.html"))
        print("Saved to", os.path.join(visualization_dir, f"prompt_wise_aggregation_{ebf_key}.html"))
        if not args.no_modelwise_plot:
            # for storytelling experiments, as they have special source_model -> target_model aggregation mapping
            # so this part will not be used for storytelling experiments. Instead, the separate snippet will handle it
            # create prompt-aggregated plots
            # clean the figure
            fig = go.Figure()
            model_wise_aggregated = dict()
            std_key = "BF_std_" + ebf_key.split("ebf_")[-1]
            model_wise_aggregated_std = dict()
            for model_name in models:
                model_wise_aggregated[model_name] = {x: [] for x in constraints}
                model_wise_aggregated_std[model_name] = {x: [] for x in constraints}
            for prompt_i in prompts:
                for model_name in models:
                    for constraint in constraints:
                        if model_name not in prompt_wise_dict[prompt_i][constraint]:
                            raise ValueError(
                                f"Model {model_name} not found in prompt {prompt_i}, constraints {constraint}, current available models: {prompt_wise_dict[prompt_i][constraint].keys()}")
                        model_wise_aggregated[model_name][constraint].append(
                            prompt_wise_dict[prompt_i][constraint][model_name][ebf_key])
                        if std_key in prompt_dict_keys:
                            model_wise_aggregated_std[model_name][constraint].append(
                                prompt_wise_dict[prompt_i][constraint][model_name][std_key])
            for model_name in models:
                x_values = constraints
                y_values = [np.mean(model_wise_aggregated[model_name][constraint]) for constraint in x_values]
                fig.add_trace(go.Scatter(x=x_values, y=smooth(y_values), mode='lines+markers', name=f"{model_name}"))
            fig.write_html(os.path.join(visualization_dir, f"model_wise_comparison_{ebf_key}.html"))
            print("Saved to", os.path.join(visualization_dir, f"model_wise_comparison_{ebf_key}.html"))
            # create matplotlib plot
            # we need a new model_ebf_dict structure -- such that model_ebf_dict[model][ebf_key] = original model_ebf_dict[model]
            model_ebf_dict = {model_name_visualization_name_mapping(model_name): {ebf_key: smooth(
                [np.mean(model_wise_aggregated[model_name][constraint]) for constraint in constraints])} for model_name
                in models}
            save_path = os.path.join(visualization_dir, f"model_wise_comparison_{ebf_key}.pdf")
            std_dict = None
            if std_key in prompt_dict_keys:
                std_dict = {model_name_visualization_name_mapping(model_name): {ebf_key: smooth(
                    [np.mean(model_wise_aggregated_std[model_name][constraint]) for constraint in constraints])} for
                    model_name
                    in models}
            matplotlib_plot(constraints, model_ebf_dict, save_path, tag=ebf_key,
                            y_label=ebf_name_visualization_name_mapping(ebf_key), std_dict=std_dict, fontsize=50,
                            linewidth=5)
            print("Saved to", os.path.join(visualization_dir, f"model_wise_comparison_{ebf_key}.pdf"))

    if args.piecewise_ebf:
        stat_piecewise(args, prompts, prompt_wise_dict, models, constraints, visualization_dir, smooth)
        stat_piecewise(args, prompts, prompt_wise_dict, models, constraints, visualization_dir, smooth, fixed_start=True)

    # write out all prompts to csv, using dictwriter
    with open(os.path.join(visualization_dir, "all_prompts.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "model"])
        writer.writeheader()
        for idx, (prompt, model) in enumerate(all_prompts):
            writer.writerow({"id": f"prompt-{idx}", "prompt": prompt, "model": model})


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.output_root_dir, args.ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    # redirect stdout to a file
    sys.stdout = open(os.path.join(str(output_dir),
                                   f"stat_ctrlgen_multi_constraints_offset_{args.offset}"
                                   f"_high_ent_threshold_{args.high_entropy_threshold}"
                                   f"{'' if args.maxlen is None else '_maxlen_' + str(args.maxlen)}.log"),
                      "w")
    constraints = [int(x) for x in args.constraints.split(",")]
    models, prompt_wise_dict = stat_dict_generation(args)

    # plot the prompt-wise stats using plotly
    # only do it with offset = 0
    if args.offset == 0:
        visualization_dir = os.path.join(str(output_dir),
                                         f"visualization_promptwise_{args.offset}{'' if args.maxlen is None else f'_maxlen_{args.maxlen}'}{'_smoothing_' + str(args.smoothing_factor)}")
        stat_visualization(visualization_dir, args, models, prompt_wise_dict, constraints)
