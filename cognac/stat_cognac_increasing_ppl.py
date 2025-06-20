import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import plotly.graph_objects as go
import csv

from uncertainty_quantification.uncertainty_computation import \
    from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies, \
    compute_ebf_from_length_wise_and_sample_wise_entropies

if __name__ == '__main__':
    # models = ["Llama-2-13b-hf", "Llama-2-70b-hf", "Llama-2-70b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-7b-chat-hf", "Yi-34B-Chat", "Mixtral-8x7B-Instruct-v0.1"]
    parser = argparse.ArgumentParser(description='CognacStatParser.')
    parser.add_argument('--ckpt_dir', type=str, default="output_manual_check_cognac_app_ctrlgen_multi_constraints",
                        help='model name')
    parser.add_argument("--high_entropy_threshold", type=float, default=0.7, help="high entropy threshold")
    parser.add_argument("--output_root_dir", type=str, default="stat_cognac_app_ctrlgen_multi_constraints", )
    parser.add_argument("--offset", type=int, default=0, help="offset for the prompt")
    parser.add_argument("--maxlen", type=int, default=None, help="max length for the prompt")
    args = parser.parse_args()
    output_dir = os.path.join(args.output_root_dir, args.ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    # redirect stdout to a file
    sys.stdout = open(os.path.join(str(output_dir),
                                   f"stat_cognac_app_ctrlgen_multi_constraints_offset_{args.offset}"
                                   f"_high_ent_threshold_{args.high_entropy_threshold}"
                                   f"{'' if args.maxlen is None else '_maxlen_' + str(args.maxlen)}.log"),
                      "w")
    models = ["Llama-2-13b-hf", "Llama-2-70b-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf", ]
    ckpt_dir = args.ckpt_dir
    constraints = [1, 2, 3, 4, 5]
    ebf_types = None
    prompt_wise_dict = dict()
    for model_name in models:
        ckpt_name = os.path.join(ckpt_dir,
                                 f"cognac_app_ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt")
        if not os.path.exists(ckpt_name):
            print(f"File not found: {ckpt_name}")
            continue
        ckpt = torch.load(ckpt_name)
        # verify the dump file a bit
        _constraint_keys = list(ckpt.keys())
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
                    prompt_wise_dict[idx] = {"prompts": []}
                if constraint not in prompt_wise_dict[idx]:
                    prompt_wise_dict[idx][constraint] = dict()
                if model_name not in prompt_wise_dict[idx][constraint]:
                    prompt_wise_dict[idx][constraint][model_name] = dict()
                prompt_wise_dict[idx]["prompts"].append((prompt, model_name))
                num_unique_outputs = len(set(all_output_texts))
                if args.offset >= max([len(x) for x in entropies]):
                    continue
                length_wise_entropies, sample_wise_seq_mean_entropies = from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies(
                    entropies, maxlen=args.maxlen)
                ebf = compute_ebf_from_length_wise_and_sample_wise_entropies(length_wise_entropies,
                                                                             sample_wise_seq_mean_entropies,
                                                                             offset=args.offset)
                for ebf_key in ebf.keys():
                    diagnostics[constraint]['prompt_wise_stats'][f'ebf_{ebf_key}'].append(ebf[ebf_key])
                    prompt_wise_dict[idx][constraint][model_name][f'ebf_{ebf_key}'] = ebf[ebf_key]

                if ebf_types is None:
                    ebf_types = [f'ebf_{ebf_key}' for ebf_key in ebf.keys()]
                # start counting too short outputs
                too_short_output_count = 0
                for token_text_i, token_text in enumerate(token_texts):
                    bool_too_short_flag = False
                    if len(token_text) <= 2 or len(token_text) <= args.offset:
                        # mostly just ['</s>']
                        # too_short_output_count += 1
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

    # plot the prompt-wise stats using plotly
    # only do it with offset = 0
    if args.offset == 0:
        visualization_dir = os.path.join(str(output_dir), f"visualization_promptwise_{args.offset}{'' if args.maxlen is None else f'_maxlen_{args.maxlen}'}")
        os.makedirs(visualization_dir, exist_ok=True)
        prompts = list(prompt_wise_dict.keys())
        prompts.sort()
        fig = go.Figure()
        all_prompts = []
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
                y_values = [prompt_wise_dict[prompt_i][constraint][model_name]['ebf_mc_ppl'] for constraint in x_values]
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=f"{model_name}_{prompt_i}"))
        fig.write_html(os.path.join(visualization_dir, f"prompt_wise_aggregation.html"))
        print("Saved to", os.path.join(visualization_dir, f"prompt_wise_aggregation.html"))
        # create prompt-aggregated plots
        # clean the figure
        fig = go.Figure()
        model_wise_aggregated = dict()
        for model_name in models:
            model_wise_aggregated[model_name] = {x: [] for x in constraints}
        for prompt_i in prompts:
            for model_name in models:
                for constraint in constraints:
                    if model_name not in prompt_wise_dict[prompt_i][constraint]:
                        raise ValueError(
                            f"Model {model_name} not found in prompt {prompt_i}, constraints {constraint}, current available models: {prompt_wise_dict[prompt_i][constraint].keys()}")
                    model_wise_aggregated[model_name][constraint].append(
                        prompt_wise_dict[prompt_i][constraint][model_name]['ebf_mc_ppl'])
        for model_name in models:
            x_values = constraints
            y_values = [np.mean(model_wise_aggregated[model_name][constraint]) for constraint in x_values]
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=f"{model_name}"))
        fig.write_html(os.path.join(visualization_dir, f"model_wise_comparison.html"))
        print("Saved to", os.path.join(visualization_dir, f"model_wise_comparison.html"))
        # write out all prompts to csv, using dictwriter
        with open(os.path.join(visualization_dir, "all_prompts.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt", "model"])
            writer.writeheader()
            for idx, (prompt, model) in enumerate(all_prompts):
                writer.writerow({"id": f"prompt-{idx}", "prompt": prompt, "model": model})
