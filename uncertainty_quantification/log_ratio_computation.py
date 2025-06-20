# shared by storytelling, language modeling, etc.
import argparse
import copy
import gc
import numpy as np
import json
import glob
import os
import random
import traceback
from collections import defaultdict
from csv import DictWriter
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from uncertainty_quantification.consts import ALL_MODELS
from uncertainty_quantification.loglik_computation import get_tokenwise_entropy_from_vllm_outputs, compute_loglik

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--source_dir', type=str, help='source dir', default="response_storywriting")
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=1024)
    parser.add_argument("--min_p", type=float, default=0, help="minimum p value for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="minimum p value for sampling")
    parser.add_argument("--enforce_min_p", action="store_true", help="enforce min p")
    parser.add_argument("--force_recompute", action="store_true", help="force recompute")
    parser.add_argument("--constraints", type=str, default="0,1,2,3,4,5", help="constraint levels")
    args = parser.parse_args()
    random.seed(42)
    root_dir = args.source_dir
    print("Now processing source dir: {}".format(os.path.abspath(root_dir)))
    constraint_levels = [int(x) for x in args.constraints.split(",")]
    subdirs = [(x, "application_ctrlgen_multi_constraints_{}".format(x)) for x in constraint_levels]
    ps = [args.min_p] if args.min_p > 0 else [args.top_p]
    if args.enforce_min_p:
        print("Enforce min p")
        ps = [0.1]
    example_metric_flag = True
    sample_indices = None
    top_p_mode = False if args.min_p > 0 or args.enforce_min_p else True
    sample_output_indices_dict = dict()
    visualization_dir = f"output_logratio_{root_dir}_app_ctrlgen_multi_constraints_max_tokens_{args.max_tokens}_min_p_{args.min_p}_top_p_{args.top_p}{'' if not args.enforce_min_p else '_enforce_min_p_0.1'}"
    os.makedirs(visualization_dir, exist_ok=True)
    if args.model != "ALL":
        final_results_dict = dict()
        prompt_families = set()
        model_set = set()
        model_name = os.path.basename(args.model)
        ckpt_name = os.path.join(visualization_dir,
                                 f"ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt")
        model_name_to_path = {os.path.basename(x): x for x in ALL_MODELS}
        if os.path.exists(ckpt_name) and not args.force_recompute:
            try:
                ckpt_results, prompt_families, model_set = torch.load(ckpt_name)
                print("Loaded from checkpoint at: {}".format(ckpt_name))
                if isinstance(ckpt_results, list) and len(ckpt_results) == 3:
                    final_results_dict, loaded_metric_keys, loaded_ebf_types = ckpt_results
                else:
                    final_results_dict = ckpt_results
                loaded_constraint_levels = list(final_results_dict.keys())
            except Exception as e:
                print(e)
                print("Failed to load from checkpoint: {}".format(ckpt_name))
                # print stack trace
                traceback.print_exc()
                exit()
        for constraint_level, subdir in subdirs:
            source_dir = os.path.join(root_dir, subdir)
            if args.min_p > 0:
                # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
                files = glob.glob(os.path.join(source_dir,
                                               f"{model_name}_response*max_tokens_{args.max_tokens}*min_p_{args.min_p}_*.pt.update_full_spectrum"))
            else:
                # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
                files = glob.glob(os.path.join(source_dir,
                                               f"{model_name}_response*max_tokens_{args.max_tokens}*top_p_{args.top_p}_*.pt.update_full_spectrum"))
            final_results_dict[constraint_level] = dict()
            print("Find {} files to compute entropy profile, they are: \n{}".format(len(files), json.dumps(files, indent=4)))
            # originally, we want to process all files -- maybe they differ in the number of output samples, the number of output logits, etc.
            # however, later we realize we don't need to process all files, we can just process one file, and then we can get the entropy profile for all files
            if len(files) > 1:
                # check whether the two files only differ in the number of log_probs
                # e.g., ...._log_probs_{x}_... and ...._log_probs_{y}_...
                # if so, we can just process one file
                file_basenames = [os.path.basename(x) for x in files]
                # extract x, y, ... from ..._log_probs_{x}_... and ..._log_probs_{y}_...
                log_probs_nums = [int(x.split("_log_probs_")[1].split("_")[0]) for x in file_basenames]
                # check whether files only differ in the number of log_probs
                reference = files[0]
                log_probs_reference = log_probs_nums[0]
                duplicate_flag = True
                for idx in range(1, len(files)):
                    log_probs_num = log_probs_nums[idx]
                    if reference.replace(f"_log_probs_{log_probs_reference}_", f"_log_probs_{log_probs_num}_") != files[idx]:
                        duplicate_flag = False
                        break
                if duplicate_flag:
                    # choose the one with the largest number of log_probs
                    log_probs_num = max(log_probs_nums)
                    files = [x for x in files if f"_log_probs_{log_probs_num}_" in x]

            assert len(files) == 1, "We should have only one file found, but the program found: {}".format(files)

            filename = files[0]
            model = os.path.basename(filename).split("_response")[0]
            model_set.add(model)
            tokenizer = AutoTokenizer.from_pretrained(model_name_to_path[model])
            try:
                metadata_filename = filename.replace(".pt.update_full_spectrum", ".metadata")
                assert os.path.exists(metadata_filename), "Please run main.py first!"
                # for cognac experiments, the last element can be either a tuple or just two elements (hierarchy, updated_args)
                # storytelling, word-level language modeling, etc
                # prompts, roles, all_task_prompts, all_original_prompts, args = torch.load(metadata_filename)
                # cognac
                # prompts, answers, sources, remained = torch.load(metadata_filename)
                # if isinstance(remained, tuple):
                #     hierarchy, updated_args = remained
                # else:
                #     hierarchy = remained
                #     updated_args = get_default_args()
                # mmlu
                # 1. prompts, roles, answers, options, all_constrained_prompts, args = torch.load(metadata_filename)
                # 2. (early version) prompts, roles, answers, all_constrained_prompts, args = torch.load(metadata_filename)
                metadata = torch.load(metadata_filename)
                # usually, the first element is prompts, the last element is args
                prompts = metadata[0]
                roles, answers, options = metadata[1: 4]
                print("Loaded from metadata file")
                assert os.path.exists(filename), "Please run main.py first!"
                database = torch.load(filename)
                loglik_dict = dict()
                logratio_dict = dict()
                instance_dict = dict()
                stat_shared_prefix_length = dict()
                for idx in tqdm(range(len(database)), desc=f"Processing database",
                                leave=False):
                    data = database[idx]
                    prompt = prompts[idx]
                    role = roles[idx]
                    answer = answers[idx]
                    option = options[idx]
                    all_outputs = data.outputs
                    all_outputs_texts = [x.text.strip() for x in all_outputs]
                    prompt_tokens = data.prompt_token_ids
                    prompt_logprobs = data.prompt_logprobs
                    # note, here we should compute prompt loglik, this part should not be impacted by decoding method, as we are working with "prompt"
                    # [2024/09/01]: unfortunately, sometimes loglik of some token would be -inf (happened in llama2-13b-series), we need to add a tolerance
                    prompt_loglik = compute_loglik(prompt_tokens, prompt_logprobs)
                    _prompt_family, _task, _sample_id, _option_j, _is_good = role.split("\t")
                    prompt_families.add(_prompt_family)
                    _is_good = eval(_is_good)
                    if _prompt_family not in loglik_dict:
                        loglik_dict[_prompt_family] = dict()
                        logratio_dict[_prompt_family] = dict()
                        instance_dict[_prompt_family] = dict()
                        stat_shared_prefix_length[_prompt_family] = []
                    sample_key = (_task, _sample_id)
                    if sample_key not in loglik_dict[_prompt_family]:
                        loglik_dict[_prompt_family][sample_key] = []
                        instance_dict[_prompt_family][sample_key] = []
                    loglik_dict[_prompt_family][sample_key].append({"prompt_loglik": prompt_loglik, "is_good": _is_good})
                    instance_dict[_prompt_family][sample_key].append(idx)
                    if len(instance_dict[_prompt_family][sample_key]) == 4:
                        # mmlu is 4-way classification
                        prompt_token_instance = [database[x].prompt_token_ids for x in instance_dict[_prompt_family][sample_key]]
                        prompt_logprobs_instance = [database[x].prompt_logprobs for x in instance_dict[_prompt_family][sample_key]]
                        # find longest common prefix in prompt_token_instance, each element is a list of token ids
                        shared_prefix = []
                        for i in range(len(prompt_token_instance[0])):
                            if len(set([x[i] for x in prompt_token_instance])) == 1:
                                shared_prefix.append(prompt_token_instance[0][i])
                            else:
                                break
                        stat_shared_prefix_length[_prompt_family].append(len(shared_prefix))
                        option_token_instance = [x[len(shared_prefix):] for x in prompt_token_instance]
                        option_logprobs_instance = [x[len(shared_prefix):] for x in prompt_logprobs_instance]
                        for option_i in range(len(option_token_instance)):
                            option_tokens = option_token_instance[option_i]
                            option_logprobs = option_logprobs_instance[option_i]
                            option_loglik = compute_loglik(option_tokens, option_logprobs)
                            loglik_dict[_prompt_family][sample_key][option_i]["option_loglik"] = option_loglik

                for _prompt_family in loglik_dict:
                    for sample_key in loglik_dict[_prompt_family]:
                        _logliks = [x["prompt_loglik"] for x in loglik_dict[_prompt_family][sample_key]]
                        _is_goods = [x["is_good"] for x in loglik_dict[_prompt_family][sample_key]]
                        _dist = torch.softmax(torch.tensor(_logliks), dim=0)
                        _good_parts = torch.tensor([x for x, y in zip(_dist, _is_goods) if y]).sum()
                        if sample_key not in logratio_dict[_prompt_family]:
                            logratio_dict[_prompt_family][sample_key] = dict()
                        logratio_dict[_prompt_family][sample_key]['log_good_part_complete_ratio'] = torch.log(_good_parts).item()
                        _logliks = [x["option_loglik"] for x in loglik_dict[_prompt_family][sample_key]]
                        _dist = torch.softmax(torch.tensor(_logliks), dim=0)
                        _good_parts = torch.tensor([x for x, y in zip(_dist, _is_goods) if y]).sum()
                        logratio_dict[_prompt_family][sample_key]['log_good_part_option_ratio'] = torch.log(_good_parts).item()
                        # correct output loglik
                        good_loglik = [x for x, y in zip(_logliks, _is_goods) if y]
                        assert len(good_loglik) > 0, "No good loglik found"
                        logratio_dict[_prompt_family][sample_key]['log_correct'] = torch.logsumexp(torch.tensor(good_loglik), dim=0).item()
                    print("Shared prefix length {}: {}".format(_prompt_family, np.mean(stat_shared_prefix_length[_prompt_family])))
                final_results_dict[constraint_level][model] = logratio_dict

            except Exception as e:
                print(e)
                # print stack trace
                traceback.print_exc()
                print("Failed to process file: {}".format(filename))
            torch.save((final_results_dict, prompt_families, model_set), ckpt_name)
    else:
        # plot the results
        # x-axis: constraint level
        # y-axis: logratio
        # for each model-prompt_family pair, we have a line
        ckpt_names = glob.glob(os.path.join(visualization_dir, "ctrlgen_multi_constraints_investigation_increasing_PPL_*.pt"))
        prompt_families = set()
        model_set = set()
        final_results_dict = dict()

        for ckpt_name in ckpt_names:
            ckpt_results, _prompt_families, _model_set = torch.load(ckpt_name)
            _final_results_dict = ckpt_results
            prompt_families |= _prompt_families
            model_set |= _model_set
            for constraint_level in _final_results_dict:
                if constraint_level not in final_results_dict:
                    final_results_dict[constraint_level] = dict()
                for model in _final_results_dict[constraint_level]:
                    assert model not in final_results_dict[constraint_level], "Model already exists"
                    final_results_dict[constraint_level][model] = copy.deepcopy(_final_results_dict[constraint_level][model])
        x_vals = constraint_levels
        for attribute, attribute_alias in zip(["log_good_part_complete_ratio", "log_good_part_option_ratio", "log_correct"],
                                                ["logratio_complete", "logratio_option", "log_correct"]):
            y_label = " ".join([x.capitalize() for x in attribute_alias.split("_")])
            fig, ax1 = plt.subplots(figsize=(20, 15))
            fontsize = 50
            linewidth = 5
            # generate #(models) * #(prompt_families) colors
            colors = plt.cm.get_cmap('tab20', len(model_set) * len(prompt_families))
            counter = 0
            for model in model_set:
                for prompt_family in prompt_families:
                    y_vals = []
                    for constraint_level in x_vals:
                        elems = [x[attribute] for x in final_results_dict[constraint_level][model][prompt_family].values()]
                        y_vals.append(np.mean(elems))
                    ax1.plot(x_vals, y_vals, label=f"{model}-{prompt_family}", linewidth=linewidth, color=colors(counter))
                    counter += 1
            ax1.set_xlabel('Constraint Level', fontsize=fontsize)
            ax1.set_ylabel(y_label, fontsize=fontsize)
            ax1.set_title('Log Ratio Comparison Across Constraint Levels', fontsize=fontsize)
            ax1.legend()
            plt.savefig(os.path.join(visualization_dir, f"{attribute_alias}_comparison.png"))
            plt.clf()
            # using plotly to generate interactive plots
            fig = go.Figure()
            for model in model_set:
                for prompt_family in prompt_families:
                    y_vals = []
                    for constraint_level in x_vals:
                        elems = [x[attribute] for x in final_results_dict[constraint_level][model][prompt_family].values()]
                        y_vals.append(np.mean(elems))
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f"{model}-{prompt_family}"))
            fig.update_layout(title='Log Ratio Comparison Across Constraint Levels', xaxis_title='Constraint Level', yaxis_title=y_label)
            # save the plot
            fig.write_html(os.path.join(visualization_dir, f"{attribute_alias}_comparison.html"))

