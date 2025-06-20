# shared by storytelling, language modeling, etc.
import argparse
import gc
import glob
import json
import os
import random
import traceback

import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from uncertainty_quantification.common_utils import get_update_full_spectrum_file_pattern
from uncertainty_quantification.consts import ALL_MODELS
from uncertainty_quantification.loglik_computation import (compute_loglik, get_tokenwise_logprob_from_vllm_outputs,
                                                           get_tokenwise_entropy_from_vllm_outputs)
from uncertainty_quantification.visualization_utils import matplotlib_plot, model_name_visualization_name_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--source_dir', type=str, help='source dir', default="response_storywriting")
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=1024)
    parser.add_argument("--min_p", type=float, default=0, help="minimum p value for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="minimum p value for sampling")
    parser.add_argument("--force_recompute", action="store_true", help="force recompute")
    parser.add_argument("--constraints", type=str, default="0,1,2,3,4,5", help="constraint levels")
    parser.add_argument("--additional_file_search_pattern", type=str, default="", help="additional file search pattern")
    parser.add_argument("--min_k", type=int, default=20, help="min-K%, used to detect data in training set")
    args = parser.parse_args()
    random.seed(42)
    root_dir = args.source_dir
    print("Now processing source dir: {}".format(os.path.abspath(root_dir)))
    constraint_levels = [int(x) for x in args.constraints.split(",")]
    subdirs = [(x, "application_ctrlgen_multi_constraints_{}".format(x)) for x in constraint_levels]
    example_metric_flag = True
    sample_indices = None
    sample_output_indices_dict = dict()
    visualization_dir = f"output_loglik_{root_dir}_max_tokens_{args.max_tokens}_min_p_{args.min_p}_top_p_{args.top_p}"
    os.makedirs(visualization_dir, exist_ok=True)
    final_results_dict = dict()
    model_name = os.path.basename(args.model)
    if model_name != "ALL":
        ckpt_name = os.path.join(visualization_dir,
                                 f"loglik_analysis_{model_name}.pt")
        model_name_to_path = {os.path.basename(x): x for x in ALL_MODELS}
        if os.path.exists(ckpt_name) and not args.force_recompute:
            try:
                # final_results_dict, metric_keys, ebf_types = torch.load(ckpt_name)
                ckpt_results = torch.load(ckpt_name)
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
            file_pattern = get_update_full_spectrum_file_pattern(source_dir, model_name, args)
            files = glob.glob(file_pattern)
            # originally, we want to process all files -- maybe they differ in the number of output samples, the number of output logits, etc.
            # however, later we realize we don't need to process all files, we can just process one file, and then we can get the entropy profile for all files
            assert len(
                files) == 1, "We should have only one file found, but the program found: {}\nFile Pattern: {}".format(
                files, file_pattern)
            filename = files[0]
            model = os.path.basename(filename).split("_response")[0]
            tokenizer = AutoTokenizer.from_pretrained(model_name_to_path[model])
            if constraint_level in final_results_dict and model in set(
                    final_results_dict[constraint_level].keys()) and not args.force_recompute:
                print("Skip constraint level {} for model {}".format(constraint_level, model))
                continue
            else:
                if constraint_level not in final_results_dict:
                    final_results_dict[constraint_level] = dict()
                else:
                    final_results_dict[constraint_level][model] = dict()
            print("Find {} files to compute loglik profile, they are: \n{}".format(len(files),
                                                                                   json.dumps(files, indent=4)))
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
                print("Loaded from metadata file")
                assert os.path.exists(filename), "Please run main.py first!"
                database = torch.load(filename)
                loglik_profile = {"prompt": [], "output": [], "prompt_per_token_logprob": [],
                                  "output_per_token_logprob": [], "output_per_token_logprob_truncated": [],
                                  "entropy": [],
                                  "metadata": metadata}
                for idx in tqdm(range(len(database)), desc=f"Processing database",
                                leave=False):
                    data = database[idx]
                    prompt = prompts[idx]
                    all_outputs = data.outputs
                    all_outputs_texts = [x.text.strip() for x in all_outputs]
                    if data.prompt_logprobs is None:
                        prompt_loglik = None
                        prompt_per_token_loglik = None
                    else:
                        prompt_loglik = compute_loglik(data.prompt_token_ids, data.prompt_logprobs)
                        prompt_per_token_loglik = get_tokenwise_logprob_from_vllm_outputs(data.prompt_token_ids,
                                                                                          data.prompt_logprobs)
                    prompt_loglik_profile = [prompt_loglik, len(data.prompt_token_ids), prompt]
                    output_loglik_profiles = [[x.cumulative_logprob, len(x.token_ids), x.text, idx] for x in
                                              all_outputs]
                    output_per_token_loglik_profiles = [get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs)
                                                        for x in all_outputs]
                    output_per_token_loglik_profiles_truncated = [
                        get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs, top_p=args.top_p) for x in
                        all_outputs]
                    entropies = get_tokenwise_entropy_from_vllm_outputs(all_outputs, args.top_p, top_p_mode=True)
                    entropies = [x[0] for x in entropies]
                    loglik_profile["prompt"].append(prompt_loglik_profile)
                    loglik_profile["output"].extend(output_loglik_profiles)
                    loglik_profile["prompt_per_token_logprob"].append(prompt_per_token_loglik)
                    loglik_profile["output_per_token_logprob"].append(output_per_token_loglik_profiles)
                    loglik_profile['output_per_token_logprob_truncated'].append(
                        output_per_token_loglik_profiles_truncated)
                    loglik_profile['entropy'].append(entropies)
                assert model not in final_results_dict[
                    constraint_level], "Model already exists: {}, Please check whether the full-spectrum pattern matches more than one ckpt".format(
                    model)
                final_results_dict[constraint_level][model] = loglik_profile
                del database
                gc.collect()

            except Exception as e:
                print(e)
                # print stack trace
                traceback.print_exc()
                print("Failed to process file: {}".format(filename))
            torch.save(final_results_dict, ckpt_name)
        print("All loglik profile processed!")
    else:
        ckpt_names = glob.glob(os.path.join(visualization_dir, "loglik_analysis*.pt"))
        # plot loglik profile change w.r.t different constraint levels
        # we need to plot both plotly and matplotlib
        # we need to plot both prompt and output
        # step-1: plot prompt loglik (average) w.r.t different constraint levels
        loglik_records = dict()
        for ckpt_name in ckpt_names:
            final_results_dict = torch.load(ckpt_name)
            models = list(final_results_dict[constraint_levels[0]].keys())
            assert len(
                models) == 1, "We should have only one model, but the program found: {}\nFile Pattern: {}".format(
                models, ckpt_name)
            for _model in models:
                _renamed_model = model_name_visualization_name_mapping(_model)
                assert _renamed_model not in loglik_records, "Model already exists: {}".format(_renamed_model)
                if "storywriting" not in args.source_dir:
                    loglik_records[_renamed_model] = {"prompt": [], "output": [], "prompt_min_k": []}
                    for constraint_level in constraint_levels:
                        loglik_records[_renamed_model]["prompt"].append(
                            np.mean([x[0] / x[1] for x in final_results_dict[constraint_level][_model]["prompt"]]))
                        loglik_records[_renamed_model]["output"].append(
                            np.mean([x[0] / x[1] for x in final_results_dict[constraint_level][_model]["output"]]))
                        buf = []
                        for prompt_logprobs in final_results_dict[constraint_level][_model]["prompt_per_token_logprob"]:
                            # sort the output loglik profiles by the loglik value, replace any -inf with -12
                            if len(prompt_logprobs) == 0:
                                buf.append(0)
                            else:
                                _prompt_logprobs = [x for x in prompt_logprobs if x != float("-inf")]
                                _prompt_logprobs.sort()
                                # only take min_k% of the loglik values
                                min_k = int(len(_prompt_logprobs) * args.min_k / 100)
                                buf.append(np.mean(_prompt_logprobs[:min_k]))
                        loglik_records[_renamed_model]["prompt_min_k"].append(np.mean(buf))

                else:
                    loglik_records[_renamed_model] = {"prompt": dict(), "output": dict(), "prompt_min_k": dict()}
                    for constraint_level in constraint_levels:
                        metadata = final_results_dict[constraint_level][_model]["metadata"]
                        all_original_prompts = metadata[-2]
                        source_model_families = set([x[1] for x in all_original_prompts])
                        loglik_records[_renamed_model]["prompt"][constraint_level] = dict()
                        loglik_records[_renamed_model]["output"][constraint_level] = dict()
                        loglik_records[_renamed_model]["prompt_min_k"][constraint_level] = dict()
                        for idx, source_model_family in enumerate(source_model_families):
                            source_model_family_prompt_indices = [idx for idx, x in enumerate(all_original_prompts) if
                                                                  x[1] == source_model_family]
                            loglik_records[_renamed_model]["prompt"][constraint_level][source_model_family] = np.mean([
                                final_results_dict[constraint_level][_model]["prompt"][prompt_i][0] /
                                final_results_dict[constraint_level][_model]["prompt"][prompt_i][1]
                                for prompt_i in source_model_family_prompt_indices])
                            source_model_family_prompt_indices = set(source_model_family_prompt_indices)
                            source_model_family_output_indices = [idx for idx, x in enumerate(
                                final_results_dict[constraint_level][_model]["output"]) if x[3] in source_model_family_prompt_indices]
                            loglik_records[_renamed_model]["output"][constraint_level][source_model_family] = np.mean([
                                final_results_dict[constraint_level][_model]["output"][output_i][0] /
                                final_results_dict[constraint_level][_model]["output"][output_i][1]
                                for output_i in source_model_family_output_indices])
                            buf = []
                            for prompt_logprobs in final_results_dict[constraint_level][_model]["prompt_per_token_logprob"]:
                                # sort the output loglik profiles by the loglik value, replace any -inf with -12
                                if len(prompt_logprobs) == 0:
                                    buf.append(0)
                                else:
                                    _prompt_logprobs = [x for x in prompt_logprobs if x != float("-inf")]
                                    _prompt_logprobs.sort()
                                    # only take min_k% of the loglik values
                                    min_k = int(len(_prompt_logprobs) * args.min_k / 100)
                                    buf.append(np.mean(_prompt_logprobs[:min_k]))
                            loglik_records[_renamed_model]["prompt_min_k"][constraint_level][
                                source_model_family] = np.mean(buf)

        x_values = constraint_levels
        fig = go.Figure()
        if "storywriting" not in args.source_dir:
            for model in loglik_records:
                y_values = loglik_records[model]["prompt"]
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=model))
            fig.write_html(os.path.join(visualization_dir, "prompt_loglik_profile.html"))
            # plot matplotlib
            save_path = os.path.join(visualization_dir, "prompt_loglik_profile.pdf")
            matplotlib_plot(constraint_levels, loglik_records, save_path, tag="prompt", y_label="Loglik (prompt)",
                            fontsize=50, linewidth=5)
            # step-2: plot output loglik (average) w.r.t different constraint levels
            fig = go.Figure()
            for model in loglik_records:
                y_values = loglik_records[model]["output"]
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=model))
            fig.write_html(os.path.join(visualization_dir, "output_loglik_profile.html"))
            # plot matplotlib
            save_path = os.path.join(visualization_dir, "output_loglik_profile.pdf")
            matplotlib_plot(constraint_levels, loglik_records, save_path, tag="output", y_label="Loglik (output)",
                            fontsize=50, linewidth=5)
            # step-3: plot prompt loglik (min-K%) w.r.t different constraint levels
            fig = go.Figure()
            for model in loglik_records:
                y_values = loglik_records[model]["prompt_min_k"]
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=model))
            fig.write_html(os.path.join(visualization_dir, "prompt_min_k_loglik_profile.html"))
            # plot matplotlib
            save_path = os.path.join(visualization_dir, "prompt_min_k_loglik_profile.pdf")
            matplotlib_plot(constraint_levels, loglik_records, save_path, tag="prompt_min_k", y_label="Loglik (min_k)",
                            fontsize=50, linewidth=5)
        else:
            for model in loglik_records:
                for source_model_family in loglik_records[model]["prompt"][constraint_levels[0]]:
                    y_values = [loglik_records[model]["prompt"][constraint_level][source_model_family] for
                                constraint_level in constraint_levels]
                    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers',
                                             name=f"{source_model_family}->{model}"))
            fig.write_html(os.path.join(visualization_dir, "prompt_wise_aggregation_source_model_family.html"))
            # plot matplotlib, per source model family, create separate pdf plot
            for source_model_family in loglik_records[model]["prompt"][constraint_levels[0]]:
                save_path = os.path.join(visualization_dir, f"prompt_wise_aggregation_{source_model_family}.pdf")
                # obtain the loglik records for this source model family
                loglik_records_per_source_model_family = dict()
                for model in loglik_records:
                    loglik_records_per_source_model_family[model] = {
                        "prompt": [loglik_records[model]["prompt"][constraint_level][source_model_family] for
                                   constraint_level in constraint_levels]}
                matplotlib_plot(constraint_levels, loglik_records_per_source_model_family, save_path, tag="prompt",
                                y_label="Loglik (prompt)", fontsize=50, linewidth=5)
            # step-2: plot output loglik (average) w.r.t different constraint levels
            fig = go.Figure()
            for model in loglik_records:
                for source_model_family in loglik_records[model]["output"][constraint_levels[0]]:
                    y_values = [loglik_records[model]["output"][constraint_level][source_model_family] for
                                constraint_level in constraint_levels]
                    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers',
                                             name=f"{source_model_family}->{model}"))
            fig.write_html(os.path.join(visualization_dir, "output_wise_aggregation_source_model_family.html"))
            # plot matplotlib, per source model family, create separate pdf plot
            for source_model_family in loglik_records[model]["output"][constraint_levels[0]]:
                save_path = os.path.join(visualization_dir, f"output_wise_aggregation_{source_model_family}.pdf")
                # obtain the loglik records for this source model family
                loglik_records_per_source_model_family = dict()
                for model in loglik_records:
                    loglik_records_per_source_model_family[model] = {
                        "output": [loglik_records[model]["output"][constraint_level][source_model_family] for
                                   constraint_level in constraint_levels]}
                matplotlib_plot(constraint_levels, loglik_records_per_source_model_family, save_path, tag="output",
                                y_label="Loglik (output)", fontsize=50, linewidth=5)
            # step-3: plot prompt loglik (min-K%) w.r.t different constraint levels
            fig = go.Figure()
            for model in loglik_records:
                for source_model_family in loglik_records[model]["prompt_min_k"][constraint_levels[0]]:
                    y_values = [loglik_records[model]["prompt_min_k"][constraint_level][source_model_family] for
                                constraint_level in constraint_levels]
                    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers',
                                             name=f"{source_model_family}->{model}"))
            fig.write_html(os.path.join(visualization_dir, "prompt_min_k_wise_aggregation_source_model_family.html"))
            # plot matplotlib, per source model family, create separate pdf plot
            for source_model_family in loglik_records[model]["prompt_min_k"][constraint_levels[0]]:
                save_path = os.path.join(visualization_dir, f"prompt_min_k_wise_aggregation_{source_model_family}.pdf")
                # obtain the loglik records for this source model family
                loglik_records_per_source_model_family = dict()
                for model in loglik_records:
                    loglik_records_per_source_model_family[model] = {
                        "prompt_min_k": [loglik_records[model]["prompt_min_k"][constraint_level][source_model_family]
                                         for constraint_level in constraint_levels]}
                matplotlib_plot(constraint_levels, loglik_records_per_source_model_family, save_path,
                                tag="prompt_min_k", y_label="Loglik (min_k)", fontsize=50, linewidth=5)
