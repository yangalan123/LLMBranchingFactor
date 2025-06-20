# shared by storytelling, language modeling, etc.
import argparse
import gc
import glob
import json
import os
import random
import traceback
from collections import defaultdict
from csv import DictWriter

import plotly.graph_objects as go
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from uncertainty_quantification.consts import ALL_MODELS
from uncertainty_quantification.loglik_computation import get_tokenwise_entropy_from_vllm_outputs

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
    parser.add_argument("--additional_file_search_pattern", type=str, default="", help="additional file search pattern")
    parser.add_argument("--additional_output_dir_pattern", type=str, default="", help="additional output dir pattern")
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
    visualization_dir = (f"output_manual_check_{root_dir}_app_ctrlgen_multi_constraints"
                         f"_max_tokens_{args.max_tokens}_min_p_{args.min_p}_top_p_{args.top_p}"
                         f"{'' if not args.enforce_min_p else '_enforce_min_p_0.1'}"
                         f"{args.additional_output_dir_pattern}")
    os.makedirs(visualization_dir, exist_ok=True)
    final_results_dict = dict()
    model_name = os.path.basename(args.model)
    ckpt_name = os.path.join(visualization_dir,
                             f"ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt")
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
            traceback.print_exc()
            exit()
    for constraint_level, subdir in subdirs:
        source_dir = os.path.join(root_dir, subdir)
        if args.min_p > 0:
            # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
            file_pattern = os.path.join(source_dir,
                                        f"{model_name}_response*max_tokens_{args.max_tokens}*min_p_{args.min_p}_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
        else:
            # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
            file_pattern = os.path.join(source_dir,
                                        f"{model_name}_response*max_tokens_{args.max_tokens}*top_p_{args.top_p}_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
        files = glob.glob(file_pattern)
        final_results_dict[constraint_level] = dict()
        print(
            "Find {} files to compute entropy profile, they are: \n{}".format(len(files), json.dumps(files, indent=4)))
        # originally, we want to process all files -- maybe they differ in the number of output samples, the number of output logits, etc.
        # however, later we realize we don't need to process all files, we can just process one file, and then we can get the entropy profile for all files
        try:
            assert len(
                files) == 1, "We should have only one file found, but the program found: {}\nFile Pattern: {}".format(files,
                                                                                                                      file_pattern)
        except AssertionError as e:
            print(e)
            print("Please check the file pattern and whether this file exists!")
            continue
        filename = files[0]
        # for filename in tqdm(files, desc="Processing files", leave=False):
        model = os.path.basename(filename).split("_response")[0]
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
            print("Loaded from metadata file")
            assert os.path.exists(filename), "Please run main.py first!"
            database = torch.load(filename)
            # min_p -> entropy list per output
            entropies_profile = defaultdict(list)
            for idx in tqdm(range(len(database)), desc=f"Processing database",
                            leave=False):
                data = database[idx]
                prompt = prompts[idx]
                all_outputs = data.outputs
                all_outputs_texts = [x.text.strip() for x in all_outputs]
                for p in ps:
                    _entropies_profile = get_tokenwise_entropy_from_vllm_outputs(all_outputs, p=p,
                                                                                 top_p_mode=top_p_mode)
                    entropies = [x[0] for x in _entropies_profile]
                    token_ids = [x[1] for x in _entropies_profile]
                    token_texts = [tokenizer.convert_ids_to_tokens(x) for x in token_ids]
                    entropies_profile[p].append([prompt, all_outputs_texts, token_texts, entropies])
            assert model not in final_results_dict[
                constraint_level], "Model already exists: {}, Please check whether the full-spectrum pattern matches more than one ckpt".format(
                model)
            final_results_dict[constraint_level][model] = entropies_profile
            sample_p = ps[0]
            entropy_profile = entropies_profile[sample_p]
            # first, write out prompt + all_output_texts to a csv file
            if args.min_p > 0 or args.enforce_min_p:
                csv_filename = os.path.join(visualization_dir,
                                            f"{model}_min_p_{sample_p}_constraint_{constraint_level}_entropy_profile.csv")
            else:
                csv_filename = os.path.join(visualization_dir,
                                            f"{model}_top_p_{sample_p}_constraint_{constraint_level}_entropy_profile.csv")
            with open(csv_filename, "w", newline="") as f:
                writer = DictWriter(f, fieldnames=["id", "prompt", "output_text", "tokens"], escapechar='\\')
                writer.writeheader()
                prompt_i = 0
                for prompt, all_output_texts, token_texts, entropies in entropy_profile:
                    output_j = 0
                    for output_text in all_output_texts:
                        writer.writerow({"prompt": prompt, "output_text": output_text,
                                         "id": f"prompt-{prompt_i}-output-{output_j}",
                                         "tokens": str(token_texts[output_j])})
                        output_j += 1
                    prompt_i += 1
            # random sample prompt-output-entropy profile to plot using plotly
            # x-axis: token_text
            # y-axis: entropy
            sample_size = 20
            if args.min_p > 0 or args.enforce_min_p:
                visualization_collection_dir = os.path.join(visualization_dir,
                                                            f"{model}_min_p_{sample_p}_constraint_{constraint_level}_entropy_profile")
            else:
                visualization_collection_dir = os.path.join(visualization_dir,
                                                            f"{model}_top_p_{sample_p}_constraint_{constraint_level}_entropy_profile")
            os.makedirs(visualization_collection_dir, exist_ok=True)
            if sample_indices is None:
                sample_indices = random.sample(range(len(entropy_profile)), sample_size)
            for sample_i in sample_indices:
                prompt, all_output_texts, token_texts, entropies = entropy_profile[sample_i]
                # randomly sample three outputs
                if sample_i not in sample_output_indices_dict:
                    sample_output_indices_dict[sample_i] = random.sample(range(len(all_output_texts)), 3)
                sample_output_indices = sample_output_indices_dict[sample_i]
                for sample_j in sample_output_indices:
                    output_text = all_output_texts[sample_j]
                    entropy = entropies[sample_j]
                    fig = go.Figure()
                    # add line plot
                    fig.add_trace(go.Scatter(x=token_texts[sample_j], y=entropies[sample_j], mode='lines+markers',
                                             name=f"prompt-{sample_i}-output-{sample_j}"))
                    fig.update_layout(title=f"{model} Entropy Profile", xaxis_title="Token Text",
                                      yaxis_title="Entropy")
                    fig_filename = os.path.join(visualization_collection_dir,
                                                f"prompt-{sample_i}-output-{sample_j}.html")
                    fig.write_html(fig_filename)
            del database
            gc.collect()

        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Failed to process file: {}".format(filename))
        torch.save(final_results_dict, ckpt_name)
    print("All entropy profile processed!")
