import argparse
from csv import DictWriter
import plotly.graph_objects as go
import traceback
import glob
import os
import random
from collections import defaultdict
from consts import ALL_MODELS

import torch
from cognac.cognac_utils import get_default_args

from uncertainty_quantification.loglik_computation import get_tokenwise_entropy_from_vllm_outputs
from transformers import AutoTokenizer

from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--source_dir', type=str, help='source dir', default="cognac_responses")
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_cognac_app_1000.pt",
                        help="task file")
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=512)
    parser.add_argument("--min_p", type=float, default=0, help="minimum p value for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="minimum p value for sampling")
    parser.add_argument("--enforce_min_p", action="store_true", help="enforce min p")
    args = parser.parse_args()
    random.seed(42)

    root_dir = args.source_dir
    # model = args.model
    constraint_levels = [1, 2, 3, 4, 5]
    subdirs = [(x, "application_ctrlgen_multi_constraints_{}".format(x)) for x in constraint_levels]
    tasks, task_to_ids = torch.load(args.task_selection_filename)
    task_starts_id = dict()
    cur_task_data_id = 0
    for task in tasks:
        if task not in task_starts_id:
            task_starts_id[task] = cur_task_data_id
            # prompting format is given by the cognac_origin task, and there is no need to *2 as we do not have cot for now
            cur_task_data_id += len(task_to_ids[task])
    # roles = ['standard', 'cot']
    roles = ['std']
    ps = [args.min_p] if args.min_p > 0 else [args.top_p]
    if args.enforce_min_p:
        print("Enforce min p")
        ps = [0.1]
    example_metric_flag = True
    sample_indices = None
    top_p_mode = False if args.min_p > 0 or args.enforce_min_p else True
    sample_output_indices_dict = dict()
    visualization_dir = f"output_manual_check_{root_dir}_app_ctrlgen_multi_constraints_max_tokens_{args.max_tokens}_min_p_{args.min_p}_top_p_{args.top_p}{'' if not args.enforce_min_p else '_enforce_min_p_0.1'}"
    os.makedirs(visualization_dir, exist_ok=True)
    final_results_dict = dict()
    model_name = os.path.basename(args.model)
    ckpt_name = os.path.join(visualization_dir, f"cognac_app_ctrlgen_multi_constraints_investigation_increasing_PPL_{model_name}.pt")
    model_name_to_path = {os.path.basename(x): x for x in ALL_MODELS}
    if os.path.exists(ckpt_name):
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
            files = glob.glob(os.path.join(source_dir,
                                           f"{model_name}_response*max_tokens_{args.max_tokens}*min_p_{args.min_p}_*.pt.update_full_spectrum"))
        else:
            # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
            files = glob.glob(os.path.join(source_dir,
                                           f"{model_name}_response*max_tokens_{args.max_tokens}*top_p_{args.top_p}_*.pt.update_full_spectrum"))
        models = set([os.path.basename(x).split("_response")[0] for x in files if "Yi-34B" not in x])
        if constraint_level in final_results_dict and models.issubset(set(final_results_dict[constraint_level].keys())):
            print("Skip constraint level: {}".format(constraint_level))
            continue
        final_results_dict[constraint_level] = dict()
        for filename in tqdm(files, desc="Processing files", leave=False):
            model = os.path.basename(filename).split("_response")[0]
            tokenizer = AutoTokenizer.from_pretrained(model_name_to_path[model])
            try:
                metadata_filename = filename.replace(".pt.update_full_spectrum", ".metadata")
                # standard_prompt = load_task_prompts(STANDARD_PROMPT_PATH)
                # cot_prompt = load_task_prompts(COT_PROMPT_PATH)
                assert os.path.exists(metadata_filename), "Please run main.py first!"
                # prompts, answers, sources, tasks = torch.load(metadata_filename)
                prompts, answers, sources, remained = torch.load(metadata_filename)
                if isinstance(remained, tuple):
                    hierarchy, updated_args = remained
                else:
                    hierarchy = remained
                    updated_args = get_default_args()
                print("Loaded from metadata file")
                assert os.path.exists(filename), "Please run main.py first!"
                database = torch.load(filename)
                # model_names.add(model)
                # min_p -> entropy list per output
                entropies_profile = defaultdict(list)
                for task_i, task in enumerate(tasks):
                    start_id = task_starts_id[task]
                    task_data_num = len(task_to_ids[task])
                    for idx in tqdm(range(start_id, start_id + task_data_num), desc=f"Processing task {task}",
                                    leave=False):
                        # for idx in range(start_id, start_id + task_data_num * 2):
                        original_instance_i = task_to_ids[task][(idx - start_id) % task_data_num]
                        data = database[idx]
                        all_outputs = data.outputs
                        all_outputs_texts = [x.text.strip() for x in all_outputs]
                        answer = answers[idx]
                        role = sources[idx]
                        prompt = prompts[idx]
                        for p in ps:
                            _entropies_profile = get_tokenwise_entropy_from_vllm_outputs(all_outputs, p=p, top_p_mode=top_p_mode)
                            entropies = [x[0] for x in _entropies_profile]
                            token_ids = [x[1] for x in _entropies_profile]
                            token_texts = [tokenizer.convert_ids_to_tokens(x) for x in token_ids]
                            entropies_profile[p].append([prompt, all_outputs_texts, token_texts, entropies])
                assert model not in final_results_dict[constraint_level], "Model already exists: {}, Please check whether the full-spectrum pattern matches more than one ckpt".format(model)
                final_results_dict[constraint_level][model] = entropies_profile
                sample_p = ps[0]
                entropy_profile = entropies_profile[sample_p]
                # first, write out prompt + all_output_texts to a csv file
                if args.min_p > 0 or args.enforce_min_p:
                    csv_filename = os.path.join(visualization_dir, f"{model}_min_p_{sample_p}_constraint_{constraint_level}_entropy_profile.csv")
                else:
                    csv_filename = os.path.join(visualization_dir, f"{model}_top_p_{sample_p}_constraint_{constraint_level}_entropy_profile.csv")
                with open(csv_filename, "w", newline="") as f:
                    writer = DictWriter(f, fieldnames=["id", "prompt", "output_text", "tokens"])
                    writer.writeheader()
                    prompt_i = 0
                    for prompt, all_output_texts, token_texts, entropies in entropy_profile:
                        output_j = 0
                        for output_text in all_output_texts:
                            writer.writerow({"prompt": prompt, "output_text": output_text, "id": f"prompt-{prompt_i}-output-{output_j}", "tokens": str(token_texts[output_j])})
                            output_j += 1
                        prompt_i += 1
                # random sample prompt-output-entropy profile to plot using plotly
                # x-axis: token_text
                # y-axis: entropy
                sample_size = 20
                if args.min_p > 0 or args.enforce_min_p:
                    visualization_collection_dir = os.path.join(visualization_dir, f"{model}_min_p_{sample_p}_constraint_{constraint_level}_entropy_profile")
                else:
                    visualization_collection_dir = os.path.join(visualization_dir, f"{model}_top_p_{sample_p}_constraint_{constraint_level}_entropy_profile")
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
                        fig.add_trace(go.Scatter(x=token_texts[sample_j], y=entropies[sample_j], mode='lines+markers', name=f"prompt-{sample_i}-output-{sample_j}"))
                        fig.update_layout(title=f"{model} Entropy Profile", xaxis_title="Token Text", yaxis_title="Entropy")
                        fig_filename = os.path.join(visualization_collection_dir, f"prompt-{sample_i}-output-{sample_j}.html")
                        fig.write_html(fig_filename)

            except Exception as e:
                print(e)
                # print stack trace
                traceback.print_exc()
                print("Failed to process file: {}".format(filename))
        torch.save(final_results_dict, ckpt_name)
    print("All models processed!")
