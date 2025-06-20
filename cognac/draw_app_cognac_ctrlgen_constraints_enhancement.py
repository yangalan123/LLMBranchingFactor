import argparse
import plotly.graph_objects as go
import traceback
import glob
import numpy as np
import os

import torch
# from consts import task_prompt, roles
# from mmlu_prompt_utils import DATA_DIR, COT_PROMPT_PATH, STANDARD_PROMPT_PATH, load_task_prompts, get_test_prompts
from cognac.cognac_utils import get_default_args
from cognac.cognac_metrics import compute_prediction_metrics

from uncertainty_quantification.uncertainty_computation import compute_ebf_from_vllm_outputs
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=512)
    parser.add_argument("--min_p", type=float, default=0, help="minimum p value for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="minimum p value for sampling")
    parser.add_argument('--source_dir', type=str, help='source dir', default="cognac_responses")
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_cognac_app_1000.pt",
                        help="task file")
    parser.add_argument("--enforce_min_p", action="store_true", help="enforce min p")
    args = parser.parse_args()

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
    ps = [0.1]
    if args.enforce_min_p:
        print("Enforce min p")
    example_metric_flag = True
    metric_keys = None
    ebf_types = None
    visualization_dir = f"visualization_{root_dir}_app_ctrlgen_multi_constraints_max_tokens_{args.max_tokens}_min_p_{args.min_p}_top_p_{args.top_p}{'' if not args.enforce_min_p else '_enforce_min_p_0.1'}"
    os.makedirs(visualization_dir, exist_ok=True)
    # copy task selection file to visualization dir
    os.system(f"cp {args.task_selection_filename} {visualization_dir}")
    final_results_dict = dict()
    model_names = set()
    # ckpt_name = os.path.join(visualization_dir, "cognac_app_ctrlgen_multi_constraints_final_results_full.pt")
    ckpt_name = os.path.join(visualization_dir, "cognac_app_ctrlgen_multi_constraints_final_results.pt")
    top_p_mode = False
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
            model_names = [x for x in final_results_dict[loaded_constraint_levels[0]].keys() if "Yi-34B" not in x]
            metric_keys = final_results_dict[loaded_constraint_levels[0]][model_names[0]]["performance"][tasks[0]][roles[0]].keys()
            metric_keys = [x for x in metric_keys if "_std" not in x]
            ebf_types = list(final_results_dict[loaded_constraint_levels[0]][model_names[0]]["ebf"][tasks[0]][roles[0]][0][ps[0]].keys())
            model_names = set(model_names)
            print("model_names:", model_names)
            print("metric_keys: ", metric_keys)
            print("ebf_types: ", ebf_types)
        except Exception as e:
            print(e)
            print("Failed to load from checkpoint: {}".format(ckpt_name))
            # print stack trace
            traceback.print_exc()
            exit()


    for constraint_level, subdir in subdirs:
        source_dir = os.path.join(root_dir, subdir)
        if args.min_p > 0:
            files = glob.glob(os.path.join(source_dir, f"*max_tokens_{args.max_tokens}*min_p_{args.min_p}_*.pt.update_full_spectrum"))
        else:
            # debug (07/06/2024): if we do not insert an extra underscore after {top_p}, 0.95 will match 0.9
            files = glob.glob(os.path.join(source_dir, f"*max_tokens_{args.max_tokens}*top_p_{args.top_p}_*.pt.update_full_spectrum"))
            if not args.enforce_min_p:
                top_p_mode = True
                ps = [args.top_p]

        models = set([os.path.basename(x).split("_response")[0] for x in files if "Yi-34B" not in x])
        if constraint_level in final_results_dict and models.issubset(set(final_results_dict[constraint_level].keys())):
            print("Skip constraint level: {}".format(constraint_level))
            continue
        final_results_dict[constraint_level] = dict()
        for filename in tqdm(files, desc="Processing files", leave=False):
            model = os.path.basename(filename).split("_response")[0]
            try:
                # metadata_filename = filename.replace(".pt", ".metadata")
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

                print(f"Loaded from metadata file: {metadata_filename}")
                assert os.path.exists(filename), "Please run main.py first!"
                database = torch.load(filename)
                print("Loaded from database file: {}".format(filename))
                task_wise_performance = dict()
                task_wise_ebf = dict()
                model_names.add(model)
                for task_i, task in enumerate(tasks):
                    start_id = task_starts_id[task]
                    task_data_num = len(task_to_ids[task])
                    # task_wise_performance[task] = {'standard': dict(), 'cot': dict()}
                    # task_wise_ebf[task] = {'standard': [], 'cot': []}
                    # we actually use eval_version=0 prompt (prompt_id=0 in diverse_instruction.csv)
                    task_wise_performance[task] = {'std': dict()}
                    task_wise_ebf[task] = {'std': []}
                    for idx in tqdm(range(start_id, start_id + task_data_num), desc=f"Processing task {task}",
                                    leave=False):
                        original_instance_i = task_to_ids[task][(idx - start_id) % task_data_num]
                        data = database[idx]
                        all_outputs = data.outputs
                        all_outputs_texts = [x.text.strip() for x in all_outputs]
                        answer = answers[idx]
                        role = sources[idx]
                        estimated_ebf = compute_ebf_from_vllm_outputs(all_outputs, ps=ps, top_p_mode=top_p_mode)
                        if ebf_types is None:
                            ebf_types = list(estimated_ebf[ps[0]].keys())
                        ebf_add_flag = True
                        for k, v in estimated_ebf[ps[0]].items():
                            if v < 0:
                                ebf_add_flag = False
                                break
                        if ebf_add_flag:
                            # task_wise_ebf[task][role].append(estimated_ebf[ps[0]])
                            task_wise_ebf[task][role].append(estimated_ebf)
                        assert prompts[idx] == data.prompt, "Prompt not match!"
                        metrics = []
                        for _text in all_outputs_texts:
                            metric = compute_prediction_metrics({"datapoint": answer, "generated_text": _text}, hierarchy, "wordnet", multi_constraints_eval=updated_args.multi_constraints > 1 or "multi_constraints" in answer)['prediction_metric']
                            metrics.append(metric)
                            if metric_keys is None:
                                metric_keys = list(metric.keys())
                                if example_metric_flag:
                                    print("Example metric: {}".format(metric))
                                    print("Example path: {}".format(_text))
                                    print("Example answer: {}".format(answer))
                                    example_metric_flag = False
                        for metric_key in metric_keys:
                            perf = [x[metric_key] for x in metrics]
                            if metric_key not in task_wise_performance[task][role]:
                                task_wise_performance[task][role][metric_key] = []
                                task_wise_performance[task][role][metric_key + '_std'] = []
                            task_wise_performance[task][role][metric_key].append(
                                sum(perf) / len(perf))
                            task_wise_performance[task][role][metric_key + '_std'].append(
                                np.std(perf))
                assert model not in final_results_dict[constraint_level], "Model already exists: {}, Please check whether the full-spectrum pattern matches more than one ckpt".format(model)
                final_results_dict[constraint_level][model] = {"performance": task_wise_performance, "ebf": task_wise_ebf}
            except Exception as e:
                print(e)
                # print stack trace
                traceback.print_exc()
                print("Failed to process file: {}".format(filename))
        torch.save([final_results_dict, metric_keys, ebf_types], ckpt_name)
    print("All models processed!")

    # Plotting
    # 1) x-axis: constraint level, y-axis: performance
    # 2) x-axis: constraint level, y-axis: ebf
    # 3) x-axis: ebf, y-axis: performance
    # first plot
    for metric_key in metric_keys:
        fig = go.Figure()
        for model in model_names:
            for task_i, task in enumerate(tasks):
                for role_i, role in enumerate(roles):
                    accs = [np.mean(final_results_dict[x][model]['performance'][task][role][metric_key]) for x in constraint_levels]
                    stds = [np.mean(final_results_dict[x][model]['performance'][task][role][metric_key + "_std"]) for x in constraint_levels]
                    # add the acc line, with stds as the error bar
                    fig.add_trace(go.Scatter(x=constraint_levels, y=accs, mode='markers+lines', name=f'{task}-{role}-{metric_key} ({model})'))
                    # print(f'{task}-{role}-{metric_key} ({model}): {[dict(level=x, val=y, std=z) for x, y, z in zip(constraint_levels, accs, stds)]}')
                    for min_p_i, min_p in enumerate(ps):
                        for ebf_i, ebf_type in enumerate(ebf_types):
                            if ebf_type == "ht_ppl":
                                continue
                            # add the ebf line
                            ebf_mean = [np.mean([x[min_p][ebf_type] for x in final_results_dict[y][model]['ebf'][task][role]]) for y in constraint_levels]
                            fig.add_trace(go.Scatter(x=constraint_levels, y=ebf_mean, mode='lines', name=f'{task}-{role}-ebf ({min_p}, {ebf_type})-{model}'))
        fig.update_layout(title=f'{metric_key}-EBF correlation', xaxis_title='Constraint Level', yaxis_title="Value", legend_title="EBF")
        fig.write_html(os.path.join(visualization_dir, f'{metric_key}_comprehensive.html'))
        print(f"Saved {metric_key}_comprehensive.html")

    # second plot, this time only plots ebf (this figure is just for ease of demo)
    for model in model_names:
        fig = go.Figure()
        for task_i, task in enumerate(tasks):
            for role_i, role in enumerate(roles):
                for min_p_i, min_p in enumerate(ps):
                    for ebf_i, ebf_type in enumerate(ebf_types):
                        if ebf_type == "ht_ppl":
                            continue
                        # add the ebf line
                        ebf_mean = [np.mean([x[min_p][ebf_type] for x in final_results_dict[y][model]['ebf'][task][role]]) for y in constraint_levels]
                        fig.add_trace(go.Scatter(x=constraint_levels, y=ebf_mean, mode='lines', name=f'{task}-{role}-ebf ({min_p}, {ebf_type})-{model}'))
                        # print(
                        #     f'{task}-{role}-ebf ({min_p}, {ebf_type})-{model}: {[dict(level=x, val=y) for x, y in zip(constraint_levels, ebf_mean)]}')
        fig.update_layout(title=f'EBF Trend', xaxis_title='Constraint Level', yaxis_title="Value")
        fig.write_html(os.path.join(visualization_dir, f'ebf_only.html'))
        print(f"Saved ebf_only.html")

    # third plot, this time plots ebf-performance correlation
    for metric_key in metric_keys:
        fig = go.Figure()
        for model in model_names:
            for task_i, task in enumerate(tasks):
                for role_i, role in enumerate(roles):
                    for min_p_i, min_p in enumerate(ps):
                        for ebf_i, ebf_type in enumerate(ebf_types):
                            if ebf_type == "ht_ppl":
                                continue
                            # add the ebf line
                            ebf_mean = [np.mean([x[min_p][ebf_type] for x in final_results_dict[y][model]['ebf'][task][role]]) for y in constraint_levels]
                            accs = [np.mean(final_results_dict[x][model]['performance'][task][role][metric_key]) for x in constraint_levels]
                            fig.add_trace(go.Scatter(x=ebf_mean, y=accs, mode='markers+lines', name=f'{task}-{role}-ebf ({min_p}, {ebf_type})-{model}'))
        fig.update_layout(title=f'{metric_key}-EBF correlation', xaxis_title='EBF', yaxis_title="Performance")
        fig.write_html(os.path.join(visualization_dir, f'{metric_key}_performance_ebf_scatter.html'))
        print(f"Saved {metric_key}_performance_ebf_scatter.html")


