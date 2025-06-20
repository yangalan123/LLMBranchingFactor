import argparse
import plotly.graph_objects as go
import traceback
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from collections import defaultdict

import torch
# from consts import task_prompt, roles
from cognac.cognac_utils import get_default_args
from cognac.cognac_metrics import compute_prediction_metrics

from uncertainty_quantification.uncertainty_computation import compute_ebf_from_vllm_outputs
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    parser.add_argument('--source_dir', type=str, help='source dir', default="cognac_responses/application_ctrlgen_datawise")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_cognac_app_300.pt",
                        help="task file")
    args = parser.parse_args()

    source_dir = args.source_dir
    model = args.model
    files = glob.glob(os.path.join(source_dir, "*{}*.pt".format(os.path.basename(model))))
    # os.makedirs(output_root_dir, exist_ok=True)


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
    min_ps = [0.1]
    repeat_times = 10
    example_metric_flag = True

    # file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_seed{}.pt".format(os.path.basename(model), args.sample_counts,
    #                                                                                     args.max_tokens, args.log_probs, args.min_p, args.seed)
    # file_name = os.path.join(input_root_dir, file_name)
    for filename in tqdm(files, desc="Processing files", leave=False):
        print("Processing file: {}".format(filename))
        try:
            metadata_filename = filename.replace(".pt", ".metadata")
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
            task_wise_performance = dict()
            task_wise_ebf = dict()
            self_consistency_ranges = range(1, 51)
            metric_keys = None
            ebf_types = None
            for task_i, task in enumerate(tasks):
                start_id = task_starts_id[task]
                task_data_num = len(task_to_ids[task])
                # we actually use eval_version=0 prompt (prompt_id=0 in diverse_instruction.csv)
                task_wise_performance[task] = {'std': dict()}
                task_wise_ebf[task] = {'std': []}

                for idx in tqdm(range(start_id, start_id + task_data_num), desc=f"Processing task {task}", leave=False):
                    original_instance_i = task_to_ids[task][(idx - start_id) % task_data_num]
                    data = database[idx]
                    all_outputs = data.outputs
                    all_outputs_texts = [x.text.strip() for x in all_outputs]
                    answer = answers[idx]
                    role = sources[idx]
                    estimated_ebf = compute_ebf_from_vllm_outputs(all_outputs, ps=min_ps)
                    if ebf_types is None:
                        ebf_types = list(estimated_ebf[min_ps[0]].keys())
                    task_wise_ebf[task][role].append(estimated_ebf)
                    assert prompts[idx] == data.prompt, "Prompt not match!"
                    for number_of_sampled_path in self_consistency_ranges:
                        if number_of_sampled_path not in task_wise_performance[task][role]:
                            task_wise_performance[task][role][number_of_sampled_path] = defaultdict(list)
                        metrics = []
                        for repeat_j in range(repeat_times):
                            sampled_path = random.sample(all_outputs_texts, number_of_sampled_path)
                            for _path in sampled_path:
                                metric = compute_prediction_metrics({"datapoint": answer, "generated_text": _path}, hierarchy, "wordnet", multi_constraints_eval=updated_args.multi_constraints > 1 or "multi_constraints" in answer)['prediction_metric']
                                metrics.append(metric)
                                if metric_keys is None:
                                    metric_keys = list(metric.keys())
                                    if example_metric_flag:
                                        print("Example metric: {}".format(metric))
                                        print("Example path: {}".format(_path))
                                        print("Example answer: {}".format(answer))
                                        example_metric_flag = False

                        for metric_key in metric_keys:
                            perf = [x[metric_key] for x in metrics]
                            task_wise_performance[task][role][number_of_sampled_path][metric_key].append(sum(perf) / len(perf))
                            task_wise_performance[task][role][number_of_sampled_path][metric_key + '_std'].append(np.std(perf))

            # plot the results
            # x: number_of_sampled_path
            # y: acc, error_bar: std
            # additionally draw a horizontal line for ebf
            # plot both roles in the same figure
            # use different colors and legend for each role
            # save the figure as a pdf file
            visualization_dir = os.path.join(source_dir, "visualization", os.path.basename(filename).replace(".pt", ""))
            os.makedirs(visualization_dir, exist_ok=True)
            colors = sns.color_palette("Set1", n_colors=len(tasks) * len(roles) * len(min_ps) * (2 * len(metric_keys) + 1))
            for task_i, task in enumerate(tasks):
                # split the above codes to plot two charts -- one for number_of_sampled_paths vs acc, the other is a bar chart for ebf
                color_count = 0
                for metric_key in metric_keys:
                    fig, ax = plt.subplots()
                    for role_i, role in enumerate(roles):
                        number_of_sampled_paths = list(task_wise_performance[task][role].keys())
                        number_of_sampled_paths.sort()
                        accs = [np.mean(task_wise_performance[task][role][x][metric_key]) for x in number_of_sampled_paths]
                        stds = [np.mean(task_wise_performance[task][role][x][metric_key + '_std']) for x in number_of_sampled_paths]
                        ax.errorbar(number_of_sampled_paths, accs, yerr=stds, fmt='o', label=role+f"_{metric_key}", color=colors[color_count])
                        color_count += 1
                    ax.set_xlabel('#(Reasoning Paths) (=#(outputs))')
                    ax.set_ylabel(f'{metric_key}')
                    plt.title(f'{task} {metric_key}')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(visualization_dir, f'{task}_{metric_key}_self_consistency.pdf'))
                    plt.clf()
                    plt.close()
                fig, ax = plt.subplots()
                color_count = 0
                for role_i, role in enumerate(roles):
                    for min_p_i, min_p in enumerate(min_ps):
                        for ebf_i, ebf_type in enumerate(ebf_types):
                            if ebf_type == "ht_ppl":
                                continue
                            ebf_mean = np.mean([x[min_p][ebf_type] for x in task_wise_ebf[task][role]])
                            ax.bar(f'{role}-{min_p}-{ebf_type}', ebf_mean, color=colors[color_count])
                            color_count += 1
                ax.set_xlabel('Role-Min_p')
                ax.set_ylabel('Estimated BF')
                plt.title(f'{task} ebf self_consistency')
                plt.tight_layout()
                plt.savefig(os.path.join(visualization_dir, f'{task}_ebf_self_consistency.pdf'))
                plt.clf()
                plt.close()

            # rewrite the codes using plotly to give a nicer-looking figure
            # feel free to use another set of colors for the lines
            # the results should be saved as a html file
            # the figure should be interactive
            for metric_key in metric_keys:
                fig = go.Figure()
                for task_i, task in enumerate(tasks):
                    for role_i, role in enumerate(roles):
                        number_of_sampled_paths = list(task_wise_performance[task][role].keys())
                        number_of_sampled_paths.sort()
                        accs = [np.mean(task_wise_performance[task][role][x][metric_key]) for x in number_of_sampled_paths]
                        stds = [np.mean(task_wise_performance[task][role][x][metric_key + '_std']) for x in number_of_sampled_paths]
                        # add the acc line, with stds as the error bar
                        fig.add_trace(go.Scatter(x=number_of_sampled_paths, y=accs, mode='markers+lines', name=f'{task}-{role}-{metric_key}'))
                        sample_counts = len(task_wise_performance[task][role][number_of_sampled_paths[0]][metric_key])
                        max_acc_paths = []
                        for sample_i in range(sample_counts):
                            accs = [task_wise_performance[task][role][x][metric_key][sample_i] for x in number_of_sampled_paths]
                            acc_quantile = np.quantile(accs, 0.8)
                            acc_above_quantile = [(x_i, x) for x_i, x in enumerate(accs) if x >= acc_quantile]
                            acc_above_quantile.sort(key=lambda x: x[0])
                            max_acc_path = number_of_sampled_paths[acc_above_quantile[0][0]]
                            max_acc_paths.append(max_acc_path)
                        # add a line for max_acc_path
                        fig.add_trace(go.Scatter(x=number_of_sampled_paths, y=[np.mean(max_acc_paths)] * len(number_of_sampled_paths), mode='lines', name=f'{task}-{role}-quantile_{metric_key}_path'))
                        for min_p_i, min_p in enumerate(min_ps):
                            for ebf_i, ebf_type in enumerate(ebf_types):
                                if ebf_type == "ht_ppl":
                                    continue
                                # add the ebf line
                                ebf_mean = np.mean([x[min_p][ebf_type] for x in task_wise_ebf[task][role]])
                                fig.add_trace(go.Scatter(x=number_of_sampled_paths, y=[ebf_mean] * len(number_of_sampled_paths), mode='lines', name=f'{task}-{role}-ebf ({min_p}, {ebf_type})'))
                fig.update_layout(title=f'{metric_key}-EBF correlation', xaxis_title='#(Reasoning Paths) (=#(outputs))', yaxis_title='Estimated BF')
                fig.write_html(os.path.join(visualization_dir, f'{metric_key}_ebf_self_consistency.html'))

            for metric_key in metric_keys:
                fig = go.Figure()
                for task_i, task in enumerate(tasks):
                    for role_i, role in enumerate(roles):
                        number_of_sampled_paths = list(task_wise_performance[task][role].keys())
                        number_of_sampled_paths.sort()
                        sample_counts = len(task_wise_performance[task][role][number_of_sampled_paths[0]][metric_key])
                        for min_p in min_ps:
                            max_acc_paths = []
                            ebfs = defaultdict(list)
                            for sample_i in range(sample_counts):
                                accs = [task_wise_performance[task][role][x][metric_key][sample_i] for x in number_of_sampled_paths]
                                acc_argsort = np.argsort(accs)
                                acc_quantile = np.quantile(accs, 0.8)
                                acc_above_quantile = [(x_i, x) for x_i, x in enumerate(accs) if x >= acc_quantile]
                                acc_above_quantile.sort(key=lambda x: x[0])
                                max_acc_path = number_of_sampled_paths[acc_above_quantile[0][0]]

                                # max_acc_path = number_of_sampled_paths[np.argmax(accs)]
                                max_acc_paths.append(max_acc_path)
                                for ebf_key in ebf_types:
                                    ebfs[ebf_key].append(task_wise_ebf[task][role][sample_i][min_p][ebf_key])
                                    # ebfs['cond_ppl_prod'].append(task_wise_ebf[task][role][sample_i][min_p]['cond_ppl_prod'])
                            for ebf_key in ebf_types:
                                fig.add_trace(go.Scatter(x=max_acc_paths, y=ebfs[ebf_key], mode='markers', name=f'{task}-{role}-ebf-{ebf_key}_{min_p}'))
                            # fig.add_trace(go.Scatter(x=max_acc_paths, y=ebfs['cond_ppl_prod'], mode='markers', name=f'{task}-{role}-ebf-cond_ppl_prod_{min_p}'))
                        # add y=x line
                fig.add_trace(go.Scatter(x=[0, 32], y=[0, 32], mode='lines', name='y=x'))
                fig.update_layout(title=f'{metric_key}-EBF Scatter', xaxis_title='#(Reasoning Paths) (=#(outputs))', yaxis_title='Estimated BF')
                fig.write_html(os.path.join(visualization_dir, f'{metric_key}_ebf_scatter.html'))



                        # # add the acc line, with stds as the error bar
                        # fig.add_trace(go.Scatter(x=number_of_sampled_paths, y=accs, mode='lines', name=f'{task}-{role}-acc-{sample_i}'))
        except Exception as e:
            print(e)
            # print stack trace
            traceback.print_exc()
            print("Failed to process file: {}".format(filename))






