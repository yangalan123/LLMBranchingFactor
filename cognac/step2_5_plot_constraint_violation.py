import gc
import glob
from collections import defaultdict
import numpy as np

import torch
import os
from plotly import graph_objects as go
from tqdm import tqdm
from uncertainty_quantification.loglik_computation import compute_loglik
from uncertainty_quantification.visualization_utils import matplotlib_plot, model_name_visualization_name_mapping
if __name__ == '__main__':
    models=("meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/Meta-Llama-3-70B-Instruct",)
    output_root_dir = "cognac_responses_200_violating_constraint"
    constraints = [2, 3, 4, 5]
    final_result_dict = defaultdict(dict)
    source_constraint = 1
    for model in tqdm(models):
        model_base = os.path.basename(model)
        constraint_dirs = [os.path.join(output_root_dir, f"application_ctrlgen_multi_constraints_{constraint}") for constraint in constraints]
        for constraint_i, constraint_dir in enumerate(constraint_dirs):
            final_result_dict[model][constraint_i] = {
                "loglik_delta": [],
                "loglik_new": [],
                "loglik_old": []
            }
            suffix = f"from_{source_constraint}_to_{constraints[constraint_i]}"
            filenames = glob.glob(os.path.join(constraint_dir, f"{model_base}_response*top_p_0.9_*{suffix}"))
            assert len(filenames) == 1, f"Model {model_base} has {len(filenames)} {suffix} for constraint {constraint_dir}: {filenames}"
            filename = filenames[0]
            responses = torch.load(filename)
            metadata_filename = filename + ".metadata"
            [prompts, all_original_prompts, all_original_logliks, all_original_metrics] = torch.load(metadata_filename)
            bad_example_flag = True
            fail_to_parse = 0
            response_counter = 0
            for instance_i in all_original_prompts:
                target_prompt, start_id, end_id, source_prompt = instance_i
                response_group = responses[start_id:end_id]
                original_output_logliks_group = [x[-1] for x in all_original_logliks[start_id:end_id]]
                original_target_prompt_logliks_group = [x[0] for x in all_original_logliks[start_id:end_id]]
                # actually all these target_prompt_logliks should be the same, but just in case of floating error, let's use std to check
                try:
                    assert np.std(original_target_prompt_logliks_group) < 1e-6, f"std of target prompt logliks is {np.std(original_target_prompt_logliks_group)}"
                except:
                    print("Original target prompt_logliks_group", original_target_prompt_logliks_group)
                    exit()
                target_prompt_logliks = np.mean(original_target_prompt_logliks_group)

                # metrics_group = all_original_metrics[start_id:end_id]
                logliks = []
                for _iter, response_group_i in enumerate(response_group):
                    response_counter += 1
                    outputs = response_group_i.outputs
                    output_prompt_loglik = compute_loglik(response_group_i.prompt_token_ids, response_group_i.prompt_logprobs)
                    # log P(Y_1 | X_5) = log P(Y_1, X_5) - log P(X_5)
                    logliks.append(output_prompt_loglik - target_prompt_logliks)
                # log P(Y_1 | X_5) - log P(Y_1 | X_1)
                final_result_dict[model][constraint_i]["loglik_delta"].append(np.mean(logliks) - np.mean(original_output_logliks_group))
                final_result_dict[model][constraint_i]["loglik_new"].append(np.mean(logliks))
                final_result_dict[model][constraint_i]["loglik_old"].append(np.mean(original_output_logliks_group))
                # final_result_dict[model][constraint_i].append([np.mean(tag_nums), np.std(tag_nums), len(tag_nums), fail_to_parse, response_counter])
            del responses
            del prompts, all_original_prompts, all_original_logliks, all_original_metrics
            gc.collect()
    visualization_dir = "visualization_constraint_violation"
    os.makedirs(visualization_dir, exist_ok=True)
    for key in ["loglik_delta", "loglik_new", "loglik_old"]:
        # x: constraints, y: mean number of tags
        # note that for y, we will use weighted mean, weight is the number of tags
        fig = go.Figure()
        for model, constraint_dict in final_result_dict.items():
            xs = list(range(len(constraint_dict)))
            xs = [x+1 for x in xs]
            ys = []
            errs = []
            for constraint_i, tag_info in constraint_dict.items():
                ys.append(np.mean(tag_info[key]))
                errs.append(np.mean(tag_info[key]))
            # fig.add_trace(go.Scatter(x=xs, y=ys, error_y=dict(type='data', array=errs), mode='lines+markers', name=model))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=model))
            # change x-axis to constraint level ("from-1-to-x")
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=xs, ticktext=[f"output-1-prompt-{x}" for x in constraints]))


        # save the plot
        fig.write_html(os.path.join(visualization_dir, f"{key}.html"))
        # create matplotlib plot
        if key == "loglik_delta":
            local_dict = dict()
            for model in final_result_dict:
                renamed_model = os.path.basename(model)
                local_dict[model_name_visualization_name_mapping(renamed_model)] = {key: [np.mean(final_result_dict[model][x][key]) for x in range(len(constraints))] }
            save_path = os.path.join(visualization_dir, f"{key}.pdf")
            matplotlib_plot(constraints, local_dict, save_path, tag=key, y_label="LogLik Diff", fontsize=50, linewidth=5, figsize=(20, 15), simple_axis_adjust=True)
            print("Saved to", save_path)



