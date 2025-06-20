import gc
import glob
from collections import defaultdict
import numpy as np

import torch
import os
from plotly import graph_objects as go
from tqdm import tqdm
if __name__ == '__main__':
    models=("meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            )
            # "meta-llama/Meta-Llama-3-70B")
    output_root_dir = "response_news_tagger"
    constraints = [0, 1, 2, 3, 4, 5]
    final_result_dict = defaultdict(dict)
    for model in tqdm(models):
        model_base = os.path.basename(model)
        constraint_dirs = [os.path.join(output_root_dir, f"application_ctrlgen_multi_constraints_{constraint}") for constraint in constraints]
        for constraint_i, constraint_dir in enumerate(constraint_dirs):
            final_result_dict[model][constraint_i] = {
                "mean_num_tags": [],
                "std_num_tags": [],
                "num_tags": [],
            }
            filenames = glob.glob(os.path.join(constraint_dir, f"{model_base}_response*tags"))
            assert len(filenames) == 1, f"Model {model_base} has {len(filenames)} tags for constraint {constraint_dir}"
            filename = filenames[0]
            responses = torch.load(filename)
            metadata_filename = filename.replace(".tags", ".metadata")
            prompts, all_original_outputs, all_original_prompts, args = torch.load(metadata_filename)
            bad_example_flag = True
            fail_to_parse = 0
            response_counter = 0
            for instance_i in all_original_prompts:
                original_prompt, start_id, end_id = instance_i
                response_group = responses[start_id:end_id]
                tag_nums = []
                for _iter, response_group_i in enumerate(response_group):
                    response_counter += 1
                    outputs = response_group_i.outputs
                    output_texts = [output.text for output in outputs if "News Tags:" in output.text]
                    # extract tags from each text in output_texts -- should start with News Tags: and end with .
                    # seperated by comma
                    tags = []
                    for output_text in output_texts:
                        tags.append(output_text.split("News Tags:")[1].split(".")[0].strip().split(","))
                    if len(tags) == 0:
                        if bad_example_flag:
                            print("Failed to parse tags from output_texts for model-{} constraint-{} instance-{} instance_output-{}".format(model, constraint_i, instance_i, _iter))
                            print("Example fail-to-parse output_texts: {}".format(outputs[0].text))
                            bad_example_flag = False
                        fail_to_parse += 1
                        tag_nums.append(0)
                        continue
                    _tag_nums = [len(tag) for tag in tags]
                    tag_nums.append(np.mean(_tag_nums))
                # final_result_dict[model][constraint_i].append([np.mean(tag_nums), np.std(tag_nums), len(tag_nums), fail_to_parse, response_counter])
                final_result_dict[model][constraint_i]["mean_num_tags"].append(np.mean(tag_nums))
                final_result_dict[model][constraint_i]["std_num_tags"].append(np.std(tag_nums))
                final_result_dict[model][constraint_i]["num_tags"].append(len(tag_nums))
            final_result_dict[model][constraint_i]["fail_to_parse_rate"] = fail_to_parse / response_counter
            del responses
            del prompts, all_original_outputs, all_original_prompts, args
            gc.collect()
    fig = go.Figure()
    # x: constraints, y: mean number of tags
    # note that for y, we will use weighted mean, weight is the number of tags
    for model, constraint_dict in final_result_dict.items():
        xs = list(range(len(constraint_dict)))
        ys = []
        errs = []
        # for constraint_i, tag_info in constraint_dict.items():
        #     tag_info = np.array(tag_info)
        #     mean_num_tags = np.average(tag_info[:, 0], weights=tag_info[:, 2])
        #     std_num_tags = np.sqrt(np.average((tag_info[:, 0] - mean_num_tags) ** 2, weights=tag_info[:, 2]))
        #     ys.append(mean_num_tags)
        #     errs.append(std_num_tags)
        # fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=model))
        for constraint_i, tag_info in constraint_dict.items():
            ys.append(np.mean(tag_info["mean_num_tags"]))
            errs.append(np.mean(tag_info["std_num_tags"]))
        fig.add_trace(go.Scatter(x=xs, y=ys, error_y=dict(type='data', array=errs), mode='lines+markers', name=model))


    # save the plot
    visualization_dir = "visualization_tagnumber"
    os.makedirs(visualization_dir, exist_ok=True)
    fig.write_html(os.path.join(visualization_dir, "tag_number.html"))
    # plot fail-to-parse-rate
    fig = go.Figure()
    for model, constraint_dict in final_result_dict.items():
        xs = list(range(len(constraint_dict)))
        ys = []
        for constraint_i, tag_info in constraint_dict.items():
            ys.append(tag_info["fail_to_parse_rate"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=model))
    fig.write_html(os.path.join(visualization_dir, "fail_to_parse_rate.html"))



