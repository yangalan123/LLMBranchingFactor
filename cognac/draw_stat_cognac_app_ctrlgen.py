import argparse
import glob
import plotly.graph_objects as go
import os

import re

def extract_offset(filename):
    match = re.search(r"offset_(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        return None  # No offset found

def extract_float_list(text):
    match = re.search(r"\[\s*(-?\d+(\.\d+)?)\s*(,\s*-?\d+(\.\d+)?)*\s*\]", text)
    if match:
        float_str_list = match.group().strip("[]").split(",")
        return [float(x) for x in float_str_list]  # Convert to float list
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RunDrawStatCognacAppCtrlGenParsing.')
    parser.add_argument("--stat_dir", type=str, default="stat_cognac_app_ctrlgen_multi_constraints/output_manual_check_cognac_responses_200_app_ctrlgen_multi_constraints_max_tokens_512_min_p_0_top_p_0.9", help="stat dir")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--high_entropy_threshold", type=float, default=0.7, help="high entropy threshold")
    args = parser.parse_args()
    stat_dir = args.stat_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(stat_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(stat_dir, f"*_high_ent_threshold_{args.high_entropy_threshold}.log"))
    model_dict = {}
    constraints_number = None
    ebf_type = None
    for file in files:
        print(f"Processing {file}")
        # extract offset from file name with regex -- offset is the number in "offset_{}_high_ent_threshold_{}.log"
        offset = extract_offset(file)
        with open(file, "r") as f:
            lines = f.readlines()
        cur_model = None
        for line in lines:
            if "Model" in line:
                cur_model = line.strip().split(" ")[-1]
                if cur_model not in model_dict:
                    model_dict[cur_model] = dict()
            if "avg_ebf" in line:
                if ebf_type is None:
                    ebf_type = line.strip().split(": ")[0]
                ebf_values_across_constraints = extract_float_list(line)
                # ebf_values looks like ([1,2,3,4,5]), parse it to be a list of floats
                assert isinstance(ebf_values_across_constraints, list)
                ebf_values_across_constraints = [float(ebf) for ebf in ebf_values_across_constraints]
                if ebf_type not in model_dict[cur_model]:
                    model_dict[cur_model][ebf_type] = dict()
                model_dict[cur_model][ebf_type][offset] = ebf_values_across_constraints
                if constraints_number is None:
                    constraints_number = len(ebf_values_across_constraints)
                else:
                    assert constraints_number == len(ebf_values_across_constraints), "constraints number should be the same, detected different constraints number: {} vs {}".format(constraints_number, len(ebf_values_across_constraints))
    # plot start
    # x-axis: offset number
    # y-axis: ebf value for model+ebf_type+constraint_level(index in ebf_values_across_constraints)
    fig = go.Figure()
    for model in model_dict:
        x_values = list(model_dict[model][ebf_type].keys())
        x_values.sort()
        constraints_levels = [i for i in range(constraints_number)]
        for i in range(constraints_number):
            y_values = [model_dict[model][ebf_type][offset][i] for offset in x_values]
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=f"{model}_constraint_{i + 1}"))
    fig.update_layout(title=f"{ebf_type} across different constraints", xaxis_title='Offset', yaxis_title=f'{ebf_type} Value')
    fig.write_html(os.path.join(output_dir, f"{ebf_type}_across_constraints.html"))
    print("Done!")





