import gc
import glob
import traceback
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
from uncertainty_quantification.loglik_computation import get_logprob_per_token_from_vllm_outputs

def print_statistics(stats):
    print(f"Top_p avg required number of logprobs: {np.mean(stats['top_p'])}")
    print(f"Min_p avg required number of logprobs: {np.mean(stats['min_p'])}")
    print(f"Top_p max required number of logprobs: {np.max(stats['top_p'])}")
    print(f"Min_p max required number of logprobs: {np.max(stats['min_p'])}")
    # 95% quantile
    print(f"Top_p 95% quantile required number of logprobs: {np.quantile(stats['top_p'], 0.95)}")
    print(f"Min_p 95% quantile required number of logprobs: {np.quantile(stats['min_p'], 0.95)}")
    # 99% quantile
    print(f"Top_p 99% quantile required number of logprobs: {np.quantile(stats['top_p'], 0.99)}")
    print(f"Min_p 99% quantile required number of logprobs: {np.quantile(stats['min_p'], 0.99)}")
    # print support
    print(f"Top_p support: {len(stats['top_p'])}")
    print(f"Min_p support: {len(stats['min_p'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute necessary top_p and min_p count')
    parser.add_argument("--model", type=str, help="model name")
    args = parser.parse_args()
    model = args.model
    model_name = os.path.basename(model)
    p = 0.9
    output_root_dir = "output_logprobs_num_counter"
    os.makedirs(output_root_dir, exist_ok=True)
    ckpt_stat_dictname = os.path.join(output_root_dir, f"check_logprob_{model_name}_num_stat_dict_p_{p}_file_wise.pt")

    if os.path.exists(ckpt_stat_dictname):
        try:
            stat_dict, filewise_dict = torch.load(ckpt_stat_dictname)
        except Exception as e:
            print(f"Error loading {ckpt_stat_dictname}, Exception: {e}")
            stat_dict, filewise_dict = dict(), dict()
    else:
        stat_dict, filewise_dict = dict(), dict()
    # for model in tqdm(models, desc="Processing models", leave=False, position=0):
    print(f"Model: {model}")
    if model in stat_dict and stat_dict[model]["status"] == "finished":
        print(f"Model {model} already finished")
    else:
        # continue
        stat_dict[model_name] = {"top_p": [], "min_p": [], "status": "unfinished"}
        for parent_dir in tqdm(["response_mmlu_256", "response_mmlu_expand_options"], desc="Processing directories", leave=False, position=1):
            subdirs = os.listdir(parent_dir)
            for subdir in tqdm(subdirs, desc="Processing subdirectories", leave=False, position=2):
                dirname = os.path.join(parent_dir, subdir)
                # in case some spectrum file not completed yet
                pt_files = glob.glob(os.path.join(dirname, "{}_response*.pt".format(model_name)))
                for pt_file in tqdm(pt_files, desc="Processing files", leave=False, position=3):
                    if pt_file in filewise_dict and filewise_dict[pt_file]["status"] == "finished":
                        continue
                    filewise_dict[pt_file] = {"top_p": [], "min_p": [], "status": "unfinished"}
                    try:
                        database = torch.load(pt_file)
                        for response in tqdm(database, desc="Processing responses", leave=False, position=4):
                            outputs = response.outputs
                            for output in tqdm(outputs, desc="Processing outputs", leave=False, position=5):
                                logprobs = output.logprobs
                                for token_logprobs in logprobs:
                                    logprobs_item = list(token_logprobs.items())
                                    logprobs_item = [(x[0], get_logprob_per_token_from_vllm_outputs(x[1])) for x in logprobs_item]
                                    logprobs_item.sort(key=lambda x: x[1], reverse=True)
                                    # get top_p necessary number
                                    probs = np.exp([float(x[1]) for x in logprobs_item])
                                    prefix_sum = 0
                                    for _item_i, _item in enumerate(logprobs_item):
                                        if prefix_sum + probs[_item_i] <= p:
                                            prefix_sum += probs[_item_i]
                                        else:
                                            stat_dict[model_name]["top_p"].append(_item_i + 1)
                                            filewise_dict[pt_file]["top_p"].append(_item_i + 1)
                                            break
                                    # get min_p necessary number
                                    for _item_i, _item in enumerate(logprobs_item):
                                        if float(_item[1]) < np.log(p):
                                            stat_dict[model_name]["min_p"].append(_item_i + 1)
                                            filewise_dict[pt_file]["min_p"].append(_item_i + 1)
                                            break
                        del database
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing: {pt_file}, Exception: {e}")
                        # print stack trace
                        traceback.print_exc()
                        if "database" in locals():
                            del database
                        gc.collect()


            stat_dict[model_name]["status"] = "finished"
            filewise_dict[pt_file]["status"] = "finished"
            torch.save([stat_dict, filewise_dict], ckpt_stat_dictname)
    for model in stat_dict:
        print(f"Model: {model}")
        if len(stat_dict[model]['top_p']) == 0:
            print("Empty logits, perhaps the model is not finished yet")
            continue
        print_statistics(stat_dict[model])

    for pt_file in filewise_dict:
        print(f"File: {pt_file}")
        if len(filewise_dict[pt_file]['top_p']) == 0:
            print("Empty logits, perhaps the model is not finished yet")
            continue
        print_statistics(filewise_dict[pt_file])

#