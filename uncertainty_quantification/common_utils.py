import random
import subprocess

import numpy as np
import torch
import os


def get_gpu_memory_and_usage_rate(logger):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout
            gpu_info_lines = gpu_info.strip().split("\n")
            gpu_summary = [line.split(", ") for line in gpu_info_lines]
            for gpu in gpu_summary:
                logger.info(f"GPU {gpu[0]}: {gpu[1]}, Total Memory: {gpu[2]} MB, Used Memory: {gpu[3]} MB, Free Memory: {gpu[4]} MB, Utilization: {gpu[5]}%")
        else:
            logger.error("Error executing nvidia-smi")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def recursive_type_convert_numpy_to_tensor(item):
    if isinstance(item, dict):
        for key in item:
            item[key] = recursive_type_convert_numpy_to_tensor(item[key])
    elif isinstance(item, list):
        for i in range(len(item)):
            item[i] = recursive_type_convert_numpy_to_tensor(item[i])
    elif isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    return item

def get_update_full_spectrum_file_pattern(source_dir, model_name, args):
    if args.min_p > 0:
        # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
        file_pattern = os.path.join(source_dir,
                                    f"{model_name}_response*max_tokens_{args.max_tokens}*min_p_{args.min_p}_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
    else:
        # we have to make sure model_name directly followed by _response -- as llama-3-8B and llama-3-8B-Instruct share the same prefix (not the problem for other models)
        file_pattern = os.path.join(source_dir,
                                    f"{model_name}_response*max_tokens_{args.max_tokens}*top_p_{args.top_p}_*{args.additional_file_search_pattern + '*' if len(args.additional_file_search_pattern) > 0 else ''}.pt.update_full_spectrum")
    return file_pattern
