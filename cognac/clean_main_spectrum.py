import argparse
import gc

import torch
import os

# from consts import task_prompt, roles
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from cognac_utils import get_cognac_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
    step1_forward_args(parser, sample_counts=50, max_tokens=512, log_probs=70, chat_template_path=None,
                       output_root_dir="cognac_ctrl_outputs", seed=42, min_p=0, top_p=1.0, temperature=1.0)
    parser.add_argument("--task_selection_filename", type=str, default="sampled_task_cognac_app_300.pt", help="task file")
    parser.add_argument("--multi_constraints", type=int, default=1, help="multi constraints (cognac only)")
    args = parser.parse_args()

    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    if "Yi" in model:
        print(
            "We are sorry, but Yi-series model currently (as of 06/25/2024) is not supported well (dashboard cannot initialize) for VLLM, so we need to skip it for now")
        exit()
    chat_template_path = args.chat_template_path

    if args.min_p > 0:
        file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_seed{}.pt".format(os.path.basename(model),
                                                                                            args.sample_counts,
                                                                                            args.max_tokens, args.log_probs,
                                                                                            args.min_p, args.seed)
    else:
        file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_top_p_{}_seed{}.pt".format(os.path.basename(model),
                                                                                            args.sample_counts,
                                                                                            args.max_tokens, args.log_probs,
                                                                                            args.top_p, args.seed)
    file_name = os.path.join(output_root_dir, file_name)
    manager = ForwardManager(args, ckpt_freq=128)
    if os.path.exists(args.task_selection_filename):
        # tasks, task_to_ids = torch.load("sampled_task.pt")
        tasks, task_to_ids = torch.load(args.task_selection_filename)
        assert list(task_to_ids.keys()) == ['cognac'], "Only one task is allowed for cognac"
    else:
        print("Please run create_sampled_task_cognac.py first!")
        exit()

    if args.beam_search:
        print("Using beam search for decoding, following vllm, we set temperature to be 0 to disable randomness")
        args.temperature = 0.0

    metadata_filename = file_name.replace(".pt", ".metadata")
    if os.path.exists(metadata_filename):
        prompts, answers, sources, _ = torch.load(metadata_filename)
        print("Loaded from metadata file")
    else:
        prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data(model, chat_template_path,
                                                                               select_ids=task_to_ids['cognac'],
                                                                               update_args=args)
        torch.save([prompts, answers, sources, (hierarchy, updated_args)], metadata_filename)
    maybe_exist_flag = False
    response = None
    if os.path.exists(file_name):
        print("File exists: {}".format(file_name))
        try:
            response = torch.load(file_name)
            assert len(response) == len(prompts), "length mismatch"
            maybe_exist_flag = True
        except Exception as e:
            print("File exists but cannot be loaded: {} (Exception: {})".format(file_name, e))
            # if response is already loaded, delete it to save memory as forward manager would load again
            if response is not None:
                del response
                gc.collect()
            maybe_exist_flag = False
    if not maybe_exist_flag:
        print("File does not exist, under cleaning mode, will exit directly")
        exit()
        # gpu_memory_utilization = compute_gpu_memory_utilization(model)
        # # response = manager.forward(prompts, file_name, max_num_seqs=4 if "llama-3" in model.lower() else 8, gpu_memory_utilization=gpu_memory_utilization)
        # max_num_seqs = 16
        # # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
        # if "70b" in model.lower() and "llama-3" in model.lower():
        #     gpu_memory_utilization = 0.5
        #     max_num_seqs = 128
        # if "8b" in model.lower():
        #     gpu_memory_utilization = 0.3
        #     max_num_seqs = 128
        # response = manager.forward(prompts, file_name, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization)
        # torch.save(response, file_name)
        #if "llama-3" in model.lower():
            #print("Llama-3: have to restart to avoid weird problems from vllm hanging (0.4.3 and 0.5.4), please relaunch the script")
            #exit()
    # step-2: get full spectrum of probability
    manager.fillin_logits_routine(response, file_name, max_num_seqs=4, gpu_memory_utilization=0.8, clean_mode=True)


