import argparse
import gc
import os
import numpy as np

import torch
from vllm import LLM, SamplingParams
# from consts import task_prompt, roles
from tqdm import tqdm

parser = argparse.ArgumentParser(description='NormalNLPTaskBranchingFactorEvaluationArgsParsing.')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--sample_counts', type=int, help='sample counts', default=50)
parser.add_argument('--max_tokens', type=int, help='max tokens', default=512)
parser.add_argument('--log_probs', type=int, help='log probs', default=50)
parser.add_argument('--chat_template_path', type=str, help='chat template path', default=None)
parser.add_argument('--output_root_dir', type=str, help='output root dir', default="cognac_ctrl_outputs")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--min_p", type=float, default=0, help="minimum p value for sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="minimum p value for sampling")
parser.add_argument("--beam_search", action="store_true", help="use beam search for decoding")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
parser.add_argument("--task_selection_filename", type=str, default="sampled_task_cognac_app_300.pt", help="task file")
parser.add_argument("--multi_constraints", type=int, default=1, help="multi constraints (cognac only)")
args = parser.parse_args()

output_root_dir = args.output_root_dir
os.makedirs(output_root_dir, exist_ok=True)

model = args.model
if "Yi" in model:
    print("We are sorry, but Yi-series model currently (as of 06/25/2024) is not supported well (dashboard cannot initialize) for VLLM, so we need to skip it for now")
    exit()
chat_template_path = args.chat_template_path

if args.min_p > 0:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_seed{}.pt".format(os.path.basename(model), args.sample_counts,
                                                                                        args.max_tokens, args.log_probs, args.min_p, args.seed)
else:
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_top_p_{}_seed{}.pt".format(os.path.basename(model), args.sample_counts,
                                                                                        args.max_tokens, args.log_probs, args.top_p, args.seed)
file_name = os.path.join(output_root_dir, file_name)
if not os.path.exists(file_name):
    print("File not exists: {}".format(file_name))
    exit()
else:
    response = torch.load(file_name)
# tasks = list(standard_prompt.keys())
#if os.path.exists("sampled_task.pt"):
if os.path.exists(args.task_selection_filename):
    #tasks, task_to_ids = torch.load("sampled_task.pt")
    tasks, task_to_ids = torch.load(args.task_selection_filename)
    assert list(task_to_ids.keys()) == ['cognac'], "Only one task is allowed for cognac"
else:
    print("Please run create_sampled_task_cognac.py first!")
    exit()

if args.beam_search:
    print("Using beam search for decoding, following vllm, we set temperature to be 0 to disable randomness")
    args.temperature = 0.0

    # random.seed(1)
    # tasks = random.sample(tasks, 25)
    # task_to_ids = dict()

metadata_filename = file_name.replace(".pt", ".metadata")
# standard_prompt = load_task_prompts(STANDARD_PROMPT_PATH)
# cot_prompt = load_task_prompts(COT_PROMPT_PATH)
assert os.path.exists(metadata_filename), "metadata file not exists: {}".format(metadata_filename)
prompts, answers, sources, _ = torch.load(metadata_filename)
print("Loaded from metadata file")

# we need to combine output token ids and prompt token ids in the original response to make up a new one
update_filename = file_name.replace(".pt", ".pt.update_full_spectrum")
exists_update_flag = False
if os.path.exists(update_filename):
    try:
        updated_response = torch.load(update_filename)
        exists_update_flag = True
        print("File exists: {}".format(update_filename))
        del updated_response
        gc.collect()
        exit()
    except Exception as e:
        print("Error loading update file: {}, we have to recreate it".format(e))
if not exists_update_flag:
    # 1) prepare new prompt_token_ids
    new_prompt_token_ids = []
    for _response in tqdm(response, desc="Preparing New Prompt Token IDs"):
        outputs = _response.outputs
        prompt_token_ids = _response.prompt_token_ids
        for output in outputs:
            output_token_ids = output.token_ids
            new_prompt_token_ids.append(prompt_token_ids + output_token_ids)

    # response = llm.generate(prompts, sampling_params, use_tqdm=True)
    # torch.save(response, file_name)
    patch_filename = file_name.replace(".pt", ".patch_ckpt")
    exist_patch_flag = False
    if os.path.exists(patch_filename):
        try:
            update_response = torch.load(patch_filename)
            exist_patch_flag = True
        except Exception as e:
            print("Error loading patch file: {}, we have to recreate it".format(e))

    if not exist_patch_flag:
        # in llama-3-8b/70b, we have to reduce utilization to 0.8 and max_num_seqs to 4 to avoid OOM
        llm = LLM(model=model, tensor_parallel_size=4, seed=args.seed, gpu_memory_utilization=0.8,
                  max_logprobs=args.log_probs + 20, max_num_seqs=4)
        sampling_params = SamplingParams(n=1, max_tokens=1, logprobs=args.log_probs,
                                         prompt_logprobs=args.log_probs, use_beam_search=args.beam_search,
                                         temperature=args.temperature,
                                         )
        average_length = np.mean([len(x) for x in new_prompt_token_ids])
        print("Average Token Limits", average_length)
        max_length = np.max([len(x) for x in new_prompt_token_ids])
        print("Max Token Limits", max_length)
        update_response = llm.generate(prompt_token_ids=new_prompt_token_ids, sampling_params=sampling_params, use_tqdm=True)
        torch.save(update_response, patch_filename)
    counter = 0
    for response_i in tqdm(range(len(response)), desc="Updating Response"):
        outputs = response[response_i].outputs
        prompt_token_ids = response[response_i].prompt_token_ids
        output_num = len(outputs)
        for output_j in range(output_num):
            output_token_ids = outputs[output_j].token_ids
            assert update_response[counter].prompt_token_ids == prompt_token_ids + output_token_ids, "Prompt Token IDs Mismatch"
            update_prompt_logprobs = update_response[counter].prompt_logprobs[len(prompt_token_ids):]
            assert len(response[response_i].outputs[output_j].logprobs) == len(update_prompt_logprobs), "Logprobs Length Mismatch: {} vs {}".format(len(response[response_i].outputs[output_j].logprobs), len(update_prompt_logprobs))
            response[response_i].outputs[output_j].logprobs = update_prompt_logprobs
            counter += 1

    torch.save(response, update_filename)
