import argparse
import os

import torch
from vllm import LLM, SamplingParams
# from consts import task_prompt, roles
from cognac_utils import get_cognac_data

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
if os.path.exists(file_name):
    print("File exists: {}".format(file_name))
    exit()
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
if os.path.exists(metadata_filename):
    prompts, answers, sources, _ = torch.load(metadata_filename)
    print("Loaded from metadata file")
else:
    # all_data = get_gsm8k_data()
    # prompts, answers, sources, regex_postprocessor = create_gsm8k_data(model, all_data, chat_template_path, cot="std" not in output_root_dir, select_ids=task_to_ids['gsm8k'])
    prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data(model, chat_template_path, select_ids=task_to_ids['cognac'], update_args=args)
    # prompts, answers, sources, tasks = (model, chat_template_path, tasks, task_to_ids, metadata_filename)
    torch.save([prompts, answers, sources, (hierarchy, updated_args)], metadata_filename)
# if need_sample_task_flag:
#     torch.save([tasks, task_to_ids], "sampled_task.pt")
#     print("Sampled tasks and saved to sampled_task.pt")
def eta_truncation_logit_processor(token_ids, logits, eta=args.min_p):
    probs = torch.softmax(logits, dim=-1)
    prob_mask = probs < eta
    logits = logits.masked_fill_(prob_mask, float('-inf'))
    return logits
    # return torch.clamp(logit, min=-10.0, max=10.0)

llm = LLM(model=model, tensor_parallel_size=4, seed=args.seed, gpu_memory_utilization=0.8, max_logprobs=args.log_probs + 20)
if args.min_p > 0:
    sampling_params = SamplingParams(n=args.sample_counts, max_tokens=args.max_tokens, logprobs=args.log_probs,
                                     prompt_logprobs=args.log_probs, min_p=args.min_p, use_beam_search=args.beam_search, temperature=args.temperature,
                                     logits_processors=[eta_truncation_logit_processor])
else:
    # Use top_p nucleaus sampling
    sampling_params = SamplingParams(n=args.sample_counts, max_tokens=args.max_tokens, logprobs=args.log_probs,
                                     prompt_logprobs=args.log_probs, top_p=args.top_p, use_beam_search=args.beam_search, temperature=args.temperature)
response = llm.generate(prompts, sampling_params, use_tqdm=True)
torch.save(response, file_name)
