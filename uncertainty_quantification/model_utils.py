from vllm import LLM, SamplingParams
import torch
import os
from uncertainty_quantification.logit_processor import eta_truncation_logit_processor
def configure_model(args, max_num_seqs=None, gpu_memory_utilization=None, enforce_eager=False, cpu_offload_gb=0):
    vllm_args = dict()
    vllm_args['model'] = args.model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    vllm_args['tensor_parallel_size'] = len(available_gpus)
    if not hasattr(args, "disable_chunked_prefill") or not args.disable_chunked_prefill:
        vllm_args['enable_chunked_prefill'] = True
    else:
        vllm_args['enable_chunked_prefill'] = False
    if not hasattr(args, "max_num_batched_tokens") and vllm_args['enable_chunked_prefill']:
        vllm_args['max_num_batched_tokens'] = 256

    if "gpt2-xl" in vllm_args['model']:
        # ValueError: Total number of attention heads (25) must be divisible by tensor parallel size (4).
        vllm_args['tensor_parallel_size'] = 1
    # default setting
    # if gpu_memory_utilization is None:
    #     gpu_memory_utilization = 0.9
    if max_num_seqs is None:
        # max_num_seqs = 256
        vllm_args['max_num_seqs'] = 256
    vllm_args['seed'] = args.seed
    vllm_args['gpu_memory_utilization'] = gpu_memory_utilization
    vllm_args['enforce_eager'] = enforce_eager
    vllm_args['cpu_offload_gb'] = cpu_offload_gb
    vllm_args['max_logprobs'] = args.log_probs
    llm = LLM(**vllm_args)

    logit_processor = lambda token_ids, log_probs: eta_truncation_logit_processor(token_ids, log_probs, eta=args.min_p)
    return llm, logit_processor

def compute_gpu_memory_utilization(model):
    # after a lof of debugging, I finally realized that gpu_memory_utilization should not be set to be too large -- it is basically model_weight + kv cache + cuda graph
    # so if it is too large, it would be oom as prompt_logprobs memory is not reserved in advance
    # see https://github.com/vllm-project/vllm/issues/5907
    # let's do some dirty hack for now
    gpu_memory_utilization = 0.5
    # we are assuming a100 * 4 here, so 80g * 4 * 0.5 = 160g, should be large enough even for 70B model (roughly 140g)
    # if a40 * 4, then 40g * 4 * 0.5 = 80g, should be large enough for 13B model (but not enough for 70B model)
    # if "llama-3" in model.lower():
    #     if "70b" not in model.lower():
    #         gpu_memory_utilization = 0.4
    # device_count = torch.cuda.device_count()
    # memory_size_per_device = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    # total_memory_size = device_count * memory_size_per_device
    # large_model_flags = ["70b", "8x22", "72b"]
    # if any([x in model.lower() for x in large_model_flags]):
    #     if gpu_memory_utilization * total_memory_size <= 140:
    #         # a40 case
    #         gpu_memory_utilization = 0.9
    return gpu_memory_utilization

def get_max_num_seq_and_gpu_util_for_logits_compute(model):
    if "70b" in model.lower() and "llama-3" in model.lower():
        gpu_memory_utilization = 0.55
        max_num_seqs = 128
    if "8b" in model.lower():
        gpu_memory_utilization = 0.3
        max_num_seqs = 128  # a100
    return max_num_seqs, gpu_memory_utilization
