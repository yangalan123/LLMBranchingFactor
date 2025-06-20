import argparse
def step1_forward_args(parser: argparse.ArgumentParser, **kwargs):
    # to accomodate the difference in default arg settings for different tasks, we introduce kwargs to allow setting default values
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--sample_counts', type=int, help='sample counts', default=kwargs.get('sample_counts', 50))
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=kwargs.get('max_tokens', 512))
    parser.add_argument('--log_probs', type=int, help='log probs', default=kwargs.get('log_probs', 50))
    parser.add_argument('--prompt_log_probs', type=int, help='prompt log probs', default=kwargs.get('prompt_log_probs', 50))
    parser.add_argument("--max_num_batched_tokens", type=int, default=kwargs.get("max_num_batched_tokens", 256), help="max num batched tokens")
    parser.add_argument("--disable_chunked_prefill", action="store_true", help="enable chunked prefill")
    parser.add_argument('--chat_template_path', type=str, help='chat template path', default=kwargs.get('chat_template_path', None))
    parser.add_argument('--output_root_dir', type=str, help='output root dir', default=kwargs.get('output_root_dir', "cognac_ctrl_outputs"))
    parser.add_argument("--seed", type=int, default=kwargs.get("seed", 42), help="random seed for initialization")
    parser.add_argument("--min_p", type=float, default=kwargs.get("min_p", 0.1), help="minimum p value for sampling")
    parser.add_argument("--top_p", type=float, default=kwargs.get("top_p", 1.0), help="minimum p value for sampling")
    parser.add_argument("--top_k", type=int, default=kwargs.get("top_k", -1), help="top-K tokens for sampling")
    parser.add_argument("--beam_search", action="store_true", help="use beam search for decoding")
    parser.add_argument("--temperature", type=float, default=kwargs.get("temperature", 1.0), help="temperature for sampling")
    parser.add_argument("--ckpt_freq", type=int, default=kwargs.get("ckpt_freq", 128), help="checkpoint frequency")
    return parser
