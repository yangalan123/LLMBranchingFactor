# from storytelling.consts import plot
from storytelling.ttcw.data_utils import load_all_plots
import csv
import argparse
import os
from vllm import LLM, SamplingParams
from uncertainty_quantification.logit_processor import eta_truncation_logit_processor
from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.model_utils import configure_model
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StoryWritingArgsParsing.')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--sample_counts', type=int, help='sample counts', default=50)
    parser.add_argument('--max_tokens', type=int, help='max tokens', default=1024)
    parser.add_argument('--log_probs', type=int, help='log probs', default=100)
    parser.add_argument('--chat_template_path', type=str, help='chat template path', default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--beam_search", action="store_true", help="use beam search for decoding")
    parser.add_argument("--min_p", type=float, default=0.1, help="minimum p value for truncation sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value for nucleus sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--output_root_dir", type=str, help='output root dir',
                        default="generation_storywriting")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)
    model = args.model
    chat_template_path = args.chat_template_path
    file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}.pt".format(
        os.path.basename(model),
        args.sample_counts,
        args.max_tokens, args.log_probs,
        args.min_p, args.top_p, args.seed)
    file_name = os.path.join(output_root_dir, file_name)
    # for now, let's just do single plot
    # plots = [plot, ]
    plots, plot_ids = load_all_plots()
    word_count = 512
    if os.path.exists(file_name):
        print("File exists: {}".format(file_name))
        response = torch.load(file_name)
        llm = None
    else:
        tokenizer = setup_tokenizer(model, chat_template_path)
        metadata_filename = file_name.replace(".pt", ".metadata")
        if os.path.exists(metadata_filename):
            prompts, plots, args = torch.load(metadata_filename)
        else:
            prompts = []
            for _plot in plots:
                task_prompt = f"Write a New Yorker-style story given the plot below. Make sure it is at least {word_count} words. Directly start with the story, do not say things like ‘Here’s the story [...]'\n\nPlot: {_plot}\n\nStory: "
                prompt = format_prompt(model, task_prompt, tokenizer, system_message="Write a story based on the given plot.")
                prompts.append(prompt)
            torch.save((prompts, plots, args), metadata_filename)
        llm, logit_processor = configure_model(args)
        if abs(args.top_p - 1.0) < 1e-3:
            sampling_params = SamplingParams(n=args.sample_counts, max_tokens=args.max_tokens, logprobs=args.log_probs,
                                             temperature=args.temperature, min_p=args.min_p,
                                             prompt_logprobs=args.log_probs, logits_processors=[logit_processor])
        else:
            sampling_params = SamplingParams(n=args.sample_counts, max_tokens=args.max_tokens, logprobs=args.log_probs,
                                             temperature=args.temperature, top_p=args.top_p,
                                             prompt_logprobs=args.log_probs)
        response = llm.generate(prompts, sampling_params, use_tqdm=True)
        torch.save(response, file_name)

    file_text_version_name = file_name.replace(".pt", ".csv")
    with open(file_text_version_name, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response", "id"])
        writer.writeheader()
        for resp_i, resp in enumerate(response):
            for output_i, outputs in enumerate(resp.outputs):
                writer.writerow({"prompt": resp.prompt, "response": outputs.text, "id": "plot_{}_output_{}".format(plot_ids[resp_i], output_i)})
