import argparse
import gc
import glob

from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams
from nltk.tokenize import sent_tokenize, word_tokenize

from uncertainty_quantification.tokenizer_utils import setup_tokenizer, format_prompt
from uncertainty_quantification.model_utils import configure_model
# from language_modeling.data_utils import get_data_from_huggingface, roles
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NewsTaggerArgsParsing.')
    # the values here are default values only, can be changed in the command line
    parser = step1_forward_args(parser, sample_counts=50, max_tokens=1024, log_probs=1, min_p=0.1, top_p=1.0, temperature=1.0, output_root_dir="response_news_tagger")
    parser.add_argument("--constraint_level", type=int, default=0, help="constraint level (in #(words) * multiplier/#(sentences))")
    parser.add_argument("--source_model", type=str, help="source model")
    parser.add_argument("--source_root_dir", type=str, help="source model")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    model = args.model
    chat_template_path = args.chat_template_path

    # file_name = "{}_response_n_{}_max_tokens_{}_log_probs_{}_min_p_{}_top_p_{}_seed{}.pt".format(
    #     os.path.basename(model),
    #     args.sample_counts,
    #     args.max_tokens, args.log_probs,
    #     args.min_p, args.top_p, args.seed,)
        # '_input_file_{}'.format(os.path.basename(args.input_file)) if args.input_file is not None else "")
    # file_name = os.path.join(output_root_dir, file_name)
    input_filenames = glob.glob(os.path.join(args.source_root_dir, "{}_response*update_full_spectrum".format(os.path.basename(args.source_model))))
    assert len(input_filenames)
    tokenizer = setup_tokenizer(model, chat_template_path)
    manager = ForwardManager(args, ckpt_freq=1024)
    for input_filename in input_filenames:
        print("Processing: {}".format(input_filename))
        output_filename = os.path.join(output_root_dir, os.path.basename(input_filename).replace(".pt.update_full_spectrum", ".tags"))
        metadata_filename = output_filename.replace(".tags", ".metadata")
        if os.path.exists(metadata_filename):
            prompts, all_original_outputs, all_original_prompts, args = torch.load(metadata_filename)
        else:
            prompts = []
            original_data = torch.load(input_filename)
            all_original_outputs = []
            all_original_prompts = []
            for response in original_data:
                original_prompt = response.prompt
                start_id = len(all_original_outputs)
                for output in response.outputs:
                    all_original_outputs.append(original_prompt + output.text)
                end_id = len(all_original_outputs)
                all_original_prompts.append((original_prompt, start_id, end_id))

            for text in all_original_outputs:
                system_message = ("You are a news tagger. Please tag the following news article. "
                                  "Your News Tags should start with 'News Tags:' and be followed by a list of tags separated by commas. "
                                  "The Tag list should be finished with a period. "
                                  "For example, 'News Tags: politics, sports, entertainment.'. ")
                prompts.append(format_prompt(model, text, tokenizer, system_message))
            print("Example prompt: {}".format(prompts[0]))
            torch.save((prompts, all_original_outputs, all_original_prompts, args), metadata_filename)

        maybe_exist_flag = False
        response = None
        # step-1: get original generated story
        if os.path.exists(output_filename):
            print("File exists: {}".format(output_filename))
            try:
                response = torch.load(output_filename)
                assert len(response) == len(prompts), "length mismatch"
                maybe_exist_flag = True
            except Exception as e:
                print("File exists but cannot be loaded: {} (Exception: {})".format(output_filename, e))
                # if response is already loaded, delete it to save memory as forward manager would load again
                if response is not None:
                    del response
                    gc.collect()
                maybe_exist_flag = False
        if not maybe_exist_flag:
            gpu_memory_utilization = compute_gpu_memory_utilization(model)
            response = manager.forward(prompts, output_filename, max_num_seqs=128, gpu_memory_utilization=gpu_memory_utilization)
            torch.save(response, output_filename)
            print("Example response: {}".format(response[0].outputs[0].text))
            print("Output saved to: {}".format(output_filename))
            if "llama-3" in model.lower():
                print("Llama-3: have to restart to avoid weird problems from vllm hanging (0.4.3 and 0.5.4), please relaunch the script")
                exit()
