import torch
import gc
import glob
import argparse
import os
from uncertainty_quantification.loglik_computation import compute_loglik
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.model_utils import compute_gpu_memory_utilization
from cognac_metrics import compute_prediction_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NewsTaggerArgsParsing.')
    # the values here are default values only, can be changed in the command line
    step1_forward_args(parser, sample_counts=1, max_tokens=1, log_probs=1, chat_template_path=None,
                       output_root_dir="cognac_responses_200_violating_constraint", seed=42, min_p=0, top_p=1.0, temperature=1.0)
    parser.add_argument("--source_root_dir", type=str, help="source model", default="cognac_responses_200")
    parser.add_argument("--source_constraint_level", type=int, default=1, help="constraint level that serves as the source")
    parser.add_argument("--target_constraint_level", type=int, default=5, help="target constraint level that will be used to examine the source")
    args = parser.parse_args()
    model = args.model
    source_root_dir = args.source_root_dir
    source_constraint_dir = os.path.join(args.source_root_dir, "application_ctrlgen_multi_constraints_{}".format(args.source_constraint_level))
    target_constraint_dir = os.path.join(args.source_root_dir, "application_ctrlgen_multi_constraints_{}".format(args.target_constraint_level))
    model_basename = os.path.basename(args.model)
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    source_filenames = glob.glob(os.path.join(source_constraint_dir, "{}_response*_top_p_0.9_*update_full_spectrum".format(os.path.basename(args.model))))
    suffix = ".from_{}_to_{}".format(args.source_constraint_level, args.target_constraint_level)
    manager = ForwardManager(args, ckpt_freq=64)
    print(source_filenames)
    #exit()

    for source_filename in source_filenames:
        print("Processing: {}".format(source_filename))
        output_filename = os.path.join(output_root_dir, os.path.basename(source_filename).replace(".pt.update_full_spectrum", suffix))
        output_metadata_filename = output_filename + ".metadata"
        source_metadata_filename = source_filename.replace(".pt.update_full_spectrum", ".metadata")
        target_metadata_filename = source_metadata_filename.replace("application_ctrlgen_multi_constraints_{}".format(args.source_constraint_level), "application_ctrlgen_multi_constraints_{}".format(args.target_constraint_level))
        if os.path.exists(output_metadata_filename):
            [prompts, all_original_prompts, all_original_logliks, all_original_metrics] = torch.load(output_metadata_filename)
            print("Loaded from: {}".format(output_metadata_filename))
        else:
            prompts = []
            source_constraint_prompts, full_source_cognac_data, _, (hierarchy, update_args) = torch.load(source_metadata_filename)
            target_constraint_prompts, full_target_cognac_data, _, _ = torch.load(target_metadata_filename)
            source_responses = torch.load(source_filename)
            target_filename = source_filename.replace("application_ctrlgen_multi_constraints_{}".format(args.source_constraint_level), "application_ctrlgen_multi_constraints_{}".format(args.target_constraint_level))
            target_responses = torch.load(target_filename)
            # original_data = torch.load(source_filename)
            # all_original_outputs = []
            all_original_prompts = []
            all_original_logliks = []
            all_original_metrics = []
            for data_i, (source_response, target_prompt) in enumerate(zip(source_responses, target_constraint_prompts)):
                # original_prompt = source_response.prompt
                start_id = len(prompts)
                # source_forbidden_words, source_topical_words = extract_forbidden_words_topical_words_wordnet(hierarchy, full_source_cognac_data[data_i], multi_constraints_eval=args.source_constraint_level > 1)
                # target_forbidden_words, target_topical_words = extract_forbidden_words_topical_words_wordnet(hierarchy, full_target_cognac_data[data_i], multi_constraints_eval=args.target_constraint_level > 1)
                source_prompt = source_response.prompt
                # prompt_logliks = compute_loglik(output.prompt_token_ids, output.prompt_logprobs, tolerance_inf=1e-12)
                assert target_prompt == target_responses[data_i].prompt, "Prompt mismatch"
                target_response = target_responses[data_i]
                target_prompt_logliks = compute_loglik(target_response.prompt_token_ids, target_response.prompt_logprobs)
                source_prompt_logliks = compute_loglik(source_response.prompt_token_ids, source_response.prompt_logprobs)
                for output in source_response.outputs:
                    _normalized_text = output.text.lower()
                    target_prediction_item = {
                        # enforce stronger constraints to compute metric
                        "datapoint": full_target_cognac_data[data_i],
                        "generated_text": _normalized_text,
                    }
                    target_prediction_metric = compute_prediction_metrics(target_prediction_item, hierarchy, "wordnet", multi_constraints_eval=args.target_constraint_level > 1)["prediction_metric"]
                    source_prediction_item = {
                        # enforce original constraints to compute metric
                        "datapoint": full_source_cognac_data[data_i],
                        "generated_text": _normalized_text,
                    }
                    source_prediction_metric = compute_prediction_metrics(source_prediction_item, hierarchy, "wordnet", multi_constraints_eval=args.source_constraint_level > 1)["prediction_metric"]
                    if target_prediction_metric['violated'] and not source_prediction_metric['violated']:
                        # compute loglik for the re-combined sequence
                        prompts.append(target_prompt + output.text)
                        output_logliks = compute_loglik(output.token_ids, output.logprobs)
                        all_original_logliks.append([target_prompt_logliks, source_prompt_logliks, output_logliks])
                        all_original_metrics.append([source_prediction_metric, target_prediction_metric])

                end_id = len(prompts)
                if end_id > start_id:
                    all_original_prompts.append((target_prompt, start_id, end_id, source_prompt))
            torch.save([prompts, all_original_prompts, all_original_logliks, all_original_metrics], output_metadata_filename)
            print("Saved to: {}".format(output_metadata_filename))

        print("Finished processing: {}".format(source_filename))
        print("Total number of prompts: {}".format(len(prompts)))
        maybe_exist_flag = False
        response = None
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
            # as we use a40 for quick run, we need to check whether the model is 70B -- if it is, we have to use 0.8
            if "70b" in model.lower():
                gpu_memory_utilization = 0.5
            if "8b" in model.lower():
                gpu_memory_utilization = 0.3
            response = manager.forward(prompts, output_filename, max_num_seqs=64, gpu_memory_utilization=gpu_memory_utilization)
            torch.save(response, output_filename)
            if "llama-3" in model.lower():
                print("Llama-3: have to restart to avoid weird problems from vllm hanging (0.4.3 and 0.5.4), please relaunch the script")
                exit()
        # step-2: get full spectrum of probability
        # manager.fillin_logits_routine(response, output_filename, max_num_seqs=4, gpu_memory_utilization=gpu_memory_utilization)



