import gc
import logging
import random
import numpy as np
import os

import torch
from tqdm import tqdm, trange
from vllm import SamplingParams

from uncertainty_quantification.model_utils import configure_model
from uncertainty_quantification.semantic_unceratinty import get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
from uncertainty_quantification.io_utils import StoreManager




class ForwardManager:
    def __init__(self, args, ckpt_freq=100, temp_dir=None, accumulation_batch_size=None):
        self.args = args
        self.model = args.model
        self.chat_template_path = args.chat_template_path
        self.llm = None
        self.logit_processor = None
        # to avoid oom and to save timely, we save the model every ckpt_freq iterations
        self.ckpt_freq = ckpt_freq
        if accumulation_batch_size is not None:
            self.accumulation_batch_size = accumulation_batch_size
        else:
            self.accumulation_batch_size = 35 * ckpt_freq
        temp_dir = os.path.join(args.output_root_dir, "temp") if temp_dir is None else temp_dir
        self.store = StoreManager(temp_dir=temp_dir)

    def setup_model(self, max_num_seqs=None, gpu_memory_utilization=None, enforce_eager=False):
        self.llm, self.logit_processor = configure_model(
            self.args,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager
        )

    def forward(self, prompts, ckpt_name, sampling_params=None, prompts_as_token_ids=False, **kwargs):
        response = []
        if os.path.exists(ckpt_name):
            try:
                response = self.store.load(ckpt_name)
            except Exception as e:
                print("Error loading ckpt file: {} (error: {}), we have to recreate it".format(ckpt_name, e))
        start = len(response)
        if start == len(prompts):
            # already complete, even no need to initialize the model (lazy loading)
            return response
        if self.llm is None:
            self.setup_model(
                max_num_seqs=kwargs.get('max_num_seqs', None),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', None),
                enforce_eager=kwargs.get('enforce_eager', False)
            )
        if sampling_params is None:
            if abs(self.args.top_p - 1.0) < 1e-6 and self.args.min_p > 1e-6:
                sampling_params = SamplingParams(n=self.args.sample_counts, max_tokens=self.args.max_tokens,
                                                 logprobs=self.args.log_probs,
                                                 top_k=self.args.top_k if hasattr(self.args, "top_k") else -1,
                                                 temperature=self.args.temperature, min_p=self.args.min_p,
                                                 prompt_logprobs=self.args.log_probs if not hasattr(self.args, "prompt_log_probs") else self.args.prompt_log_probs,
                                                 min_tokens=self.args.min_tokens if hasattr(self.args, "min_tokens") else 0,
                                                 logits_processors=[self.logit_processor])
            else:
                sampling_params = SamplingParams(n=self.args.sample_counts, max_tokens=self.args.max_tokens,
                                                 logprobs=self.args.log_probs,
                                                 top_k=self.args.top_k if hasattr(self.args, "top_k") else -1,
                                                 temperature=self.args.temperature, top_p=self.args.top_p,
                                                 min_tokens=self.args.min_tokens if hasattr(self.args, "min_tokens") else 0,
                                                 prompt_logprobs=self.args.log_probs if not hasattr(self.args, "prompt_log_probs") else self.args.prompt_log_probs,
                                                 )

        for i in trange(start, len(prompts), self.ckpt_freq):
            _prompts = prompts[i:i + self.ckpt_freq]
            if prompts_as_token_ids:
                print("max_length: ", max([len(x) for x in _prompts]))
                _response = self.llm.generate(prompt_token_ids=_prompts, sampling_params=sampling_params, use_tqdm=True)
            else:
                _response = self.llm.generate(_prompts, sampling_params=sampling_params, use_tqdm=True)
            response.extend(_response)
            if i > 0 and (i - start) % self.accumulation_batch_size == 0:
                print("Starting to save the model at iteration: {}".format(i))
                # Use async save for better performance
                done_event = self.store.save(response, ckpt_name, async_write=True)
                if done_event:
                    done_event.wait()  # Wait for save to complete
                print("Saved {}/{} ({:.2f}) responses to {}".format(len(response), len(prompts), len(response) / len(prompts), ckpt_name))
        return response

    def get_spectrum_filename(self, output_filename):
        return output_filename.replace(".pt", ".pt.update_full_spectrum")

    def get_semantic_entropy_filename(self, output_filename):
        return output_filename + ".semantic_entropy"

    def check_logprobs_none(self, output_token_ids, logprobs):
        for logprob_i, logprob in enumerate(logprobs):
            if logprob is None:
                return True
            if None in logprob:
                return True
            if logprob[output_token_ids[logprob_i]] is None:
                return True
        return False

    def fillin_logits_routine(self, response, output_filename, max_num_seqs=None, gpu_memory_utilization=None,
                              try_reuse_existing_logprobs=True, check_prompt_probs=False, clean_mode=False,
                              update_filename=None, patch_filename=None, not_remove_patch_after_spectrum=False):
        # sometimes, it is possible that vllm returned none logits (when you apply some logit processors), which will break our pipeline
        # a naive solution is to re-run the forward pass
        original_prompt_token_ids = []
        continuation_token_ids = []
        example_output_flag = True
        # for llama-3, it seems computing prompt logprobs waste too much memory and would cause OOM -- see vllm/vllm/sampler.py
        # there are lots of useless computations that can take up a lot of GPU memory, e.g., log_probs, logits_sorted, etc.
        # it would be better if we can just reuse the existing response to update
        # related issues: https://github.com/vllm-project/vllm/pull/5355
        # https://github.com/vllm-project/vllm/issues/5907
        flag_reusing_trial = try_reuse_existing_logprobs
        for _response in tqdm(response):
            outputs = _response.outputs
            prompt_token_ids = _response.prompt_token_ids
            if check_prompt_probs:
                prompt_logprobs = _response.prompt_logprobs
                flag_reusing_trial = flag_reusing_trial and (
                    not self.check_logprobs_none(prompt_token_ids, prompt_logprobs))
            for output in outputs:
                output_token_ids = output.token_ids
                if example_output_flag:
                    print("Example Output: {}".format(output.text))
                    example_output_flag = False
                # starting from vllm 0.5, it seems all token ids are as of tuple type, which means they can no longer be concatenated using the + operator directly
                original_prompt_token_ids.append(list(prompt_token_ids) + list(output_token_ids))
                if try_reuse_existing_logprobs and flag_reusing_trial:
                    logprobs = output.logprobs
                    flag_reusing_trial = not self.check_logprobs_none(output_token_ids, logprobs)
                continuation_token_ids.append(output_token_ids)
        average_length = np.mean([len(x) for x in original_prompt_token_ids])
        print("Average Token (prompt + output) Limits", average_length)
        max_length = np.max([len(x) for x in original_prompt_token_ids])
        print("Max Token (prompt + output) Limits", max_length)
        update_filename = self.get_spectrum_filename(output_filename) if update_filename is None else update_filename
        exists_update_flag = False
        if os.path.exists(update_filename):
            try:
                already_update_response = self.store.load(update_filename, wait_for_pending=False)
                # assert len(already_update_response) == len(
                #     original_prompt_token_ids), "Update Length Mismatch: {} vs {}".format(
                #     len(already_update_response), len(original_prompt_token_ids))
                assert len(already_update_response) == len(
                    response), "Update Length Mismatch: {} vs {}".format(
                    len(already_update_response), len(response))
                exists_update_flag = True
                print("File Exists: {}".format(update_filename))
                del already_update_response
                gc.collect()
                if not clean_mode:
                    exit()
                else:
                    # check whether update_file is not a symlink, if it is and flag_reusing_trial is True, we should delete it to save disk space
                    if not os.path.islink(update_filename) and flag_reusing_trial:
                        print("Update File is not a symlink and we can reuse the existing logprobs, we should delete it to save disk space")
                        os.remove(update_filename)
                        print("Deleted: {}".format(update_filename))
            except Exception as e:
                print("Error loading update file: {}, we have to recreate it".format(e))
        if flag_reusing_trial:
            print("Congrats! We can reuse the existing logprobs")
            # use symlink to save disk space
            if not os.path.exists(os.path.abspath(update_filename)):
                os.symlink(os.path.abspath(output_filename), os.path.abspath(update_filename))
            else:
                assert os.path.islink(update_filename), "File exists but not a symlink: {}".format(update_filename)
                print("Symlink already exists: {} -> {}".format(os.path.abspath(update_filename), os.path.abspath(output_filename)))
            print("Created Symlink: {} -> {}".format(os.path.abspath(output_filename), os.path.abspath(update_filename)))
        else:
            print("We have to recompute the logprobs")
            if clean_mode:
                print("Exiting as we are in clean mode")
                exit()
            update_sampling_params = SamplingParams(n=1, max_tokens=1, logprobs=1,
                                                    prompt_logprobs=self.args.log_probs, # note here we cannot use prompt_logprobs=self.args.prompt_log_probs (or essentially in this case they should be equal), because we are filling in the logits
                                                    # use_beam_search=self.args.beam_search, # vllm deprecated this param in 0.6.3.post1
                                                    temperature=self.args.temperature,
                                                    )
            if hasattr(update_sampling_params, "use_beam_search"):
                # backward compatibility
                update_sampling_params.use_beam_search = self.args.beam_search
            patch_filename = output_filename.replace(".pt", ".patch.pt") if patch_filename is None else patch_filename
            exist_patch_flag = False
            if os.path.exists(patch_filename):
                try:
                    update_response = self.store.load(patch_filename, wait_for_pending=False)
                    assert len(update_response) == len(original_prompt_token_ids), "Patch Length Mismatch (possibly previous job not properly finished): {} vs {}".format(
                        len(update_response), len(original_prompt_token_ids))
                    exist_patch_flag = True
                except Exception as e:
                    print("Error loading patch file: {}, we have to recreate it".format(e))
            if not exist_patch_flag:
                update_response = self.forward(original_prompt_token_ids, patch_filename, sampling_params=update_sampling_params,
                             prompts_as_token_ids=True, max_num_seqs=max_num_seqs,
                             gpu_memory_utilization=gpu_memory_utilization)
                self.store.save(update_response, patch_filename, async_write=False)
                print("Saved Patch Response to {}".format(patch_filename))
            if not exists_update_flag:
                counter = 0
                for response_i in tqdm(range(len(response)), desc="Updating Response"):
                    outputs = response[response_i].outputs
                    prompt_token_ids = response[response_i].prompt_token_ids
                    output_num = len(outputs)
                    for output_j in range(output_num):
                        output_token_ids = outputs[output_j].token_ids
                        # starting from vllm 0.5, it seems all token ids are as of tuple type, which means they can no longer be concatenated using the + operator directly
                        assert update_response[counter].prompt_token_ids == list(prompt_token_ids) + list(
                            output_token_ids), "Prompt Token IDs Mismatch"
                        try:
                            update_prompt_logprobs = update_response[counter].prompt_logprobs[len(prompt_token_ids):]
                        except Exception as e:
                            print("Error in update_response[{}].prompt_logprobs: {} for model {}".format(counter, e,
                                                                                                         self.model))
                            print("update_response[{}].prompt_logprobs: {}".format(counter, update_response[
                                counter].prompt_logprobs))
                            print("Prompt Token ID length: {}".format(len(update_response[counter].prompt_token_ids)))
                            print("prompt_token_ids: {}".format(len(prompt_token_ids)))
                            print("output_token_ids: {}".format(len(output_token_ids)))
                            exit()
                        assert len(response[response_i].outputs[output_j].logprobs) == len(
                            update_prompt_logprobs), "Logprobs Length Mismatch: {} vs {}".format(
                            len(response[response_i].outputs[output_j].logprobs), len(update_prompt_logprobs))
                        response[response_i].outputs[output_j].logprobs = update_prompt_logprobs
                        counter += 1

            self.store.save(response, update_filename, async_write=False)
            print("Saved Updated Response to {}".format(update_filename))
            if not not_remove_patch_after_spectrum:
                os.remove(patch_filename)
            print("Deleted Patch File: {}".format(patch_filename))

    def semantic_uncertainty_computation(self, responses, ckpt_name, num_trials=1, sample_count=10, need_seed=False):
        # semantic uncertainty ICLR 2023 paper: https://arxiv.org/pdf/2302.09664
        # according to semantic uncertainty most recent nature paper (https://www.nature.com/articles/s41586-024-07421-0#MOESM1)
        # specifically appendix note 4: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07421-0/MediaObjects/41586_2024_7421_MOESM1_ESM.pdf
        # sample count (M=10) is sufficient to estimate the uncertainty
        # in original ICLR 2023 paper, they say M < 20 (appendix B: finally they decide to use M=10)
        # they also show deberta model can work well (Nature Appendix Supplementary Table 2, human-AI inter agreement on entailment), but clearly deberta model could be bad at long-context
        # also I personally do not bother to write separate code for deberta model (embedding-based inference) XD
        # so let's use prompting-based solution!
        # responses: vllm responses (should be a list), num_trials: number of trials to compute semantic entropy, sample_count (M in the papers above)
        # step 1: prepare the prompts and run the prompting
        trial_entropies = []
        # usually seeding process is handled by outside, but we can also handle it here
        if need_seed:
            random.seed(42)
        for trial in range(num_trials):
            semantic_entropies = []
            prompts = []
            ids = []
            trial_outputs = []
            trial_logits = []
            trial_lengths = []
            # to make it batchifiable, we can just compute pairwise entailment and later do aggregation
            for response_i, response in enumerate(responses):
                outputs = [output.text for output in response.outputs]
                cumulative_logprobs = [output.cumulative_logprob for output in response.outputs]
                token_ids_length = [len(output.token_ids) for output in response.outputs]
                sampled_ids = random.sample(range(len(outputs)), min(sample_count, len(outputs)))
                sampled_outputs = [outputs[sampled_id] for sampled_id in sampled_ids]
                sampled_logprobs = [cumulative_logprobs[sampled_id] for sampled_id in sampled_ids]
                sampled_token_ids_length = [token_ids_length[sampled_id] for sampled_id in sampled_ids]
                trial_outputs.append(sampled_outputs)
                trial_logits.append(sampled_logprobs)
                trial_lengths.append(sampled_token_ids_length)
                prompt = response.prompt
                # from https://www.nature.com/articles/s41586-024-07421-0#MOESM1 ("Entailment Estimator")
                # reference implementation: https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L139
                for output_i in range(len(sampled_outputs)):
                    for output_j in range(output_i + 1, len(sampled_outputs)):
                        equivalent_prompt =  """We are evaluating continuation to the prompt \"{prompt}\"\nHere are two possible continuations:\nPossible Continuation 1: {text1}\nPossible Continuation 2: {text2}\nDoes Possible Continuation 1 semantically entail Possible Continuation 2? Respond with entailment, contradiction, or neutral.\nResponse:"""
                        entailment_prompt = equivalent_prompt.format(prompt=prompt, text1=sampled_outputs[output_i], text2=sampled_outputs[output_j])
                        entailment_prompt_bidirection = equivalent_prompt.format(prompt=prompt, text2=sampled_outputs[output_i], text1=sampled_outputs[output_j])
                        prompts.extend([entailment_prompt, entailment_prompt_bidirection])
                        ids.extend([(response_i, output_i, output_j), (response_i, output_j, output_i)])
            # https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L75
            sampling_params = SamplingParams(n=1, max_tokens=30, logprobs=1,
                                             use_beam_search=self.args.beam_search,
                                             temperature=0.02)
            responses = self.forward(prompts, ckpt_name+".trial.{}".format(trial), sampling_params=sampling_params)
            # step 2: re-distribute the prompts and compute semantic entropy
            for response_i, response in enumerate(responses):
                response_positions_with_same_id = [xid for xid, x in enumerate(ids) if x[0] == response_i]
                group_responses = [response.outputs[xid].text for xid in response_positions_with_same_id]
                group_ids = [(x[1], x[2]) for x in ids if x[0] == response_i]
                string_list = trial_outputs[response_i]
                semantic_ids = get_semantic_ids(string_list, group_responses, group_ids)
                group_logits = trial_logits[response_i]
                group_lengths = trial_lengths[response_i]
                log_liks_agg = [group_logits[o_i] / group_lengths[o_i] for o_i in range(len(group_logits))]
                log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
                trial_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
                semantic_entropies.append(trial_entropy)
            trial_entropies.append(semantic_entropies)
        return np.mean(np.array(trial_entropies), axis=0)

