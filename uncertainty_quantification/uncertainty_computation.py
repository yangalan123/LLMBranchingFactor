import numpy as np
import torch

from uncertainty_quantification.common_utils import condense_arrays_with_inconsistent_lengths
from uncertainty_quantification.loglik_computation import get_token_truncated_dist_from_vllm_outputs

DEFAULT_ASYMPTOTIC_LIMIT=50

ENTROPY_PROFILE_KEY = "entropy"
LOGLIK_PROFILE_KEY = "output_per_token_logprob_truncated"


def compute_bf_values(entropies, logliks, asymptotic_limit=DEFAULT_ASYMPTOTIC_LIMIT):
    # Important Note: here entropies and logliks both have to be "Accumulated"!!
    ret = []
    assert len(entropies) == len(logliks), "different instance numbers for: {} vs {}".format(len(entropies),
                                                                                             len(logliks))
    for entropy, loglik in zip(entropies, logliks):
        if len(entropy) > asymptotic_limit:
            ret.append(-loglik[-1])
        else:
            ret.append(entropy[-1] / len(entropy))
    return ret


def accumulate_entropy_and_loglik_from_profile(distribution_profile,
                                               entropy_key=ENTROPY_PROFILE_KEY,
                                               loglik_key=LOGLIK_PROFILE_KEY):
    """Condense a distribution_profile into the "accumulated" per-prompt arrays that
    compute_bf_values expects.

    For each prompt this returns:
      * output_token_entropies: cumulative sum of the (sample-averaged) per-token
        entropy, i.e. entropy[:k].sum() at index k-1.
      * output_token_logprobs: the (sample-averaged) mean per-token loglik over the
        length-k prefix (prefix_mode + divide_by_length).

    This is the chunk that was duplicated in demo.step3_compute_bf_values, the v2
    pipeline, and the plotting helpers. The two returned lists stay index-aligned
    per prompt (one entry per profile instance), so they can be passed straight to
    compute_bf_values.
    """
    output_token_entropies = [
        np.array(condense_arrays_with_inconsistent_lengths(instance)).cumsum()
        for instance in distribution_profile[entropy_key]
    ]
    output_token_logprobs = [
        np.array(condense_arrays_with_inconsistent_lengths(
            instance, prefix_mode=True, divide_by_length=True))
        for instance in distribution_profile[loglik_key]
    ]
    return output_token_entropies, output_token_logprobs


def compute_overall_bf_from_profile(distribution_profile, asymptotic_limit=DEFAULT_ASYMPTOTIC_LIMIT,
                                    entropy_key=ENTROPY_PROFILE_KEY, loglik_key=LOGLIK_PROFILE_KEY):
    """Per-prompt BF values (exp-space) and their mean from a distribution_profile.

    Equivalent to the old demo.step3_compute_bf_values body.
    """
    output_token_entropies, output_token_logprobs = accumulate_entropy_and_loglik_from_profile(
        distribution_profile, entropy_key=entropy_key, loglik_key=loglik_key)
    bf_values = np.exp(compute_bf_values(
        output_token_entropies, output_token_logprobs, asymptotic_limit=asymptotic_limit))
    return bf_values, float(np.mean(bf_values))


def compute_bf_curve_from_profile(distribution_profile, asymptotic_limit=DEFAULT_ASYMPTOTIC_LIMIT,
                                  min_prompts_per_position=1, entropy_only=False,
                                  entropy_key=ENTROPY_PROFILE_KEY, loglik_key=LOGLIK_PROFILE_KEY):
    """BF as a function of generation position, averaged across prompts.

    At each position the per-prompt log-BF follows compute_bf_values: the mean
    per-token entropy while the prefix length is within ``asymptotic_limit``, then
    the mean per-token NLL (``-loglik``) past it. ``entropy_only=True`` (equivalently
    ``asymptotic_limit -> infinity``) keeps the slower-converging entropy estimate
    for the whole sequence, which suits stochastic tasks such as random strings.

    Returns a list of dicts: {"position", "bf", "bf_std", "n_prompts"}.
    """
    effective_limit = float("inf") if entropy_only else asymptotic_limit
    entropy_groups = distribution_profile.get(entropy_key, []) or []
    loglik_groups = distribution_profile.get(loglik_key, []) or []

    entropy_by_prompt = []
    loglik_by_prompt = []
    for idx, entropy_instance in enumerate(entropy_groups):
        if not entropy_instance:
            continue
        entropy_values = condense_arrays_with_inconsistent_lengths(entropy_instance)
        if not entropy_values:
            continue
        loglik_instance = loglik_groups[idx] if idx < len(loglik_groups) else None
        loglik_values = None
        if loglik_instance:
            condensed = condense_arrays_with_inconsistent_lengths(
                loglik_instance, prefix_mode=True, divide_by_length=True)
            if condensed:
                loglik_values = np.array(condensed)
        # The loglik branch is only needed past the asymptotic limit; when we stay in
        # the entropy regime for the whole sequence we don't require it.
        if not entropy_only and loglik_values is None:
            continue
        entropy_by_prompt.append(np.array(entropy_values).cumsum())
        loglik_by_prompt.append(loglik_values)

    curves = []
    max_len = max([len(x) for x in entropy_by_prompt] + [0])
    for pos in range(max_len):
        per_prompt_log_bf = []
        for entropies, logliks in zip(entropy_by_prompt, loglik_by_prompt):
            if len(entropies) <= pos:
                continue
            if (pos + 1) <= effective_limit:
                log_bf = entropies[pos] / (pos + 1)  # mean per-token entropy
            else:
                if logliks is None or len(logliks) <= pos:
                    continue
                log_bf = -logliks[pos]  # mean per-token NLL
            per_prompt_log_bf.append(log_bf)
        if len(per_prompt_log_bf) < min_prompts_per_position:
            break
        bf_values = np.exp(per_prompt_log_bf)
        curves.append({
            "position": pos,
            "bf": float(np.mean(bf_values)),
            "bf_std": float(np.std(bf_values)),
            "n_prompts": len(per_prompt_log_bf),
        })
    return curves


# mean-token-entropies: introduced to quantify NLG model's output uncertainty
# see: https://aclanthology.org/2020.findings-emnlp.162/
# and: https://aclanthology.org/2023.emnlp-main.611/
# ht_estimator: https://aclanthology.org/2022.acl-short.20.pdf,
# the usage of ht in entropy estimation can trace back to: https://www.stat.berkeley.edu/~binyu/ps/entropy.sub.pdf

def horvitz_thompson_weighting(probs):
    return probs / (1 - np.power((1 - probs), len(probs)) + 1e-6)


def check_token_id_not_nan_prob(token_ids, max_length, logprobs):
    for i in range(max_length):
        if np.isnan(logprobs[i][token_ids[i]].logprob):
            return False
    return True

def compute_mean_seq_entropy(entropy, offset=0, maxlen=None):
    maxlen = len(entropy) if maxlen is None else min(len(entropy), maxlen)
    return np.mean(entropy[offset:maxlen])

def from_entropy_profile_to_length_wise_entropies_and_sample_wise_seq_mean_entropies(entropies, offset=0, maxlen=None):
    length_wise_entropies = dict()
    sample_wise_sequence_mean_entropies = []
    for i, entropy in enumerate(entropies):
        maxlen = len(entropy) if maxlen is None else min(len(entropy), maxlen)
        for j in range(offset, maxlen):
            if j not in length_wise_entropies:
                length_wise_entropies[j] = []
            length_wise_entropies[j].append(entropy[j])
        sample_wise_sequence_mean_entropies.append(compute_mean_seq_entropy(entropy, offset=offset, maxlen=maxlen))

    return length_wise_entropies, sample_wise_sequence_mean_entropies

def compute_ebf_from_length_wise_entropies(length_wise_entropies, length_wise_weight=None, use_logarithm=True, offset=0):
    mean_entropy = [np.mean(length_wise_entropies[i]) for i in range(offset, len(length_wise_entropies))]
    length_wise_ppl = [np.exp(length_wise_entropies[i]) for i in range(offset, len(length_wise_entropies))]
    mean_ppl = [np.mean(length_wise_ppl[i]) for i in range(offset, len(length_wise_ppl))]
    summed_entropies = np.sum(mean_entropy)
    if length_wise_weight is None:
        length_wise_weight = dict()
        # assuming uniform weight
        # length_wise_weight = [[1.0 / len(length_wise_entropies[i]), ] * len(length_wise_entropies[i]) for i in
        #                       range(offset, len(length_wise_entropies))]
        for i in range(offset, len(length_wise_entropies)):
            length_wise_weight[i] = [1.0 / len(length_wise_entropies[i]), ] * len(length_wise_entropies[i])
    ht_entropies = []
    for i in range(offset, len(length_wise_entropies)):
        ht_weights = horvitz_thompson_weighting(np.array(length_wise_weight[i]))
        ht_entropy = np.sum(np.array(length_wise_entropies[i]) * ht_weights)
        ht_entropies.append(ht_entropy)
    ht_entropy = np.sum(ht_entropies)
    if use_logarithm:
        mc_ppl = summed_entropies
        cond_ppl_prod = np.log(np.prod(mean_ppl))
        ht_ppl = ht_entropy
    else:
        mc_ppl = np.exp(summed_entropies)
        cond_ppl_prod = np.prod(mean_ppl)
        ht_ppl = np.exp(ht_entropy)
    return {"mc_ppl": mc_ppl, "cond_ppl_prod": cond_ppl_prod, "ht_ppl": ht_ppl}

def compute_ebf_from_length_wise_and_sample_wise_entropies(length_wise_entropies, sample_wise_sequence_mean_entropies,
                                                           length_wise_weight=None, use_logarithm=True, offset=0):
    # cond_ppl_prod = np.power(np.prod(mean_ppl), 1 / len(mean_ppl))
    res = compute_ebf_from_length_wise_entropies(length_wise_entropies, length_wise_weight,
                                                                            use_logarithm, offset)
    mc_ppl = res["mc_ppl"]
    cond_ppl_prod = res["cond_ppl_prod"]
    ht_ppl = res["ht_ppl"]
    return {"mc_ppl": mc_ppl, "cond_ppl_prod": cond_ppl_prod,
            "mean_seq_entropy": np.mean(sample_wise_sequence_mean_entropies), "ht_ppl": ht_ppl, "perplexity": np.mean(np.exp(sample_wise_sequence_mean_entropies))}


def compute_ebf_from_vllm_outputs(outputs, ps, top_p_mode=False, max_length=None, use_logarithm=True):
    def _compute_ebf_from_outputs_with_p(p):
        length_wise_entropies = dict()
        length_wise_weight = dict()
        length_wise_ppl = dict()
        sample_wise_sequence_mean_entropies = []
        irregular_count = 0
        for output in outputs:
            gen_seq_len = len(output.logprobs)
            token_ids = output.token_ids
            _max_length = min(gen_seq_len, max_length) if max_length is not None else gen_seq_len
            sample_entropies = []
            loglik = 0
            tmp_token_buf = []
            if not check_token_id_not_nan_prob(token_ids, _max_length, output.logprobs):
                # contain nan on the way, ignore
                irregular_count += 1
                continue
            for length_i in range(_max_length):
                if length_i not in length_wise_entropies:
                    length_wise_entropies[length_i] = []
                    length_wise_ppl[length_i] = []
                    length_wise_weight[length_i] = []
                unnormalized_dist = output.logprobs[length_i]
                tmp_token_buf.append(list(unnormalized_dist.keys()))
                keys, normalized_dist_values = get_token_truncated_dist_from_vllm_outputs(unnormalized_dist, token_ids,
                                                                                          length_i, p, top_p_mode)
                try:
                    loglik += torch.log(normalized_dist_values[keys.index(token_ids[length_i])]).item()
                except:
                    print("length_i: ", length_i, np.exp(unnormalized_dist[token_ids[length_i]]), p)
                    print("token id:", token_ids[length_i])
                    print([x for x in range(len(tmp_token_buf)) if token_ids[length_i] in tmp_token_buf[x]])
                    print('current token ids:', token_ids[length_i - 5:])
                    print("keys: ", keys)
                    print("len-softmax:", len(normalized_dist_values))
                    exit()
                entropy = -torch.sum(normalized_dist_values * torch.log(normalized_dist_values))
                length_wise_entropies[length_i].append(entropy.item())
                length_wise_ppl[length_i].append(np.exp(entropy.item()))
                length_wise_weight[length_i].append(np.exp(loglik))
                sample_entropies.append(entropy.item())
            sample_wise_sequence_mean_entropies.append(np.mean(sample_entropies))

        if irregular_count == len(outputs):
            print("All outputs contain nan, please check the model's output")
            return {"mc_ppl": -1, "cond_ppl_prod": -1, "mean_seq_entropy": -1, "ht_ppl": -1}
        return compute_ebf_from_length_wise_and_sample_wise_entropies(length_wise_entropies,
                                                                      sample_wise_sequence_mean_entropies,
                                                                      length_wise_weight, use_logarithm)

    if type(ps) == list:
        ebf = dict()
        for _p in ps:
            ebf[_p] = _compute_ebf_from_outputs_with_p(_p)
        return ebf
    else:
        return _compute_ebf_from_outputs_with_p(ps)


def compute_ebf_from_hf_outputs(outputs, min_p, max_length=None, use_logarithm=True):
    def _compute_ebf_from_outputs_with_min_p(min_p):
        length_wise_entropies = dict()
        length_wise_ppl = dict()
        sample_wise_sequence_mean_entropies = []
        for output in outputs:
            gen_seq_len = len(output)
            _max_length = min(gen_seq_len, max_length) if max_length is not None else gen_seq_len
            sample_entropies = []
            for length_i in range(_max_length):
                if length_i not in length_wise_entropies:
                    length_wise_entropies[length_i] = []
                    length_wise_ppl[length_i] = []
                unnormalized_dist = output[length_i]
                non_inf_indices = torch.exp(unnormalized_dist) >= min_p
                filtered_unnormalized_dist = unnormalized_dist[non_inf_indices]
                normalized_dist_values = torch.softmax(filtered_unnormalized_dist, dim=0)
                entropy = -torch.sum(normalized_dist_values * torch.log(normalized_dist_values))
                length_wise_entropies[length_i].append(entropy.item())
                length_wise_ppl[length_i].append(np.exp(entropy.item()))
                sample_entropies.append(entropy.item())
            sample_wise_sequence_mean_entropies.append(np.mean(sample_entropies))
        return compute_ebf_from_length_wise_and_sample_wise_entropies(length_wise_entropies,
                                                                      sample_wise_sequence_mean_entropies,
                                                                      use_logarithm=use_logarithm)

    if type(min_p) == list:
        ebf = dict()
        for _min_p in min_p:
            ebf[_min_p] = _compute_ebf_from_outputs_with_min_p(_min_p)
        return ebf
    else:
        return _compute_ebf_from_outputs_with_min_p(min_p)


def sequence_mc_simulation_from_reconstruction_distribution(reconstruction_distribution, simulation_number=100):
    keys = list(reconstruction_distribution.keys())
    key_lengths = [reconstruction_distribution[key]['prefix_length'] for key in keys]
    max_length = max(key_lengths)
    all_entropies = []
    for simulation_i in range(simulation_number):
        key_arr = []
        loglik = 0
        entropies = []
        for length_i in range(max_length):
            cur_key = "-".join([str(x) for x in key_arr])
            distribution = reconstruction_distribution[cur_key]['renorm_dist']
            next_tokens = list(distribution.keys())
            next_tokens_prob = [distribution[key] for key in next_tokens]
            next_token = np.random.choice(next_tokens, p=next_tokens_prob)
            key_arr.append(next_token)
            entropies.append(-np.sum(next_tokens_prob * np.log(next_tokens_prob)))
            loglik += np.log(distribution[next_token])
        all_entropies.append(entropies)
    all_entropies = np.array(all_entropies)
    mean_entropies = np.mean(all_entropies, axis=0)
    estimated_complete_entropies = np.sum(mean_entropies)
    return np.exp(estimated_complete_entropies)

