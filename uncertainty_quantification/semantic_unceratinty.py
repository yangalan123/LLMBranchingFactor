import torch
import numpy as np


# https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L169
def get_semantic_ids(strings_list, group_responses, group_ids):
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    idpair_to_response_id = dict()
    for id_pair_i, id_pair in group_ids:
        idpair_to_response_id[id_pair] = id_pair_i
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                response_1 = group_responses[idpair_to_response_id[(i, j)]]
                response_2 = group_responses[idpair_to_response_id[(j, i)]]
                if are_equivalent(response_1, response_2):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


# https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L208
def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            # original implementation (below) is not numerically stable, so let's use logsumexp
            # log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            log_lik_norm = torch.tensor(id_log_likelihoods) - torch.logsumexp(torch.tensor(log_likelihoods), dim=0)
            # logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
            logsumexp_value = torch.logsumexp(log_lik_norm, dim=0)
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


# https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L244
def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def check_implication(response_text):
    # https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L75
    binary_response = response_text.lower()[:30]
    if 'entailment' in binary_response:
        return 2
    elif 'neutral' in binary_response:
        return 1
    elif 'contradiction' in binary_response:
        return 0
    else:
        # return "manual neutral"
        return 1


def are_equivalent(response_1, response_2, strict_entailment=False):
    # https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py#L172C1-L186C39
    implication_1 = check_implication(response_1)
    implication_2 = check_implication(response_2)  # pylint: disable=arguments-out-of-order
    assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

    if strict_entailment:
        semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

    else:
        implications = [implication_1, implication_2]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent
