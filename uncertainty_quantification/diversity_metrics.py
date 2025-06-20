from nltk import ngrams
from uncertainty_quantification.consts import tolerance_inf
def distinct_n(token_sets):
    """Compute the distinct-n metric for a list of tokens"""
    ns = [1, 2, 3, 4]
    all_n_grams = {
        n: [] for n in ns
    }
    for tokens in token_sets:
        for n in ns:
            all_n_grams[n].extend(list(ngrams(tokens, n)))
    distinct_n_stats = {
        n: len(set(all_n_grams[n])) / len(all_n_grams[n] + tolerance_inf)
        for n in ns
    }
    return distinct_n_stats

# codes adapted from https://github.com/GuyTevet/diversity-eval/blob/master/diversity_metrics.py
# and https://github.com/YanzhuGuo/llm-diversity/tree/main/diversity_metrics
def compute_diversity_metrics(token_sets):
    """Compute diversity metrics for a list of tokens"""
    distinct_n_stats = distinct_n(token_sets)
    return {
        "distinct_1": distinct_n_stats[1],
        "distinct_2": distinct_n_stats[2],
        "distinct_3": distinct_n_stats[3],
        "distinct_4": distinct_n_stats[4],
    }