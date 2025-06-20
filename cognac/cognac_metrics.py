# code from cognac_origin official repo: https://github.com/princeton-nlp/Cognac/tree/main
# refactored a bit to fit the current codebase
"""
Metrics.
"""
import re
from collections import defaultdict

import spacy
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

# from src.guidance_utils import remove_parenthesis

spacy_parser = spacy.load("en_core_web_lg")
def remove_parenthesis(s):
    m = re.search(r'(.*) \(.*\)', s)
    if m is None:
        return s
    else:
        return m.group(1)

def aggregate_metrics(metrics, name):
    """
    Aggregate metrics for one run.
    """
    run_stats = defaultdict(list)
    run_stats['name'] = name

    for metric in metrics:
        for k, v in metric.items():
            run_stats[k].append(v)
    num_datapoints = []
    for k, vs in run_stats.items():
        if isinstance(vs, list):
            run_stats[k] = sum(vs) / len(vs)
            num_datapoints.append(len(vs))
    #    assert not num_datapoints or len(set(num_datapoints)) == 1, num_datapoints
    run_stats['num_datapoints'] = num_datapoints[0] if num_datapoints else 0

    return dict(run_stats)


def compute_prediction_metrics(prediction, hierarchy, data_mode, multi_constraints_eval=False):
    datapoint = prediction['datapoint']
    context = datapoint['context']
    topic = datapoint['topic']
    constraint = datapoint['constraint']
    generated_text = prediction['generated_text']

    if data_mode == 'wordnet':
        generated_text = generated_text.lower()
        if not multi_constraints_eval:
            forbidden_words = hierarchy[constraint] + [constraint]

            topical_words = hierarchy[topic]
        else:
            multi_constraints = datapoint['multi_constraints']
            forbidden_words = datapoint['all_constrained_nodes']
            remaining_topical_words = datapoint['remaining_topical_words']
            remaining_nodes = datapoint['remaining_nodes']
            # assert (set(multi_constraints).issubset(set(forbidden_words))
            #         and set(remaining_topical_words).issubset(set(remaining_nodes))
            #         and set(remaining_nodes).isdisjoint(set(forbidden_words)))
            # split the above assert into three separate asserts, print the values of the sets when the assert fails
            assert set(multi_constraints).issubset(set(forbidden_words)), "multi_constraints not subset of forbidden_words:\nmulti_constraints: {}\nforbidden_words: {}".format(multi_constraints, forbidden_words)
            assert set(remaining_topical_words).issubset(set(remaining_nodes)), "remaining_topical_words not subset of remaining_nodes:\nremaining_topical_words: {}\nremaining_nodes: {}".format(remaining_topical_words, remaining_nodes)
            # assert set(remaining_nodes).isdisjoint(set(forbidden_words)), "remaining_nodes not disjoint with forbidden_words:\nremaining_nodes: {}\nforbidden_words: {}".format(remaining_nodes, forbidden_words)
            topical_words = list(set(remaining_nodes) | set(forbidden_words))
            # alternatively, we can only consider the direct children nodes
            # topical_words = list(set(multi_constraints) | set(remaining_topical_words))

        forbidden_words = set(
            list(forbidden_words) + [w + 's' for w in forbidden_words]
        )
        topical_words = set(list(topical_words) + [w + 's' for w in topical_words])

        violated = any(forbidden_word in generated_text for forbidden_word in forbidden_words)
        on_topic = any(topical_word in generated_text for topical_word in topical_words)

        topical_word_regex = '|'.join(list(topical_words))
        pattern = re.compile(rf'({topical_word_regex})')
        topical_word_matches = pattern.findall(generated_text)

        # if len(set(topical_word_matches) & set(prediction['datapoint']['current'])) > 0:
        #    on_topic = False
        # print(prediction['datapoint']['current'])
        # print(topical_word_matches)
        # print(topic)
        # print(constraint)
        # print('---')

        extracted = None
    elif data_mode == 'wikidata':

        gen_text_parsed = spacy_parser(generated_text)
        parsed_names = set([
            ent.text.lower() for ent in gen_text_parsed.ents
            if ent.label_ == 'PERSON'
        ])
        forbidden_words = hierarchy[constraint]
        forbidden_words = [remove_parenthesis(q_title) for q_id, q_title in forbidden_words]
        violated = False
        violated_word = None
        for w in forbidden_words:
            if w in generated_text:
                violated = True
                violated_word = w
                break
        # violated = len(parsed_names & forbidden_words) > 0

        topical_words = hierarchy[topic]
        topical_words = [remove_parenthesis(q_title) for q_id, q_title in topical_words]
        on_topic = False
        on_topic_word = None
        for w in topical_words:
            if w in generated_text:
                on_topic = True
                on_topic_word = w
                break
        # on_topic = len(parsed_names & topical_words) > 0
        # extracted = parsed_names
        extracted = dict(violated_word=violated_word, on_topic_word=on_topic_word)
    else:
        raise ValueError(f'Data mode {data_mode} not recognized.')
    on_topic_not_violated = (not violated) and on_topic
    # copying_bleu_score = copying_bleu(context, generated_text)
    # repetition_scores = get_repetition_scores(generated_text.split())
    prediction_metric = dict(
        violated=violated,
        on_topic=on_topic,
        on_topic_not_violated=on_topic_not_violated,
        # copying_bleu_score=copying_bleu_score,
    )
    # prediction_metric = {**prediction_metric, **repetition_scores}

    return dict(
        prediction_metric=prediction_metric,
        extracted=extracted,
    )


def copying_bleu(context, generated_text):
    prediction = word_tokenize(generated_text.strip('==').strip())
    max_score = float('-inf')
    for context_sent in context.strip().split('\n'):
        references = [word_tokenize(context_sent.strip('==').strip())]
        score = sentence_bleu(references, prediction)
        if score > max_score:
            max_score = score
    return score


def get_repetition_scores(tokens):
    metric = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(tokens, n)]
        unique_ngs = set(ngs)
        if not ngs:
            metric[f'seq-rep-{n}'] = 0.0
        else:
            metric[f'seq-rep-{n}'] = 1.0 - (len(unique_ngs) / len(ngs))
    return dict(metric)