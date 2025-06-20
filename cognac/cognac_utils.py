# code from cognac official repo: https://github.com/princeton-nlp/Cognac/tree/main
import copy
import json
import random
from argparse import Namespace
from collections import defaultdict
from queue import Queue

from cognac.cognac_consts import MAPPING
from uncertainty_quantification.tokenizer_utils import format_prompt, setup_tokenizer


def get_default_args():
    args = Namespace(
        data='wordnet',
        train_path='cognac_origin/wordnet/train.jsonl',
        dev_path='cognac_origin/wordnet/dev.jsonl',
        test_path='cognac_origin/wordnet/test.jsonl',
        hierarchy_path='cognac_origin/wordnet/topic_to_leafs.json',
        eval_version=0,
        train_num_datapoints=100000000,
        dev_num_datapoints=100000000,
        test_num_datapoints=100000000,
        multi_constraints=1,
        expansion_level_for_multi_constraint=3
    )
    return args

def get_cognac_data(model_name, chat_template_path=None, select_ids=None, update_args=None, split='train'):
    args = get_default_args()
    if update_args is not None:
        for k, v in vars(update_args).items():
            if hasattr(args, k):
                setattr(args, k, v)
    datasets, hierarchy, _ = get_data(args)
    split_data = datasets[split]
    if select_ids is None:
        select_ids = list(range(len(split_data)))
    tokenizer = setup_tokenizer(model_name, chat_template_path)
    # within the scope of our project, we only need to use one split
    # prompts = [d['context_with_instructions'] for d in split_data]
    # answers = split_data
    # sources = ['std'] * len(prompts)
    prompts, answers, sources = [], [], []
    for i in select_ids:
        prompts.append(format_prompt(model_name, split_data[i]['context_with_instructions'], tokenizer))
        answers.append(split_data[i])
        sources.append('std')
    return prompts, answers, sources, [hierarchy, args]


def normalize_datapoint(datapoint, args, hierarchy):
    if args.data == 'wordnet':
        return datapoint
    elif args.data == 'wikidata':
        normalized = dict()
        if 'example_id' in datapoint:
            normalized['id'] = datapoint['example_id']
        elif 'id' in datapoint:
            normalized['id'] = datapoint['id']

        if 'text' in datapoint:
            normalized['context'] = datapoint['text']
        elif 'context' in datapoint:
            normalized['context'] = datapoint['context']

        normalized['topic'] = (
            tuple(datapoint['p'])
            if 'topic' not in datapoint
            else datapoint['topic']
        )
        normalized['gen_qs'] = datapoint['gen_qs']
        normalized['gen_text'] = datapoint['gen_text']

        if 'constraint' in datapoint:
            normalized['constraint'] = datapoint['constraint']
        else:
            constraint_candidates = defaultdict(int)
            for gen_q in datapoint['gen_qs']:
                gen_ps = hierarchy.q_to_p[tuple(gen_q)]
                for p_name, p_values in gen_ps.items():
                    for p_value in p_values:
                        constraint_candidates[(p_name, p_value)] += 1

            SKIP = {}
            selected_constraint = None
            for constraint in constraint_candidates.keys():
                if normalized['topic'][0] != constraint[0] and constraint[0] not in SKIP:
                    selected_constraint = constraint
                    break

            normalized['constraint'] = selected_constraint
        return normalized
    else:
        raise ValueError(f'Data arg {args.data} not recognized.')


# args: data, train_path, dev_path, test_path, hierarchy_path, eval_version, train_num_datapoints, dev_num_datapoints, test_num_datapoints
def get_data(args, randomize=False, hierarchy_only=False):
    if args.data == 'wordnet':
        hierarchy = get_hierarchy(args.hierarchy_path)
    elif args.data == 'wikidata':
        # Return the modifier to update the datapoint.
        # hierarchy = get_wikidata_hierarchy()
        raise NotImplementedError("Cognac authors have not provided wikidata access yet (as of 05/20/2024)")
    else:
        raise ValueError(f'Data arg {args.data} not recognized.')

    if hierarchy_only:
        return hierarchy

    datasets = defaultdict(list)
    data_paths = {
        'train': args.train_path,
        'dev': args.dev_path,
        'test': args.test_path,
    }
    for dataset_split, dataset_path in data_paths.items():
        if dataset_path is not None:
            with open(dataset_path) as f:
                for line in f:
                    datapoint = json.loads(line.strip())
                    datapoint = normalize_datapoint(datapoint, args, hierarchy)

                    topic = datapoint['topic']
                    constraint = datapoint['constraint']
                    if (
                            args.data == 'wordnet' and
                            (
                                    constraint not in hierarchy[topic] or
                                    constraint == topic or
                                    constraint not in hierarchy
                            )
                    ):
                        # skipping bad data
                        continue

                    if args.data == 'wikidata' and constraint is None:
                        continue

                    # datasets[dataset_split].append(datapoint)
                    # for our specific purposes, we only need to use one template
                    # if args.eval_version == -1 and args.dataset_split == 'dev':
                    #     eval_version = random.choice(range(3, 6))
                    # elif args.eval_version == -1 and args.dataset_split == 'test':
                    #     eval_version = random.choice(range(6, 35))
                    # else:
                    #     eval_version = args.eval_version

                    # add multi-constraint and expansion-level logic
                    if hasattr(args, 'multi_constraints'):
                        if args.multi_constraints > 1:
                            # children = get_node_children(constraint, hierarchy, args.expansion_level_for_multi_constraint)
                            candidates = set(hierarchy[topic]) - {constraint}
                            # shall we at least have one additional constraint?
                            # no, because "have nothing to talk" situation is quite interesting as well
                            additional_constraints = random.sample(candidates, min(args.multi_constraints - 1, len(candidates)))
                            datapoint['multi_constraints'] = [constraint] + list(additional_constraints)
                            all_constrained_nodes = set()
                            for c in [constraint] + additional_constraints:
                                all_constrained_nodes.update(get_flattened_subtree(c, hierarchy, args.expansion_level_for_multi_constraint))
                            datapoint['all_constrained_nodes'] = all_constrained_nodes
                            remaining_topical_words = set(hierarchy[topic]) - all_constrained_nodes
                            remaining_nodes = set()
                            for c in remaining_topical_words:
                                remaining_nodes.update(get_flattened_subtree(c, hierarchy, None))
                            datapoint['remaining_topical_words'] = remaining_topical_words
                            datapoint['remaining_nodes'] = remaining_nodes
                        if args.multi_constraints == 0:
                            datapoint['multi_constraints'] = []
                            datapoint['all_constrained_nodes'] = set()
                            remaining_topical_words = set(hierarchy[topic])
                            datapoint['remaining_topical_words'] = remaining_topical_words
                            remaining_nodes = set()
                            for c in remaining_topical_words:
                                remaining_nodes.update(get_flattened_subtree(c, hierarchy, None))
                            datapoint['remaining_nodes'] = remaining_nodes

                            # for child in children:
                            #     new_datapoint = copy.deepcopy(datapoint)
                            #     new_datapoint['constraint'] = child
                            #     prepared_datapoint = prepare_context(new_datapoint, args, version=args.eval_version)
                            #     datasets[dataset_split].append(prepared_datapoint)

                    prepared_datapoint = prepare_context(datapoint, args, version=args.eval_version)
                    datasets[dataset_split].append(prepared_datapoint)
        if randomize:
            random.shuffle(datasets[dataset_split])

        num_datapoints = getattr(args, dataset_split + '_num_datapoints', 100000000)
        datasets[dataset_split] = datasets[dataset_split][:num_datapoints]
    return datasets, hierarchy, None


def get_wikidata_p_text(p_name, p_value):
    if p_name == 'place_of_birth':
        fill_text = f'people who were born in {p_value}'
    elif p_name == 'place_of_death':
        fill_text = f'people who died in {p_value}'
    elif p_name == 'occupation':
        fill_text = p_value
    elif p_name == 'country_of_citizenship':
        fill_text = f'people who are citizens of {p_value}'
    elif p_name == 'academic_degree':
        fill_text = f'people who hold a degree in {p_value}'
    elif p_name == 'educated_at':
        fill_text = f'people who had their education at {p_value}'
    return fill_text


def cleanup_gen_text(text):
    """
    Cleanup items
    - drop the last unfinished sentence (should finish with `==`)

    The normal case: `len(text.split('\n')) == 3`.
    """
    sents = text.strip().split('\n')
    return sents[0]


#    print(sents)
#    if len(sents) > 2 and not sents[-1].endswith('=='):
#        sents = sents[:-1]
#    return '\n'.join(sents)


def reformat_text(text):
    text_tokens = text.split(' ')
    text = ' '.join(text_tokens)
    return text


def get_hierarchy_path_to_children(path):
    hierarchy_path_to_children = dict()
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            hierarchy_path = obj['hierarchy_path']
            children = obj['children']
            hierarchy_path_to_children[tuple(hierarchy_path)] = children
    return hierarchy_path_to_children


def get_hierarchy(path=None):
    if path is None:
        path = 'cognac_origin/wordnet/topic_to_leafs.json'
    with open(path) as f:
        hierarchy_ = json.load(f)
    hierarchy = dict()
    for topic, leafs in hierarchy_.items():
        new_topic = topic.replace('_', ' ')
        new_leafs = [l.replace('_', ' ') for l in leafs]
        hierarchy[new_topic] = new_leafs
    return hierarchy

def get_flattened_subtree(node, hierarchy, expansion_level=None):
    """
    Get all children (including grand^n son) of a given `node` in the `hierarchy`.
    """
    # nodes = set(hierarchy[node] + [node])
    # children = set()
    # for n in nodes:
    #     children.update(set(hierarchy[n]))
    # stacks = [(node, 0)]
    queue = Queue()
    queue.put((node, 0))
    returned_nodes = set()
    while not queue.empty():
        n, level = queue.get()
        if expansion_level is not None and level >= expansion_level:
            break
        returned_nodes.add(n)
        if n in hierarchy:
            for child in hierarchy[n]:
                queue.put((child, level + 1))
    return returned_nodes


def get_node_children_in_text(text, node, hierarchy):
    """
    Check if `text` contains `node`'s children in the `hierarchy`.
    """
    nodes = set(hierarchy[node] + [node])
    return list(set([node for node in nodes if node in text]))


def get_instruction(datapoint, args, version):
    if args.data == 'wordnet':
        topic = datapoint['topic']
        constraint = datapoint['constraint']
        if "multi_constraints" in datapoint:
            constraint = "[{}]".format(", ".join(datapoint["multi_constraints"]))
        insert_position, *instruction_templates, mode = MAPPING[version]
        if insert_position == 'begin+end':
            instructions = [
                instruction_templates[0].format(topic),
                instruction_templates[1].format(constraint),
            ]
        elif insert_position == 'begin':
            instructions = [instruction_templates[0].format(constraint, topic), ""]
        elif insert_position == 'end':
            instructions = ["", instruction_templates[0].format(constraint, topic)]
        else:
            raise ValueError(f'`{insert_position}` not recognized.')

    elif args.data == 'wikidata':
        topic = datapoint['topic']
        constraint = datapoint['constraint']
        insert_position, *instruction_templates, mode = MAPPING[version]

        topic = get_wikidata_p_text(topic[0], topic[1])
        constraint = get_wikidata_p_text(constraint[0], constraint[1])

        if insert_position == 'begin+end':
            instructions = [
                instruction_templates[0].format(topic),
                instruction_templates[1].format(constraint),
            ]
        elif insert_position == 'begin':
            instructions = [instruction_templates[0].format(constraint, topic), ""]
        elif insert_position == 'end':
            instructions = ["", instruction_templates[0].format(constraint, topic)]
        else:
            raise ValueError(f'`{insert_position}` not recognized.')

    return dict(
        begin=instructions[0],
        end=instructions[1],
        insert_position=insert_position,
    )


def prepare_context(datapoint, args, version=-2):
    new_datapoint = copy.deepcopy(datapoint)
    blocks = new_datapoint['context']
    blocks = ['== ' + sent.strip('==').strip() + ' ==' for sent in blocks.split('==\n')]

    instruction = get_instruction(new_datapoint, args, version)
    begin = instruction['begin']
    end = instruction['end']
    insert_position = instruction['insert_position']
    blocks = [begin] + blocks + [end]
    context_with_instructions = '\n'.join(blocks).strip()

    new_datapoint['context_with_instructions'] = context_with_instructions
    new_datapoint['version'] = version
    new_datapoint['begin'] = begin
    new_datapoint['end'] = end
    new_datapoint['insert_position'] = insert_position
    return new_datapoint
# class WikidataHierarchy:
#     def __init__(self, q_to_p, p_to_q):
#         self.q_to_p = q_to_p
#         self.p_to_q = p_to_q
#
#     def __contains__(self, key):
#         if self.__getitem__(key):
#             return True
#         else:
#             return False
#
#     def __getitem__(self, p):
#         """
#         For a given `p` (i.e., topic or constraint), return a list of all Q's
#         that have this `p`.
#         Return list like [(Q7339, 'Margot Frank'), ...].
#         """
#         if isinstance(p, list):
#             p = tuple(p)
#
#         return self.p_to_q.get(p, [])


# def get_wikidata_hierarchy():
#     from evaluation.scripts.build_wikidata_dataset import (
#         load_ranked_properties,
#         load_all_entities
#     )
#     WIKIDATA_PATH = Path('path/to/your/qid2title.json')
#     q_to_p, p_to_q = load_all_entities(WIKIDATA_PATH)
#     hierarchy = WikidataHierarchy(q_to_p, p_to_q)
#     return hierarchy
