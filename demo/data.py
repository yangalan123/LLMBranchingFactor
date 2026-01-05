from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import os

# Import functions from other modules
try:
    from mmlu.mmlu_prompt_utils import create_mmlu_data_constraints
except ImportError:
    create_mmlu_data_constraints = None

try:
    from cognac.cognac_utils import get_cognac_data
except ImportError:
    get_cognac_data = None

try:
    from storytelling.consts import task_prompt, roles as storytelling_roles, ablate_task_prompt
except ImportError:
    task_prompt = None
    storytelling_roles = None
    ablate_task_prompt = None

try:
    from uncertainty_quantification.tokenizer_utils import format_prompt
except ImportError:
    format_prompt = None


def get_data(args):
    """
    Unified data loading function that handles multiple dataset types.
    Supports: wikitext, creative_storygen, bbc_news, cnn_dailymail, random_strings, 
    alpacaeval_category, just-eval-instruct, storytelling, mmlu, cognac
    """
    # Language modeling datasets (from language_modeling/data_utils.py)
    if "wikitext" in args.dataset_name.lower() or ("wikitext" in args.dataset_path.lower() and not hasattr(args, 'task_type')):
        # normal wikipedia task
        ds = load_dataset(args.dataset_path, args.dataset_name)
        test_ds = ds['test'].filter(lambda x: len(word_tokenize(x['text'])) > args.min_word_count)
        sample_counts = min(args.dataset_sample_counts, len(test_ds))
        sampled_ds = test_ds.shuffle(seed=args.seed).select(range(sample_counts))
        return [x['text'] for x in sampled_ds]
    elif "creative_storygen" in args.dataset_name.lower():
        # plot_id_to_story, all_plot_id_to_story, input_file_args = torch.load("local_generated_story_extracted.pt", weights_only=False)
        plot_id_to_story, all_plot_id_to_story, input_file_args = torch.load(args.dataset_path, weights_only=False)
        all_original_prompts = []
        model_families = None
        for plot_id in plot_id_to_story:
            if model_families is None:
                model_families = list(plot_id_to_story[plot_id].keys())
            for model_family in model_families:
                all_original_prompts.append((plot_id_to_story[plot_id][model_family], model_family))
        return [x[0] for x in all_original_prompts]
    elif "bbc_news_alltime" in args.dataset_path:
        year = args.dataset_name.split("_")[0]
        month_start, month_end = args.dataset_name.split("_")[1].split("-")
        month_start = int(month_start)
        month_end = int(month_end)
        dataset_buffer = []
        for _month in range(month_start, month_end+1):
            month_str = str(_month)
            if len(month_str) == 1:
                month_str = "0" + month_str
            ds = load_dataset(args.dataset_path, f"{year}-{month_str}")
            test_ds = ds['train'].filter(lambda x: len(word_tokenize(x['content'])) > args.min_word_count)
            sample_counts = min(args.dataset_sample_counts, len(test_ds))
            sampled_ds = test_ds.shuffle(seed=args.seed).select(range(sample_counts))
            dataset_buffer.extend([x['content'] for x in sampled_ds])
        return dataset_buffer
    elif "cnn_dailymail" in args.dataset_path:
        ds = load_dataset(args.dataset_path, args.dataset_name)
        # it is very likely that llm will be trained on cnn_dailymail, so we will use the test set
        test_ds = ds['test'].filter(lambda x: len(word_tokenize(x['article'])) > args.min_word_count)
        sample_counts = min(args.dataset_sample_counts, len(test_ds))
        sampled_ds = test_ds.shuffle(seed=args.seed).select(range(sample_counts))
        return [x['article'] for x in sampled_ds]
    elif "random_strings" in args.dataset_path:
        string_buffer = torch.load(args.dataset_path)
        string_buffer = [x for x in string_buffer if len(x.split(" ")) > args.min_word_count]
        sample_counts = min(args.dataset_sample_counts, len(string_buffer))
        # randomly sample from the string buffer
        import random
        sampled_strings = random.sample(string_buffer, sample_counts)
        return sampled_strings
    elif "alpacaeval_category" in args.dataset_path:
        ds = load_dataset("dogtooth/alpaca_eval")['eval']
        return [x['instruction'] for x in ds]
    elif "just-eval-instruct" in args.dataset_path:
        ds = load_dataset("re-align/just-eval-instruct", "default")['test']
        return [x['instruction'] for x in ds]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_path} / {args.dataset_name}")


def get_mmlu_data(args):
    """
    Load MMLU data using create_mmlu_data_constraints.
    Returns: (all_constrained_prompts, roles, answers, options)
    """
    if create_mmlu_data_constraints is None:
        raise ImportError("mmlu.mmlu_prompt_utils.create_mmlu_data_constraints not available")
    
    tasks, task_to_ids = torch.load(args.task_selection_filename)
    constraint_level = getattr(args, 'constraint_level', 0)
    expand_options = getattr(args, 'expand_options', False)
    nudging = getattr(args, 'nudging', False)
    
    nudging_kwargs = None
    if nudging:
        from loguru import logger
        nudging_kwargs = {
            "ckpt_path": getattr(args, 'nudging_ckpt_path', None),
            "model": getattr(args, 'nudging_model', None),
            "nudging_max_prefix_length": getattr(args, 'nudging_max_prefix_length', 5),
            "nudging_freq_threshold": getattr(args, 'nudging_freq_threshold', 50),
            "logger": logger
        }
    
    all_constrained_prompts, roles, answers, options = create_mmlu_data_constraints(
        tasks, task_to_ids,
        constraint_level,
        expand_options=expand_options,
        nudging=nudging,
        nudging_kwargs=nudging_kwargs
    )
    return all_constrained_prompts, roles, answers, options


def get_storytelling_data(args):
    """
    Load storytelling data from input_file or use task_prompt/ablate_task_prompt from consts.
    Returns: list of prompts (can be tuples of (prompt_text, model_family))
    """
    if args.input_file is not None and os.path.exists(args.input_file):
        plot_id_to_story, all_plot_id_to_story, input_file_args = torch.load(args.input_file, weights_only=False)
        all_original_prompts = []
        model_families = None
        for plot_id in plot_id_to_story:
            if model_families is None:
                model_families = list(plot_id_to_story[plot_id].keys())
            for model_family in model_families:
                all_original_prompts.append((plot_id_to_story[plot_id][model_family], model_family))
        return all_original_prompts
    else:
        # Use task_prompt and ablate_task_prompt from consts
        if task_prompt is None or ablate_task_prompt is None:
            raise ImportError("storytelling.consts.task_prompt and ablate_task_prompt not available")
        return ablate_task_prompt + task_prompt


def get_cognac_data_wrapper(args):
    """
    Load Cognac data using get_cognac_data.
    Returns: (prompts, answers, sources, (hierarchy, updated_args))
    """
    if get_cognac_data is None:
        raise ImportError("cognac.cognac_utils.get_cognac_data not available")
    
    model = args.model
    chat_template_path = getattr(args, 'chat_template_path', None)
    task_selection_filename = getattr(args, 'task_selection_filename', None)
    
    select_ids = None
    if task_selection_filename is not None and os.path.exists(task_selection_filename):
        tasks, task_to_ids = torch.load(task_selection_filename)
        assert list(task_to_ids.keys()) == ['cognac'], "Only one task is allowed for cognac"
        select_ids = task_to_ids['cognac']
    
    prompts, answers, sources, (hierarchy, updated_args) = get_cognac_data(
        model, chat_template_path, select_ids=select_ids, update_args=args
    )
    return prompts, answers, sources, (hierarchy, updated_args)


def apply_constraints_language_modeling_storytelling(all_original_prompts, args):
    """
    Apply constraints for language modeling and storytelling tasks.
    Uses constraint_level and max_constraint_level with sentence/word tokenization.
    """
    constrained_prompt = []
    constraint_level = getattr(args, 'constraint_level', 0)
    max_constraint_level = getattr(args, 'max_constraint_level', 10)
    word_level_constraint = getattr(args, 'word_level_constraint', False)
    word_level_constraint_multiplier = getattr(args, 'word_level_constraint_multiplier', 10)
    
    tokenize_func = sent_tokenize if not word_level_constraint else word_tokenize
    _constraint_level = constraint_level * word_level_constraint_multiplier if word_level_constraint else constraint_level
    _max_constraint_level = max_constraint_level * word_level_constraint_multiplier if word_level_constraint else max_constraint_level
    
    for _prompt in all_original_prompts:
        if isinstance(_prompt, tuple):
            _prompt_text, _model = _prompt
        else:
            _prompt_text = _prompt
        if _max_constraint_level > 0:
            tokens = tokenize_func(_prompt_text)
            if _constraint_level >= 0:
                tokens = tokens[:_constraint_level]
            tokens = tokens[:_max_constraint_level]
            constrained_prompt.append(" ".join(tokens))
        else:
            constrained_prompt.append(_prompt_text)
    return constrained_prompt


def process_prompts_with_roles(all_task_prompts, model, tokenizer, roles, system_message_template="", generated_prefix=""):
    """
    Process prompts with roles (for storytelling task).
    """
    if format_prompt is None:
        raise ImportError("uncertainty_quantification.tokenizer_utils.format_prompt not available")
    
    prompts = []
    for _task_prompt in all_task_prompts:
        for role in roles:
            if len(role) == 0:
                system_message = ""
            else:
                if role.startswith("#"):
                    system_message = role.strip("#")
                else:
                    system_message = system_message_template + role if system_message_template else role
            prompts.append(format_prompt(model, _task_prompt, tokenizer, system_message) + generated_prefix)
    return prompts