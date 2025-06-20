from datasets import load_dataset
import random

import torch
from nltk.tokenize import word_tokenize

roles = [""]

def get_data_from_huggingface(args):
    if "wikitext" in args.dataset_path.lower():
        # normal wikipedia task
        ds = load_dataset(args.dataset_path, args.dataset_name)
        test_ds = ds['test'].filter(lambda x: len(word_tokenize(x['text'])) > args.min_word_count)
        sample_counts = min(args.dataset_sample_counts, len(test_ds))
        sampled_ds = test_ds.shuffle(seed=args.seed).select(range(sample_counts))
        return [x['text'] for x in sampled_ds]
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
        sampled_strings = random.sample(string_buffer, sample_counts)
        return sampled_strings
    elif "alpacaeval_category" in args.dataset_path:
        ds = load_dataset("dogtooth/alpaca_eval")['eval']
        return [x['instruction'] for x in ds]
    elif "just-eval-instruct" in args.dataset_path:
        ds = load_dataset("re-align/just-eval-instruct", "default")['test']
        return [x['instruction'] for x in ds]


