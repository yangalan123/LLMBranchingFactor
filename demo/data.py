from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import torch

def get_data(args):
    # default dataset is wikitext-103-v1, as shown below in main argument parser
    if "wikitext" in args.dataset_name.lower():
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