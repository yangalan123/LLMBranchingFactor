import csv
import os
from storytelling.ttcw.data_utils import load_stories_with_plots
import glob
import argparse
import random
import torch
def story_postprocessing(story, remove_first_sentence=False):
    processed_story = story.strip()
    if remove_first_sentence:
        sents = processed_story.split("\n\n")
        sents = sents[1:]
        processed_story = "\n\n".join(sents)
    # exclude markdowns, html, urls, and special characters
    if processed_story.startswith('\\') or processed_story.startswith("(") or processed_story.startswith("[") or processed_story.startswith("{"):
        return None
    if processed_story.startswith("#") or processed_story.startswith("`") or processed_story.startswith("!") or processed_story.startswith("<"):
        return None
    if "http" in processed_story or "www" in processed_story:
        return None
    if len(processed_story) == 0:
        return None
    return processed_story


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StoryWritingArgsParsing.')
    parser.add_argument('--input_dir', type=str, help='input dir', default="local_generated_story_full_ttcw")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p value for nucleus sampling")
    parser.add_argument('--output_file', type=str, help='output filename', default="local_generated_story_extracted.pt")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    random.seed(args.seed)
    input_dir = args.input_dir
    top_p = args.top_p
    output_file = os.path.join(input_dir, args.output_file)
    plot_idx_to_story = load_stories_with_plots()
    input_files = glob.glob(os.path.join(input_dir, "*top_p_{}_*.csv".format(top_p)))
    all_plot_id_to_story = dict()
    for input_filename in input_files:
        print("Processing: ", input_filename)
        _plot_id_to_story = dict()
        model_name = os.path.basename(input_filename).split("_response")[0]
        with open(input_filename, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                _id = row['id']
                # extract plot_id and output_i from _id= plot_{plot_id}_output_{output_i}
                plot_id = int(_id.split("_")[1])
                output_i = _id.split("_")[-1]
                if plot_id not in _plot_id_to_story:
                    _plot_id_to_story[plot_id] = []
                story = row["response"]
                processed_story = story_postprocessing(story)
                if processed_story is not None:
                    _plot_id_to_story[plot_id].append(processed_story)
        for plot_id in _plot_id_to_story:
            assert len(_plot_id_to_story[plot_id]) >= 1, f"plot_id: {plot_id}, {_plot_id_to_story[plot_id]}"
            sampled_story = random.choice(_plot_id_to_story[plot_id])
            assert plot_id in plot_idx_to_story, f"missing plot_id in TTCW identified: {plot_id}"
            plot_idx_to_story[plot_id][model_name] = sampled_story
        all_plot_id_to_story[model_name] = _plot_id_to_story
    torch.save([plot_idx_to_story, all_plot_id_to_story, args], output_file)




