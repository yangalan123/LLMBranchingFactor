import os.path

import torch
from transformers import AutoTokenizer
import argparse
import random
from uncertainty_quantification.consts import ALL_MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RandomStrArgsParser.')
    parser.add_argument('--num_samples', type=int, default=200, help='number of samples')
    parser.add_argument('--min_length', type=int, default=256, help='minimum length of the random string')
    parser.add_argument('--max_length', type=int, default=512, help='maximum length of the random string')
    args = parser.parse_args()
    considered_models = [x for x in ALL_MODELS if "llama" in x.lower()]
    shared_vocab = set()
    for _model in considered_models:
        tokenizer = AutoTokenizer.from_pretrained(_model)
        vocab = tokenizer.get_vocab()
        all_vocab = list(vocab.keys())
        if len(shared_vocab) == 0:
            shared_vocab.update(all_vocab)
        else:
            shared_vocab.intersection_update(all_vocab)
    # get the vocabulary
    #vocab = tokenizer.get_vocab()
    #all_vocab = list(vocab.keys())
    #model_name = os.path.basename(args.model)
    #output_filename = f"random_strings_{model_name}_{args.num_samples}_{args.min_length}_{args.max_length}.pt"
    output_filename = f"random_strings_shared_llama_{args.num_samples}_{args.min_length}_{args.max_length}.pt"
    buf = []
    shared_vocab = list(shared_vocab)
    print("Shared vocab size: ", len(shared_vocab))
    for i in range(args.num_samples):
        length = random.randint(args.min_length, args.max_length)
        random_string = " ".join([random.choice(shared_vocab) for _ in range(length)])
        buf.append(random_string)
    torch.save(buf, output_filename)
    print(f"Saved to {output_filename}")


