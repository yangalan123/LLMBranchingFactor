from transformers import AutoTokenizer
import os
from uncertainty_quantification.consts import CHAT_TEMPLATES_PATH

def setup_tokenizer(model, chat_template_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model)
    if chat_template_path is None:
        if os.path.basename(model) in CHAT_TEMPLATES_PATH:
            chat_template_path = CHAT_TEMPLATES_PATH[os.path.basename(model)]
    if "chat" in model.lower() or "instruct" in model.lower():
        # chat-based model
        if chat_template_path is not None:
            chat_template = open(chat_template_path).read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            tokenizer.chat_template = chat_template

    return tokenizer


def detokenization(tokenizer, token_ids):
    # vllm detokenization method, already verified using llama-2 outputs on >1000 samples
    detokenized_output = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    detokenized_output = tokenizer.convert_tokens_to_string(detokenized_output)
    return detokenized_output


def format_prompt(model, prompt, tokenizer, system_message=None):
    if "chat" in model.lower() or "instruct" in model.lower():
        messages = [
            {'role': "system",
             'content': "Follow the given examples and answer the question." if system_message is None else system_message},
            {'role': "user", 'content': prompt}
        ]
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        new_prompt = prompt
    return new_prompt
