import os

ALL_MODELS = ["meta-llama/Llama-2-70b-chat-hf",
              "meta-llama/Llama-2-13b-chat-hf",
              "meta-llama/Llama-2-13b-hf",
              "meta-llama/Llama-2-70b-hf",
              "01-ai/Yi-34B-Chat",
              "01-ai/Yi-34B",
              "mistralai/Mixtral-8x7B-v0.1",
              "mistralai/Mixtral-8x7B-Instruct-v0.1",
              "meta-llama/Meta-Llama-3-8B",
              "meta-llama/Meta-Llama-3-8B-Instruct",
              "meta-llama/Meta-Llama-3-70B-Instruct",
              "meta-llama/Meta-Llama-3-70B",
              "Qwen/Qwen3-4B-Base"
              ]

root_path = os.environ.get(
    "BF_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
cot_hub_root = os.environ.get(
    "BF_COT_HUB_ROOT",
    os.path.join(root_path, "chain-of-thought-hub"),
)
chat_templates_root = os.environ.get(
    "BF_CHAT_TEMPLATES_ROOT",
    os.path.join(root_path, "chat_templates"),
)
mmlu_prompt_templates_path = os.path.join(cot_hub_root, "MMLU")
DATA_DIR = f"{mmlu_prompt_templates_path}/data"
COT_PROMPT_PATH = f"{mmlu_prompt_templates_path}/lib_prompt/mmlu-cot.json"
STANDARD_PROMPT_PATH = f"{mmlu_prompt_templates_path}/lib_prompt/mmlu-standard.json"
# the files below can be created by running mmlu/create_dummy_cot_file.py
FILLER_COT_PROMPT_PATH = f"{mmlu_prompt_templates_path}/lib_prompt/mmlu-filler.json"
WRONGANSWER_COT_PROMPT_PATH = f"{mmlu_prompt_templates_path}/lib_prompt/mmlu-wrong-answer.json"

CHAT_TEMPLATES_PATH = {
    "Llama-2-70b-chat-hf": os.path.join(chat_templates_root, "chat_templates", "llama-2-chat.jinja"),
    "Llama-2-13b-chat-hf": os.path.join(chat_templates_root, "chat_templates", "llama-2-chat.jinja"),
    "Llama-2-7b-chat-hf": os.path.join(chat_templates_root, "chat_templates", "llama-2-chat.jinja"),
    "Yi-34B-Chat": os.path.join(chat_templates_root, "chat_templates", "chatml.jinja"),
    "Mixtral-8x7B-Instruct-v0.1": os.path.join(chat_templates_root, "chat_templates", "mistral-instruct.jinja"),
    "meta-llama/Meta-Llama-3-8B-Instruct": os.path.join(chat_templates_root, "chat_templates", "llama-3-instruct.jinja"),
    "meta-llama/Meta-Llama-3-70B-Instruct": os.path.join(
        chat_templates_root, "chat_templates", "llama-3-instruct.jinja"),
}
tolerance_inf = 1e-12
