import json
from nltk import word_tokenize
from uncertainty_quantification.consts import mmlu_prompt_templates_path


def replace_cot_with_filler(text):
    shots = text.split("\n\n")
    processed_shots = []

    for shot in shots:
        start = shot.find("A:") + 2
        end = shot.find("The answer is")
        if start != -1 and end != -1:
            cot_segment = shot[start:end].strip()
            word_count = len(word_tokenize(cot_segment))
            filler = " ".join(["..."] * word_count)
            processed_shot = shot[:start] + " " + filler + " " + shot[end:]
        else:
            processed_shot = shot
        processed_shots.append(processed_shot)

    return "\n\n".join(processed_shots)

def wrong_answer(text):
    shots = text.split("\n\n")
    processed_shots = []

    for shot in shots:
        # Find all occurrences of "The answer is (X)" where X is A, B, C, or D
        start = shot.find("The answer is (")
        if start != -1:
            # Find the closing parenthesis
            end = shot.find(")", start)
            if end != -1:
                # Replace the answer with (A)
                processed_shot = shot[:start] + "The answer is (A)" + shot[end + 1:]
            else:
                processed_shot = shot
        else:
            processed_shot = shot
        processed_shots.append(processed_shot)

    return "\n\n".join(processed_shots)


if __name__ == '__main__':
    cot_file = f"{mmlu_prompt_templates_path}/lib_prompt/mmlu-cot.json"
    with open(cot_file, "r", encoding='utf-8') as f:
        with open(cot_file.replace("mmlu-cot", "mmlu-filler"), "w", encoding='utf-8') as f_out:
            cot_data = json.load(f)
            filler_data = {}
            for category in cot_data:
                prompt = cot_data[category]
                filler_prompt = replace_cot_with_filler(prompt)
                filler_data[category] = filler_prompt
            json.dump(filler_data, f_out, indent=2, ensure_ascii=False)
    print("Created filler COT file: ", cot_file.replace("mmlu-cot", "mmlu-filler"))
    with open(cot_file, "r", encoding='utf-8') as f:
        with open(cot_file.replace("mmlu-cot", "mmlu-wrong-answer"), "w", encoding='utf-8') as f_out:
            cot_data = json.load(f)
            filler_data = {}
            for category in cot_data:
                prompt = cot_data[category]
                filler_prompt = wrong_answer(prompt)
                filler_data[category] = filler_prompt
            json.dump(filler_data, f_out, indent=2, ensure_ascii=False)
    print("Created wrong answer COT file: ", cot_file.replace("mmlu-cot", "mmlu-wrong-answer"))