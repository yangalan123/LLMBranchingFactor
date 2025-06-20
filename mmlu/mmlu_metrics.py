from collections import Counter
import re


def extract_answer(output, model_name, prompt_family):
    """
    Extract answers (A/B/C/D) from model outputs using various patterns.

    Args:
        outputs (str): The model's response text
        model_name (str): Name of the model used
        prompt_family (str): Type of prompt used (e.g., "standard", "cot")

    Returns:
        str: Extracted answer (A, B, C, or D) or None if no valid answer found
    """

    def is_base_model(model_name):
        """Check if the model is a base model without chat/instruct capabilities."""
        return not any(keyword in model_name.lower() for keyword in ['chat', 'instruct'])

    def clean_text(text):
        """Clean text by removing extra spaces and normalizing newlines."""
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    def extract_from_patterns(text):
        """Try to extract answer using various answer indication patterns."""
        # Pattern collection for different answer formats
        patterns = [
            r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*(?:choice\s*)?[:\(]?\s*([ABCD])[\)\s]*",
            # "The answer is A" or "The answer is (A)"
            r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*(?:choice\s*)?[:\(]?\s*\n\s*([ABCD])",  # "The answer is:\nA"
            r"(?:the\s+)?(?:correct\s+)?answer\s*[:\(]?\s*\n\s*［([ABCD])］",  # Answer with full-width brackets
            r"(?:the\s+)?(?:correct\s+)?answer\s*[:\(]?\s*\n\s*（([ABCD])）",  # Answer with Chinese brackets
            r"^\s*([ABCD])\s*$"  # Single letter answer
        ]

        # Try each pattern
        text = text.upper()  # Normalize to uppercase
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def find_first_letter_answer(text):
        """Find the first standalone A/B/C/D in the text."""
        match = re.search(r'\b([ABCD])\b', text.upper())
        return match.group(1) if match else None

    # Main extraction logic
    cleaned_output = clean_text(output)

    # First try to extract using explicit answer patterns
    answer = extract_from_patterns(output)
    if answer:
        return answer

    # For base models, fall back to first letter matching
    if is_base_model(model_name):
        answer = find_first_letter_answer(cleaned_output)
        if answer:
            return answer

    # Default fallback if no answer is found
    return None

def mmlu_metrics(outputs, target, model_name, prompt_family, self_consistency=False):
    '''

    :param outputs: list of strings -- different output for the same prompt
    :param target: the answer label, can only be in [A, B, C, D]
    :param self_consistency: whether using majority voting to determine the final answer, by default No and we just report average accuracy
    :return: float -- accuracy
    '''
    # need some investigation to determine how we should extract the answer
    extracted_answers = [extract_answer(output, model_name, prompt_family) for output in outputs]
    non_answers = [x for x in extracted_answers if x is None]
    if self_consistency:
        dist_answers = Counter(extracted_answers)
        final_answer = dist_answers.most_common(1)[0][0]
        return 1 if final_answer == target else 0, len(non_answers) / len(extracted_answers)
    else:
        indicators = [1 if extracted_answer == target else 0 for extracted_answer in extracted_answers]
        return sum(indicators) / len(indicators), len(non_answers) / len(extracted_answers)


def test_extract_answer():
    test_cases = [
        ("The correct answer is choice C", "model", "standard", "C"),
        ("The answer is A", "model", "standard", "A"),
        ("The answer is (B)", "model", "standard", "B"),
        ("The correct answer is:\n    A", "model", "standard", "A"),
        ("the correct answer is:\n   (D)", "model", "standard", "D"),
        ("the correct answer is (A)", "model", "standard", "A"),
        ("Let's solve this step by step...\nA", "base-model", "standard", "A"),
        ("Answer:...\nthe answer is (B).", "chat-model", "standard", "B"),
        #("Complex reasoning... Therefore B is correct", "chat-model", "cot", "B"),
        ("The correct answer is choice C", "gpt-3.5", "standard", "C"),
        ("The answer is A", "gpt-4", "cot", "A"),
        ("The answer is (B)", "llama", "standard", "B"),
        ("The correct answer is:\nA", "chat-gpt", "cot", "A"),
        ("the correct answer is:\n（D）", "bert", "standard", "D"),
        ("the correct answer is (A)", "t5", "standard", "A"),
        ("A", "base-model", "standard", "A"),
        ("Let me think... A seems correct", "base-model", "standard", "A"),
    ]

    for input_text, model, prompt, expected in test_cases:
        result = extract_answer(input_text, model, prompt)
        assert result == expected, f"Failed on input: {input_text}\nExpected: {expected}, Got: {result}"
    print("Pass the test!")


if __name__ == "__main__":
    test_extract_answer()
