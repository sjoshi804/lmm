import re 

def spatial_map_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def spatial_map_doc_to_text(doc, lmms_eval_specific_kwargs):
    return doc["prompt"].replace(
        "First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation.",
        lmms_eval_specific_kwargs["post_prompt"]
    )

def spatial_map_process_results(doc, results):
    task = doc["task"]
    pred = results[0]
    try:
        # Parse option if MCQ
        options_map = extract_options_map(doc["prompt"])
        doc["ground_truth_option"] = options_map.get(doc["ground_truth"], "")
        
        # Extract answer from prediction
        pred = extract_answer(pred, list(options_map.keys()))
        if pred is None:
            raise ValueError(f"Cannot extract answer from prediction: {pred}")
        
        # Compare answer to ground truth / ground truth option
        pred, ground_truth, ground_truth_option = pred.lower(), doc["ground_truth"].lower(), doc["ground_truth_option"].lower()
        score = 1.0 if (pred == ground_truth or pred == ground_truth_option or int(pred) == int(ground_truth)) else 0.0
        return {task: score}
    
    except Exception as e:
        print(f"Error: {e}")
        return {task: 0.0, "parsing_error": 1.0}

def extract_answer(pred, options):
    for option in options:
        if pred.lower() == option.lower():
            return option

    match = re.search(r"(?:A:|Answer:)?\s*([ABCD])", pred, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:A:|Answer:)?\s*\b\d+\b", pred, re.IGNORECASE)
    if match:
        return match.group(0)

    return None

def extract_options_map(prompt):
    match = re.search(r"Available options:\s*\nA\.\s*(\w+)\s*\nB\.\s*(\w+)\s*\nC\.\s*(\w+)\s*\nD\.\s*(\w+)\.", prompt)
    if match:
        options_map = {
            match.group(1): "A",
            match.group(2): "B",
            match.group(3): "C",
            match.group(4): "D",
        }
        return options_map
    else:
        return {}

def remove_redundancy(text):
    """
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.
    """
    text = text.replace("**", "")
    text = text.replace(".", "")
    return text

def extract_before_is(input_string):
    """
    This function extracts the part of the string before the first occurrence of 'is'.
    The code is from: https://github.com/alvinmingwisc/spatial_reason_vlm/tree/main/eval,
    and included with minimal modifications.

    :param input_string: The string to process.
    :return: A new string containing the part before 'is'.
    """
    # Split the string at the first occurrence of 'is'
    parts = input_string.split(" is", 1)
    # Return the first part
    return parts[0].strip()

