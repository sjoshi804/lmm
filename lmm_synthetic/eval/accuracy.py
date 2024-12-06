import argparse
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM

from lmm_synthetic.mm_train.utils import load_vision_encoder
from lmm_synthetic.mm_train.gptj_vlm import GPTJ_VLM

def load_model_and_tokenizer(model_path, multimodal=False):
    """
    Load the model and tokenizer from the specified path.
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    model, tokenizer = None, None
    if multimodal:
        model = GPTJ_VLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model.config.pretrained_lm_path)
    else:
        model = GPTJForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def load_dataset(dataset_path):
    """
    Load the dataset from the specified path.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    return load_from_disk(dataset_path)

def generate_responses_lm(model, tokenizer, prompts, max_new_tokens=50, max_length=512):
    """
    Generate texts from the model for a batch of prompts using GPU if available.
    
    Args:
        model: Language model for inference.
        tokenizer: Tokenizer corresponding to the model.
        prompts: A list of text prompts.
        max_new_tokens: Maximum number of new tokens to generate.
        max_length: Maximum length of the input sequence.
    
    Returns:
        A list of generated text responses.
    """
    # Ensure the model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize prompts and move inputs to GPU
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    # Generate responses for the batch
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Decode the generated outputs
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return responses


def generate_responses_vlm(model, tokenizer, image_tensors, prompts, max_new_tokens=50, max_length=512):
    """
    Generate text responses from the model for a batch of image and text prompts on GPU.
    
    Args:
        model: GPTJ_VLM model for multimodal inference.
        tokenizer: Tokenizer corresponding to the model.
        image_tensors: A batch of preprocessed image tensors (shape: [batch_size, *image_dim]).
        prompts: A list of text prompts corresponding to the images.
        max_new_tokens: Maximum number of new tokens to generate.
        max_length: Maximum length of the sequence.
    
    Returns:
        A list of generated text responses.
    """
    # Ensure the model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize text prompts for the batch and move to GPU
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    text_input_ids = inputs['input_ids'].to(device)

    # Ensure image tensors are on the GPU
    if image_tensors is not None:
        image_tensors = image_tensors.to(device)

    # Generate responses for the entire batch
    with torch.no_grad():
        outputs = model.generate(
            images=image_tensors,
            text_input_ids=text_input_ids,
            max_length=max_length + max_new_tokens,  # Include space for new tokens
            num_beams=1,  # Default to greedy decoding
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

    # Decode the generated outputs for the batch
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses


def parse_answer(text):
    """
    Parse the answer from the generated text.
    """
    match = re.search(r'A:\s*(.*)', text)
    if match:
        return match.group(1).split('\n')[0]
    return None


def parse_grid(grid_str, K):
    """
    Parse the grid string into a 2D list of grid cells.
    """
    grid_str = '\n'.join(grid_str.split('\n')[:K])
    rows = grid_str.strip().split('\n')
    return [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows]

def parse_vocab_subset(prompt):
    """
    Parse the vocabulary subset from the prompt.
    """
    match = re.search(r"Each cell contains an object from \[(.*?)\]", prompt)
    if match:
        return [item.strip().strip("'") for item in match.group(1).split(',')]
    return []

def evaluate_model_on_dataset(model, tokenizer, dataset, split, K, num_samples=250, multimodal_data=False, multimodal_model=False, debug=False):
    """
    Evaluate the model on the given dataset and compute accuracy.
    """
    num_grids = 0
    total_per_pos = {}
    correct_per_pos = {}
    in_vocab_subset_per_pos = {}
    accuracy = 0
    in_vocab_subset_rate = 0
    
    logger.info(f"Starting evaluation on {num_samples} grids")
    pbar = tqdm(enumerate(dataset[split]), total=num_samples)
    
    _, image_transforms, _ = load_vision_encoder("clip")
    
    for i, example in pbar:
        if num_grids == num_samples:
            break
        
        num_grids += 1
        text_prompt = example["text"].split(']')[0] + '].'
        grid = parse_grid(text_prompt, K)

        prompt = example["prompt"] if multimodal_data and multimodal_model else text_prompt 
        vocab_subset = parse_vocab_subset(prompt)
        
        # Generate prompts for all positions in the grid
        position_prompts = [
            prompt + f"\nWhat object is in row {i}, column {j}?"
            for i in range(K) for j in range(K)
        ]
        
        if i == 0:
            logger.debug(f"Sample Question (0,0): {position_prompts[0]}")
            
        if multimodal_model:
            if multimodal_data:
                image_tensor = image_transforms(Image.open(example["image"])).unsqueeze(0)
                image_tensors = [image_tensor for _ in range(K * K)]
                image_tensors = torch.cat(image_tensors, dim=0)
                responses = generate_responses_vlm(model, tokenizer, image_tensors, position_prompts, max_new_tokens=5)
            else:
                responses = generate_responses_vlm(model, tokenizer, None, position_prompts, max_new_tokens=5)
        else:
            responses = generate_responses_lm(model, tokenizer, position_prompts, max_new_tokens=5)

        # Evaluate predictions
        for idx, text in enumerate(responses):
            i, j = divmod(idx, K)
            total_per_pos[(i, j)] = total_per_pos.get((i, j), 0) + 1
            parsed_answer = parse_answer(text)
            if debug:
                logger.debug("Parsed Answer: " + parsed_answer)
                logger.debug("Actual Answer: " + grid[i][j])
            if parsed_answer == grid[i][j]:
                correct_per_pos[(i, j)] = correct_per_pos.get((i, j), 0) + 1
                in_vocab_subset_per_pos[(i, j)] = in_vocab_subset_per_pos.get((i, j), 0) + 1
            elif parsed_answer in vocab_subset:
                in_vocab_subset_per_pos[(i, j)] = in_vocab_subset_per_pos.get((i, j), 0) + 1
                
        accuracy = sum(correct_per_pos.values()) / sum(total_per_pos.values())
        in_vocab_subset_rate = sum(in_vocab_subset_per_pos.values()) / sum(total_per_pos.values())
        pbar.set_description(f'Accuracy: {accuracy:.3f}, In Vocab Subset: {in_vocab_subset_rate:.3f}')

    accuracy_per_pos = {pos:  correct_per_pos.get(pos, 0) / total_per_pos[pos] for pos in total_per_pos}
    in_vocab_subset_rate_per_pos = {pos: in_vocab_subset_per_pos.get(pos, 0) / total_per_pos[pos] for pos in total_per_pos}
    results = {
        "accuracy_per_pos": {str(pos): acc for pos, acc in accuracy_per_pos.items()},
        "accuracy": accuracy,
        "in_vocab_subset_rate_per_pos": {str(pos): rate for pos, rate in in_vocab_subset_rate_per_pos.items()},
        "in_vocab_subset_rate": in_vocab_subset_rate
    }
    
    return results


def save_results(results, model_path, dataset_path, file_path):
    """
    Save results to a JSON file.
    """
    results.update({
        "model_path": model_path,
        "dataset_path": dataset_path
    })

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=3)


def plot_results(accuracy_per_pos, K, file_path):
    """
    Plot a heatmap of the accuracy per position and save it to a file.
    """
    heatmap = np.zeros((K, K))
    for pos, accuracy in accuracy_per_pos.items():
        i, j = map(int, pos.strip('()').split(','))
        heatmap[i, j] = accuracy

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.title('Accuracy per Grid Position')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xticks(np.arange(K))
    plt.yticks(np.arange(K))

    # Show the actual numbers on the heatmap
    for i in range(K):
        for j in range(K):
            plt.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center', color='white')

    # Save the heatmap to a file
    output_file_path = file_path.replace('.json', '.png')
    plt.savefig(output_file_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on a dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on")
    parser.add_argument("--K", type=int, default=3, help="Size of the grid (KxK)")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of grids to evaluate")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    args.output_file = "results/" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
    args.multimodal_data = "multimodal" in args.dataset_path
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        model_config = json.load(f)
        args.multimodal_model = model_config.get("multimodal", False)
    
    if args.multimodal_data:
        logger.info("Evaluating on multimodal dataset")

    # Load model, tokenizer, and dataset
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.multimodal_model)
    dataset = load_dataset(args.dataset_path)

    if not args.multimodal_model and args.multimodal_data:
        logger.warning("Multimodal data provided but language only model provided. Evaluating with language only data.")
        
    # Evaluate the model on the dataset
    results = evaluate_model_on_dataset(model, tokenizer, dataset, args.split, args.K, args.num_samples, args.multimodal_data, args.multimodal_model, args.debug)

    # Save the results
    save_results(results, args.model_path, args.dataset_path, args.output_file)
    plot_results(results["accuracy_per_pos"], args.K, args.output_file)

    logger.info("Final accuracy: {:.3f}".format(results["accuracy"]))

if __name__ == "__main__":
    main()
