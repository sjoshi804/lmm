import argparse
import re
import json
from datetime import datetime
from loguru import logger
from tqdm import tqdm
import torch
from transformers import GPTJForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
import os
import matplotlib.pyplot as plt


def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPTJForCausalLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def load_dataset(dataset_path):
    """
    Load the dataset from the specified path.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    return load_from_disk(dataset_path)


def generate_texts(model, tokenizer, prompts, max_new_tokens=50, max_length=512):
    """
    Generate texts from the model for a batch of prompts.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

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


def evaluate_model_on_dataset(model, tokenizer, dataset, K, multimodal=False, num_samples=250):
    """
    Evaluate the model on the given dataset and compute accuracy.
    """
    num_grids = 0
    total_per_pos = {}
    correct_per_pos = {}

    logger.info(f"Starting evaluation on {num_samples} grids")
    pbar = tqdm(dataset['validation'], total=num_samples)

    for example in pbar:
        num_grids += 1
        prompt = example["text"].split(']')[0] + '].'
        grid = parse_grid(prompt, K)

        # Generate prompts for all positions in the grid
        position_prompts = [
            prompt + f"\nWhat object is in row {i}, column {j}?"
            for i in range(K) for j in range(K)
        ]
        generated_texts = generate_texts(model, tokenizer, position_prompts, max_new_tokens=5)

        # Evaluate predictions
        for idx, text in enumerate(generated_texts):
            i, j = divmod(idx, K)
            total_per_pos[(i, j)] = total_per_pos.get((i, j), 0) + 1
            parsed_answer = parse_answer(text)
            if parsed_answer == grid[i][j]:
                correct_per_pos[(i, j)] = correct_per_pos.get((i, j), 0) + 1

        accuracy = sum(correct_per_pos.values()) / sum(total_per_pos.values())
        pbar.set_description(f'Accuracy: {accuracy:.3f}')

        if num_grids == num_samples:
            break

    accuracy_per_pos = {pos: correct_per_pos[pos] / total_per_pos[pos] for pos in total_per_pos}
    return accuracy_per_pos, accuracy


def save_results(accuracy_per_pos, accuracy, model_path, dataset_path, file_path):
    """
    Save results to a JSON file.
    """
    results = {
        "accuracy_per_pos": {str(pos): acc for pos, acc in accuracy_per_pos.items()},
        "accuracy": accuracy,
        "model_path": model_path,
        "dataset_path": dataset_path
    }

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=3)


def plot_results(accuracy_per_pos, K, file_path):
    """
    Plot a heatmap of the accuracy per position and save it to a file.
    """
    heatmap = np.zeros((K, K))
    for (i, j), accuracy in accuracy_per_pos.items():
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
    parser.add_argument("--K", type=int, default=5, help="Size of the grid (KxK)")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of grids to evaluate")

    args = parser.parse_args()
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    args.output_file = "results/" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
    args.multimodal = "multimodal" in args.dataset_path

    # Load model, tokenizer, and dataset
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    dataset = load_dataset(args.dataset_path)

    # Evaluate the model on the dataset
    accuracy_per_pos, accuracy = evaluate_model_on_dataset(model, tokenizer, dataset, args.K, args.num_samples)

    # Save the results
    save_results(accuracy_per_pos, accuracy, args.model_path, args.dataset_path, args.output_file)
    plot_results(accuracy_per_pos, args.K, args.output_file)

    logger.info(f"Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
