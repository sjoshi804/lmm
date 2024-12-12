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

def compute_vision_embeds(model, image_tensors):
    # Ensure the model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Generate responses for the entire batch
    with torch.no_grad():
        image_embeds = model.vision_encoder(image_tensors.to(device))

    # Decode the generated outputs for the batch

    return image_embeds

def construct_linear_probe_data(model, tokenizer, dataset, split, K, num_samples=250, multimodal_data=False, multimodal_model=False, debug=False):
    """
    Collect hidden states and corresponding labels for a linear probe.
    """
    num_grids = 0
    
    logger.info(f"Running on {num_samples} grids")
    pbar = tqdm(enumerate(dataset[split]), total=num_samples)
    _, image_transforms, _ = load_vision_encoder("clip")
    
    for i, example in pbar:
        if num_grids == num_samples:
            break
        
        num_grids += 1
        
        if multimodal_model:
            if multimodal_data:
                image_tensor = image_transforms(Image.open(example["image"])).unsqueeze(0)
                image_tensors = [image_tensor for _ in range(K * K)]
                image_tensors = torch.cat(image_tensors, dim=0)
                responses = compute_vision_embeds(model, tokenizer, image_tensors, position_prompts, max_new_tokens=5)
            else:
                responses = compute_vision_embeds(model, tokenizer, None, position_prompts, max_new_tokens=5)
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
    
    return inputs, labels


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
    results = construct_linear_probe_data(model, tokenizer, dataset, args.split, args.K, args.num_samples, args.multimodal_data, args.multimodal_model, args.debug)

    # Save the results
    save_results(results, args.model_path, args.dataset_path, args.output_file)
    plot_results(results["accuracy_per_pos"], args.K, args.output_file)

    logger.info("Final accuracy: {:.3f}".format(results["accuracy"]))

if __name__ == "__main__":
    main()
