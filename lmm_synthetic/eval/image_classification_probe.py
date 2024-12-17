import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from lmm_synthetic.mm_train.utils import load_vision_encoder
from lmm_synthetic.mm_train.gptj_vlm import GPTJ_VLM

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    model, tokenizer = None, None
    model = GPTJ_VLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model.config.pretrained_lm_path)
    model.eval()
    return model, tokenizer

def load_dataset(dataset_path):
    """
    Load the dataset from the specified path.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    return load_from_disk(dataset_path)

def parse_grid(grid_str, K):
    """
    Parse the grid string into a 2D list of grid cells.
    """
    grid_str = '\n'.join(grid_str.split('\n')[:K])
    rows = grid_str.strip().split('\n')
    return [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows]

def train_linear_probe(model, tokenizer, dataset, K, num_train=250):
    """
    Train Linear Probe
    """
    num_grids = 0

    pbar = tqdm(dataset["train"], total=num_train, desc="Collecting inputs and labels for linear probe")
    
    _, image_transforms, _ = load_vision_encoder("clip")
    
    for example in pbar:
        if num_grids == num_train:
            break
        
        num_grids += 1
        
        text_prompt = example["text"].split(']')[0] + '].'
        grid = parse_grid(text_prompt, K)

        prompt = example["prompt"]
        
        image_tensor = image_transforms(Image.open(example["image"])).unsqueeze(0)
        image_tokens = model.multimodal_projector(model.vision_encoder(image_tensor))
        print(image_tokens.shape)
        print(image_tokens)
        break
    
    return None 

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
    parser = argparse.ArgumentParser(description="Train and Evaluate Linear Probe on Image Embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--K", type=int, default=3, help="Size of the grid (KxK)")
    parser.add_argument("--num_train", type=int, default=10000, help="Number of samples to train linear probe")

    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    args.output_file = "results/" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
    
    # Load model, tokenizer, and dataset
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    dataset = load_dataset(args.dataset_path)
    
    linear_probe = train_linear_probe(model, tokenizer, dataset, args.K, args.num_train)
    #results = evaluate_linear_probe(model, tokenizer, dataset, args.K)

    # Save the results
    #save_results(results, args.model_path, args.dataset_path, args.output_file)

if __name__ == "__main__":
    main()
