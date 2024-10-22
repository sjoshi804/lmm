import argparse
import json
import os
import random
import time
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm

def convert_to_multimodal(config_path, dataset_path, save_path):
    logger.info("Starting conversion to multimodal dataset.")
    
    # Load json from dataset config path
    logger.info(f"Loading config from {config_path}.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load huggingface language dataset using load_from_disk huggingface function
    logger.info(f"Loading dataset from {dataset_path}.")
    dataset = load_from_disk(dataset_path)
    
    # Pull "image_pool" dict from config
    image_pool = config['image_pool']
    
    # "num_unique_images_per_class": Num unique images to use per class as path: pull from config
    num_unique_images_per_class = config['num_unique_images_per_class']
    
    # Initialize an image pool dictionary corresponding to image pool from config
    logger.info("Sampling images for each class.")
    sampled_image_pool = {}
    for word, folder in image_pool.items():
        images = os.listdir(folder)
        sampled_image_pool[word] = random.sample(images, num_unique_images_per_class)
    
    # Path for multimodal dataset
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    multimodal_dataset_path = os.path.join(save_path, os.path.basename(config_path).split('.')[0] + '_multimodal_' + timestamp)
    os.makedirs(multimodal_dataset_path, exist_ok=True)
    
    # Create an images subfolder
    images_folder = os.path.join(multimodal_dataset_path, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    # For each split in dataset
    logger.info("Processing dataset splits.")
    for split in dataset.keys():
        new_split = []
        for sample in tqdm(dataset[split], desc=f"Processing {split} split"):
            # If "grid" attribute present, use this as the grid. If not provided, parse the 2d grid from "prompt" attribute
            grid = sample.get('grid', parse_grid_from_prompt(sample['prompt']))
            
            # Map each object to an image from corresponding class from image pool
            for i, row in enumerate(grid):
                for j, word in enumerate(row):
                    image_name = random.choice(sampled_image_pool[word])
                    grid[i][j] = os.path.join(image_pool[word], image_name)
            
            # Merge the images for the grid to create 1 image
            merged_image_path = os.path.join(images_folder, f"{split}_{sample['id']}.png")
            merge_image(grid, merged_image_path)
            
            # Replace the grid with the <image_1> tag in the prompt
            sample['prompt'] = sample['prompt'].replace(str(grid), '<image_1>')
            sample['image_1'] = merged_image_path
            
            new_split.append(sample)
        
        # Save the new split
        split_path = os.path.join(multimodal_dataset_path, f"{split}.json")
        logger.info(f"Saving new split to {split_path}.")
        with open(split_path, 'w') as f:
            json.dump(new_split, f)
    
    logger.info("Conversion to multimodal dataset completed.")

def parse_grid_fro_prompt(grid_str):
    grid_str = '\n'.join(grid_str.split('\n')[:3])
    rows = grid_str.strip().split('\n')
    grid = [row.strip().split('|') for row in rows]
    # Remove any empty strings resulting from splitting and strip each element
    return [[cell.strip() for cell in row if cell.strip()] for row in grid]

def merge_image(grid, output_path):
    # Implement this function to merge images in the grid and save to output_path
    pass

if __name__ == "__main__":
    logger.info("Parsing arguments.")
    parser = argparse.ArgumentParser(description="Convert a dataset to a multimodal dataset.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Huggingface dataset.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the multimodal dataset.')

    args = parser.parse_args()

    convert_to_multimodal(args.config_path, args.dataset_path, args.save_path)