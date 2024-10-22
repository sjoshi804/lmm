import argparse
import json
import os
import random
import time
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm
from PIL import Image, ImageDraw


def merge_image(grid, final_size=(256, 256)):
    # Determine the number of rows and columns
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Set border (gap) size between images
    border_size = 10
    
    # Calculate available width and height for each image
    total_border_width = (cols - 1) * border_size
    total_border_height = (rows - 1) * border_size
    available_width = final_size[0] - total_border_width
    available_height = final_size[1] - total_border_height
    element_width = available_width // cols
    element_height = available_height // rows
    
    # Load and resize all images to fit within the calculated element size
    images = [[Image.open(path).resize((element_width, element_height), Image.Resampling.LANCZOS) 
               for path in row] for row in grid]
    
    # Create the new image with a black background
    combined_image = Image.new('RGB', final_size, 'black')
    
    # Place each image in the combined image
    for row_index, row in enumerate(images):
        for col_index, img in enumerate(row):
            x = col_index * (element_width + border_size)
            y = row_index * (element_height + border_size)
            combined_image.paste(img, (x, y))
    
    return combined_image
    # Determine the number of rows and columns
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Set border size
    border_size = 10
    
    # Calculate available width and height for each image
    total_border_width = (cols - 1) * border_size
    total_border_height = (rows - 1) * border_size
    available_width = final_size[0] - total_border_width
    available_height = final_size[1] - total_border_height
    element_width = available_width // cols
    element_height = available_height // rows
    
    # Load and resize all images to fit within the calculated element size
    images = [[Image.open(path).resize((element_width, element_height), Image.Resampling.LANCZOS) 
               for path in row] for row in grid]
    
    # Create the new image with white background
    combined_image = Image.new('RGB', final_size, 'black')
    draw = ImageDraw.Draw(combined_image)
    
    # Place each image in the combined image
    for row_index, row in enumerate(images):
        for col_index, img in enumerate(row):
            x = col_index * (element_width + border_size)
            y = row_index * (element_height + border_size)
            combined_image.paste(img, (x, y))
            
            # Draw the black border around the image if it's not the last column/row
            if col_index < cols - 1:
                draw.rectangle([x + element_width, y, x + element_width + border_size, y + element_height], fill='black')
            if row_index < rows - 1:
                draw.rectangle([x, y + element_height, x + element_width, y + element_height + border_size], fill='black')

    return combined_image

def convert_to_multimodal(args):
    logger.info("Starting conversion to multimodal dataset.")
    
    # Load json from dataset config path
    logger.info(f"Loading config from {args.config_path}.")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Load huggingface language dataset using load_from_disk huggingface function
    logger.info(f"Loading dataset from {args.dataset_path}.")
    dataset = load_from_disk(args.dataset_path)
    
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
    multimodal_dataset_path = os.path.join(args.output_dir, os.path.basename(args.config_path).split('.')[0] + '_multimodal_' + timestamp)
    os.makedirs(multimodal_dataset_path, exist_ok=True)
    
    # Create an images subfolder
    images_folder = os.path.join(multimodal_dataset_path, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    # For each split in dataset
    logger.info("Processing dataset splits.")
    for split in dataset.keys():
        new_split = []
        for id, sample in tqdm(enumerate(dataset[split]), total=len(dataset[split]), desc=f"Processing {split} split"):
            # If "grid" attribute present, use this as the grid. If not provided, parse the 2d grid from "text" attribute
            grid = sample.get('grid', parse_grid_from_text(sample['text']))
            
            # Map each object to an image from corresponding class from image pool
            for i, row in enumerate(grid):
                for j, word in enumerate(row):
                    image_name = random.choice(sampled_image_pool[word])
                    grid[i][j] = os.path.join(image_pool[word], image_name)
            
            # Merge the images for the grid to create 1 image
            merged_image_path = os.path.join(images_folder, f"{split}_{id}.png")
            merge_image(grid, tuple(config["image_size"])).save(merged_image_path)
            
            # Replace the grid with the <image_1> tag in the text
            sample['text'] = sample['text'].replace(str(grid), '<image_1>')
            sample['image_1'] = merged_image_path
            
            new_split.append(sample)
        
        # Save the new split
        split_path = os.path.join(multimodal_dataset_path, f"{split}.json")
        logger.info(f"Saving new split to {split_path}.")
        with open(split_path, 'w') as f:
            json.dump(new_split, f)
    
    logger.info("Conversion to multimodal dataset completed.")

def parse_grid_from_text(grid_str):
    grid_str = '\n'.join(grid_str.split('\n')[:3])
    rows = grid_str.strip().split('\n')
    grid = [row.strip().split('|') for row in rows]
    # Remove any empty strings resulting from splitting and strip each element
    return [[cell.strip() for cell in row if cell.strip()] for row in grid]

if __name__ == "__main__":
    logger.info("Parsing arguments.")
    parser = argparse.ArgumentParser(description="Convert a dataset to a multimodal dataset.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Huggingface dataset.')
    parser.add_argument('--output_dir', type=str, required=False, default="/home/sjoshi/lmm/data/generated/", help='Path to save the generated dataset.')

    args = parser.parse_args()

    convert_to_multimodal(args)