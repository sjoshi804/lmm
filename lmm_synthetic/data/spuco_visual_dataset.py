from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from PIL import Image
import random 
import os
import re


def reformat(old):
    """
    Function to turn dataset of form {'train: '', grid: ''} into {'text': '', 'prompt': '', 
    'conversations': '', 'image': ''}
    Args:
        old (dict): The old dataset
    Returns:
        new (dict): The new dataset
    """

    new = {}
    new['text'] = old['text']
    text = old['text']
    prompt_pattern = r"(The grid above.*\.)"
    prompt_match = re.search(prompt_pattern, text)
    prompt = prompt_match.group(0) if prompt_match else "Prompt not found"
    new['prompt'] = prompt
    conversations_pattern = r"(What object.*?A: .+)"
    conversations_raw = re.findall(conversations_pattern, text)
    conversations = [pair.split('A: ') for pair in conversations_raw]
    new['conversations'] = conversations
    new['image'] = "temp"
    return new

def format_dataset(dataset_path):
    """
    Function to create new, reformatted dataset
    Args:
        dataset (str): Path to new dataset
    """
    dataset = load_from_disk(dataset_path)
    new_data = {split: [reformat(entry) for entry in entries] for split, entries in dataset.items()}

    new_dataset = DatasetDict({
        "train": Dataset.from_dict({
            "text": [entry["text"] for entry in new_data["train"]],
            "prompt": [entry["prompt"] for entry in new_data["train"]],
            "conversations": [entry["conversations"] for entry in new_data["train"]],
            "image": [entry["image"] for entry in new_data["train"]]
        }),
        "test": Dataset.from_dict({
            "text": [entry["text"] for entry in new_data["test"]],
            "prompt": [entry["prompt"] for entry in new_data["test"]],
            "conversations": [entry["conversations"] for entry in new_data["test"]],
            "image": [entry["image"] for entry in new_data["test"]]
        }),
        "validation": Dataset.from_dict({
            "text": [entry["text"] for entry in new_data["validation"]],
            "prompt": [entry["prompt"] for entry in new_data["validation"]],
            "conversations": [entry["conversations"] for entry in new_data["validation"]],
            "image": [entry["image"] for entry in new_data["validation"]]
        })
    })
    return new_dataset

# Necessary function for later
def parse_grid(grid_str, K):
    """
    Parse the grid string into a 2D list of grid cells.
    """
    grid_str = '\n'.join(grid_str.split('\n')[:K])
    rows = grid_str.strip().split('\n')
    return [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows]


# Creates merged grid
def merge_image(grid, final_size=(256, 256), num_unique_images = 1000, border_size = 6, set_type = "train"):
    """
    Creates 3x3 image grid from 3x3 grid of words, returns the image grid
    """
    BORDER_SIZE = border_size
    # Determine the number of rows and columns
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Calculate available width and height for each image
    total_border_width = (cols - 1) * BORDER_SIZE
    total_border_height = (rows - 1) * BORDER_SIZE
    available_width = final_size[0] - total_border_width
    available_height = final_size[1] - total_border_height
    element_width = available_width // cols
    element_height = available_height // rows
    
    # Create the new image with a black background
    combined_image = Image.new('RGB', final_size, 'black')
    
    # Place each image in the combined image
    for row_index, row in enumerate(grid):
        for col_index, word in enumerate(row):
            if set_type == "train":
                i = random.randint(0, num_unique_images)
            elif set_type == "validation":
                i = random.randint(2000, 2000 + num_unique_images)
            elif set_type == "test":
                i = random.randint(4000, 4000 + num_unique_images)
            else:
                raise ValueError(f"Unknown type: {type}")
            # Change path if necessary
            img = Image.open(f"/home/allanz/data/images/{word}/{i}.png").resize((element_height, element_width), Image.Resampling.LANCZOS)
            x = col_index * (element_width + BORDER_SIZE)
            y = row_index * (element_height + BORDER_SIZE)
            combined_image.paste(img, (x, y))
    
    return combined_image


def save_merged_grid(previous_dataset_path, set_type, num_unique_images, start, last):
    """
    Saves new grids to be used in new dataset

    Parameters:
    - previous_dataset: DatasetDict - The dataset to get the grids from
    - set_type: str - The type of dataset to get the grids from (train, validation, test)
    - num_unique_images: int - The number of unique images from each class to sample from
    - start: int - The starting index of the grids to save
    - last: int - The ending index of the grids to save
    
    Returns:
    - None
    """
    # Load the previous dataset, create subset of grids to save
    previous_dataset = load_from_disk(previous_dataset_path)
    subset = previous_dataset[set_type].select(range(start, last))
    grids = subset["text"]
    count = start 
    for grid in grids:
        temp_grid = parse_grid(grid, 3)
        img = merge_image(temp_grid, num_unique_images = num_unique_images)
        img.save(f"/home/allanz/data/grid/spuco/{set_type}/{count}.png")
        count += 1
        print(f"Grid {count} for {set_type} saved successfully!")
    print(f"Grids for {set_type} saved successfully!")


# Copy old dataset, change image path
def copy(type, old_dataset_path):
    dataset = load_from_disk(old_dataset_path)
    data = []
    for i in range(len(dataset[type])):
        temp = {'text' : dataset[type][i]['text'], 'prompt' : dataset[type][i]['prompt'], 
        'conversations' : dataset[type][i]['conversations'], 'image' : f'/home/allanz/data/grid/spuco/{type}/{i}.png'}
        data.append(temp)
        print(f"Grid {i} for {type} copied successfully!")
    return data

# Convert list of dictionaries to dictionary of lists
def convert_to_dict_of_lists(data):
    result = {}
    for key in data[0].keys():  
        result[key] = [entry[key] for entry in data]
    return result

# Save new dataset
def create_new_dataset(old_dataset_path, save_path):
    train = copy('train', old_dataset_path)
    validation = copy('validation', old_dataset_path)
    test = copy('test', old_dataset_path)
    print("Successfully copied old dataset, chaning image paths")
    
    # Convert 'train', 'validation', and 'test' lists into dicts of lists
    train_dict = convert_to_dict_of_lists(train)
    validation_dict = convert_to_dict_of_lists(validation)
    test_dict = convert_to_dict_of_lists(test)
    print("Successfully converted lists to dictionaries")

    # Create the datasets using from_dict
    train_dataset = Dataset.from_dict(train_dict)
    validation_dataset = Dataset.from_dict(validation_dict)
    test_dataset = Dataset.from_dict(test_dict)
    print("Successfully created datasets")

    # Save the datasets
    dataset_dict = DatasetDict({'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset})
    dataset_dict.save_to_disk(save_path)
    print("Successfully saved new dataset")


#save_merged_grid('/home/allanz/data/datasets/spuco/text_reformatted', "train",  1000, 0, 100000)
#save_merged_grid('/home/allanz/data/datasets/spuco/text_reformatted', "validation", 1000, 0, 1000)
#save_merged_grid('/home/allanz/data/datasets/spuco/text_reformatted', "test", 1000, 0, 1000)
#create_new_dataset('/home/allanz/data/datasets/spuco/text_reformatted','/home/allanz/data/datasets/spuco_multimodal')
