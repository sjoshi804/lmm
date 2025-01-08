from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from PIL import Image
import random 
import os

# Necessary function for later
def parse_grid(grid_str, K):
    """
    Parse the grid string into a 2D list of grid cells.
    """
    grid_str = '\n'.join(grid_str.split('\n')[:K])
    rows = grid_str.strip().split('\n')
    return [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows]

# Saves images of each class, images will be used for grid creation
def save_cifar10(output_directory):
    """
    Loads in CIFAR-10, saves images of each type to different folder
    """
    datasets = load_dataset("cifar10")
    output_folder = output_directory + "/{class_name}"
    # Get the label names and their corresponding IDs
    label_names = datasets["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(label_names):
        label2id[label] = i
        id2label[i] = label 
    # Initialize dict to store indexes of each class
    class_index = {}
    for label in label_names:
        class_index[label] = []
    # Create list of indexes for each class
    labels = datasets['train']['label']
    for i in range(len(labels)):
        class_name = id2label[labels[i]]
        class_index[class_name].append(i)
    # Save each class of images to respective folder
    for name in label_names:
        print(f"Saving {name} to" + output_folder.format(class_name=name))
        count = 0
        for index in class_index[name]:
            image = datasets['train'][index]["img"]
            image.save(os.path.join(output_folder, f"{count}.png"))
            count += 1

        print(f"Images for class {name} saved successfully!")


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
            if type == "train":
                i = random.randint(0, num_unique_images)
            elif type == "validation":
                i = random.raindint(2000, 2000 + num_unique_images)
            elif type == "test":
                i = random.raindint(4000, 4000 + num_unique_images)
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
        img.save(f"/home/allanz/data/grid/{set_type}/{count}.png")
        count += 1
        print(f"Grid {count} for {set_type} saved successfully!")
    print(f"Grids for {set_type} saved successfully!")


# Copy old dataset, change image path
def copy(type, old_dataset_path):
    dataset = load_from_disk(old_dataset_path)
    data = []
    for i in range(len(dataset[type])):
        temp = {'text' : dataset[type][i]['text'], 'prompt' : dataset[type][i]['prompt'], 
        'conversations' : dataset[type][i]['conversations'], 'image' : f'/home/allanz/data/grid/{type}/{i}.png'}
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


#save_cifar10("/home/allanz/data/images")
#save_merged_grid('/data/lmm/generated/v3_spatial_grid_multimodal', "train",  1000, 0, 100000)
#save_merged_grid('/data/lmm/generated/v3_spatial_grid_multimodal', "validation", 1000, 0, 1000)
#save_merged_grid('/data/lmm/generated/v3_spatial_grid_multimodal', "test", 1000, 0, 1000)
create_new_dataset('/data/lmm/generated/v3_spatial_grid_multimodal','/home/allanz/data/datasets/v3.1_spatial_grid_multimodal')
