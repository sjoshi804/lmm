from convert_to_multimodal import *
from data_generator import *
import argparse
import json 

#args = {num_samples, num_questions, num_rows, num_cols, vocab, vocab_subset_size, spuco, position, correlation, label, save_path = "/home/allanz/data/datasets/spuco/text_dataset"}

def create(num_samples, num_questions, num_rows, num_cols, vocab, vocab_subset_size, spuco, position, correlation, label, text_save_path, multimodal_save_path, unique_images):
    """
        Creates a synthetic multimodal dataset
    
        Args:
            num_samples (dict): General form of dataset
                ex) {"train": 100, "validation": 1, "test": 1}
            num_questions (int): Number of questions to ask about the grid
            num_rows (int): Number of rows in the grid
            num_cols (int): Number of columns in the grid
            vocab (list): List of vocabulary words (currently cifar10 labels)
            vocab_subset_size (int): Size of the subset of the vocabulary to use to create a grid
            spuco (bool): Flag to indicate whether spurious feature should be included
            position (tuple): Position of the spurious label in the grid
            correlation (float): Frequency by which label appears in position
                ex) 0.9 = 90% frequency
            label (str): Label of spurious feature, defaults to "dog"
            text_save_path (str): Path to save the text dataset
            multimodal_save_path (str): Path to save the multimodal dataset
            unique_images (bool): Flag to ensure unique images in the dataset 
        
        Returns:
            None      
    """   
    
    create_dataset(num_samples, num_questions, num_rows, num_cols, vocab, vocab_subset_size, spuco, position, correlation, label, text_save_path)    
    
    reformatted_dataset = format_dataset(text_save_path)
    if spuco == True:
        print("Spuco is true")
        save_merged_grid(reformatted_dataset, "train", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["train"]), spuco = True)
        save_merged_grid(reformatted_dataset, "test", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["test"]), spuco = True) 
        save_merged_grid(reformatted_dataset, "validation", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["validation"]), spuco = True)
        create_new_dataset(reformatted_dataset, multimodal_save_path, spuco = True)
    else:
        print("Spuco is false")
        save_merged_grid(reformatted_dataset, "train", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["train"]), spuco = False)
        save_merged_grid(reformatted_dataset, "test", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["test"]), spuco = False) 
        save_merged_grid(reformatted_dataset, "validation", num_unique_images = unique_images, start = 0, last = len(reformatted_dataset["validation"]), spuco = False)
        create_new_dataset(reformatted_dataset, multimodal_save_path, spuco = False)

# Example of a call to create a spuco dataset
create(num_samples = {"train": 100, "validation": 1, "test": 1}, num_questions = 9,  num_rows = 3,  num_cols = 3,  vocab = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], vocab_subset_size = 4, spuco = True, position = (0, 0), correlation = 9, label ="dog", text_save_path = "/home/allanz/data/datasets/spuco/test/text_dataset", multimodal_save_path = "/home/allanz/data/datasets/spuco/test/multimodal_dataset", unique_images = 1000)



def create_from_json(json_file_path: str):
    """
    Reads parameters from a JSON file and calls the create function.
    """
    # Debugging: Print the file path
    print(f"Reading JSON file from: {json_file_path}")

    # Check if the file exists and is not empty
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")
    if os.path.getsize(json_file_path) == 0:
        raise ValueError(f"The file {json_file_path} is empty.")

    # Debugging: Print the file content
    with open(json_file_path, 'r') as f:
        content = f.read()
        print("File content:", content)  # Debugging: Print the file content
        config = json.loads(content)

    # Call the create function with the parameters from the JSON file
    create(
        num_samples=config['num_samples'],
        num_questions=config['num_questions'],
        num_rows=config['num_rows'],
        num_cols=config['num_cols'],
        vocab=config['vocab'],
        vocab_subset_size=config['vocab_subset_size'],
        spuco=config['spuco'],
        position=tuple(config['position']),  # Convert list to tuple
        correlation=config['correlation'],
        label=config['label'],
        text_save_path=config['text_save_path'],
        multimodal_save_path=config['multimodal_save_path'],
        unique_images=config['unique_images']
    )



if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create dataset from JSON configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Call the function to create the dataset
    create_from_json(args.config)
