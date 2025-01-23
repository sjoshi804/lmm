import argparse
import json
import random
from datetime import datetime
from typing import List, Tuple

import datasets
import pandas as pd
from loguru import logger
from tqdm import trange
import os 

  
def create_grid(num_rows: int, num_cols: int, vocab: List[str], vocab_subset_size: int, spuco: bool, position: tuple, correlation: int, label = "dog") -> List[List[str]]:
    """
    Creates a grid with the specified number of rows and columns,
    randomly sampling objects from the provided vocabulary.

    Parameters:
    - num_rows: int - The number of rows in the grid.
    - num_cols: int - The number of columns in the grid.
    - vocab: List[str] - The vocabulary of objects to populate the grid.
    - vocab_subset_size: int - The size of the subset of the vocabulary to use.
    - spuco: bool - whether to create dataset with spurious correlation or not
    - label: str - the label to correlate with a certain position in the grid 
    - position: tuple - where to place the label in the grid 
    - correlation: input / 10 to give the frequency of the correlation to occur 
    Returns:
    - List[List[str]] - The generated grid.
    """
    grid = []
    x, y = position
    
    if random.choice([x for x in range(1,11)]) <= correlation:
        vocab_copy = vocab.copy()
        vocab_copy.remove(label)
        vocab_subset = random.sample(vocab_copy, vocab_subset_size - 1)
        vocab_subset.append(label)
        for i in range(num_rows):
            temp = []
            for j in range(num_cols):
                if i == x and j == y:
                    temp.append(label)
                else:
                    temp.append(random.choice(vocab_subset))
            grid.append(temp)
    else:
        vocab_subset = random.sample(vocab, vocab_subset_size)
        grid = [[random.choice(vocab_subset) for _ in range(num_cols)] for _ in range(num_rows)]
    return grid


def convert_grid_to_str(grid: List[List[str]]) -> str:
    """
    Converts a 2D grid into a formatted string.

    Parameters:
    - grid: List[List[str]] - The grid to convert.

    Returns:
    - str - The grid formatted as a string.
    """
    rows = ['| ' + ' | '.join(row) + ' |' for row in grid]
    return '\n'.join(rows)

def add_grid_instruction(grid: List[List[str]]) -> str:
    """
    Adds an instruction describing the grid.

    Parameters:
    - grid: List[List[str]] - The grid to describe.

    Returns:
    - str - The instruction describing the grid.
    """
    num_rows = len(grid)
    num_cols = len(grid[0])
    used_vocab_subset = set()
    for row in grid:
        for element in row:
            used_vocab_subset.add(element)
    used_vocab_subset = list(used_vocab_subset)
    return f"The grid above is size {num_rows} by {num_cols}. Each cell contains an object from {used_vocab_subset}."

def create_position_questions(grid: List[List[str]]) -> List[str]:
    """
    Adds questions about the position of each object in the grid.

    Parameters:
    - grid: List[List[str]] - The grid to generate questions for.

    Returns:
    - List[str] - The questions about the grid.
    """
    questions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            questions.append(f"What object is in row {i}, column {j}? " + f"A: {grid[i][j]}")
    return questions

def create_position_assertions(grid: List[List[str]]) -> List[str]:
    """
    Adds assertions about the position of each object in the grid.

    Parameters:
    - grid: List[List[str]] - The grid to generate questions for.

    Returns:
    - List[str] - The questions about the grid.
    """
    assertions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            assertions.append(f"row {i}, column {j}" + f"A: {grid[i][j]}")
    return assertions

def create_dataset_from_json(num_samples, num_questions, num_rows, num_cols, vocab, vocab_subset_size, spuco, position, correlation, label) -> datasets.DatasetDict:
    """
    Creates a synthetic text dataset based on parameters from a JSON file.

    Parameters:
    - args: argparse.Namespace - The arguments containing the path to the JSON configuration file.

    Returns:
    - datasets.DatasetDict - The Hugging Face DatasetDict object containing the synthetic dataset with splits.
    """


    # Validate parameters


    num_samples = dict(num_samples)
    num_train_samples = int(num_samples['train'])
    num_val_samples = int(num_samples['validation'])
    num_test_samples = int(num_samples['test'])
    total_samples = sum([num_train_samples, num_val_samples, num_test_samples])
    dt_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logger.info(f"Generating synthetic dataset with {num_samples} samples, {num_rows} rows, {num_cols} columns, and vocabulary: {vocab}")

    samples = []
    for _ in trange(total_samples, desc="Generating samples"):
        grid = create_grid(num_rows, num_cols, vocab, vocab_subset_size, spuco, position, correlation, label)
        sample_str = convert_grid_to_str(grid)
        sample_str += "\n" + add_grid_instruction(grid)
        for question in random.sample(create_position_questions(grid), num_questions):
            sample_str += "\n" + question
        samples.append({'text': sample_str, 'grid': grid})

    # Convert to a pandas DataFrame for easier dataset creation
    df = pd.DataFrame(samples)

    # Save dataset as a Hugging Face-friendly dataset
    dataset = datasets.Dataset.from_pandas(df)
    
    # Split dataset into train, validation, and test sets using the specified sample sizes
    train_test_split = dataset.train_test_split(test_size=(num_val_samples + num_test_samples) / total_samples)
    test_val_split = train_test_split['test'].train_test_split(test_size=num_test_samples / (num_val_samples + num_test_samples))
    
    # Combine splits into a DatasetDict
    dataset_dict = datasets.DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })
    
    logger.info(f"Generated dataset with {len(dataset_dict['train'])} training samples, {len(dataset_dict['validation'])} validation samples, and {len(dataset_dict['test'])} test samples.")
    
    # Ensure the output directory exists
    
    dataset_dir = "/home/allanz/data/datasets/spuco/text_dataset"
    
    # Save the entire dataset dictionary to disk
    dataset_dict.save_to_disk(dataset_dir)
    logger.info(f"Saved dataset to disk at: {dataset_dir}")

    return dataset_dict

create_dataset_from_json(
    {"train": 100000, "validation": 1000, "test": 1000},
    9,  # num_questions
    3,  # num_rows
    3,  # num_cols
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    4,  # vocab_subset_size
    True,  # spuco
    (0, 0),  # position
    9,  # correlation (90% chance)
    "dog"
)