import argparse
import json
import os
import random
from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm
from PIL import Image
import re
from datasets import DatasetDict, Dataset
import pandas as pd
from datasets import load_dataset


# Load CIFAR-10 dataset
datasets = load_dataset("cifar10")

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

# Load your image from the dataset
image = datasets['train'][0]["img"]

# Specify the path to the folder
output_folder = '/home/allanz/data/images/{class_name}'

for name in label_names:
    # Set correct output directory path
    output_directory = output_folder.format(class_name=name)
    print(f"Saving images to {output_directory}")
    count = 0
    for index in class_index[name]:
        image = datasets['train'][index]["img"]
        image.save(os.path.join(output_directory, f"{count}.png"))
        count += 1
    
    print(f"Images for class {name} saved successfully!")


