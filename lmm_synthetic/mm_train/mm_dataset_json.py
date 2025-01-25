from typing import Dict, List
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import math
import json
import os

from lmm_synthetic.data.convert_to_multimodal import parse_grid

def find_text(text, char, index):
    count = 0
    for i in range(len(text)):
        if text[i] == char:
            count += 1
            if count == index:
                return i 

class LazySupervisedDataset(Dataset):
    """Dataset for multimodal supervised fine-tuning

    Args:
        json_file_path (str): Path to the JSON configuration file.
    """

    def __init__(
        self, 
        json_file_path: str
    ) -> None:
        super(LazySupervisedDataset, self).__init__()

        # Load parameters from JSON file
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        with open(json_file_path, "r") as f:
            params = json.load(f)

        # Extract parameters
        self.debug = params.get("debug", False)
        self.vision_token_ablation = params.get("vision_token_ablation", False)
        max_data_size = params.get("max_data_size", -1)
        alignment = params.get("alignment", False)
        image_grid = params.get("image_grid", False)
        sub_sampling = params.get("sub_sampling", False)
        num_samples = params.get("num_samples", 3)
        distinct_image = params.get("distinct_image", False)
        num_distinct_img = params.get("num_distinct_img", 0.5)
        num_distinct_questions = params.get("num_distinct_questions", 2)
        data_path = params.get("data_path", "")
        split = params.get("split", "train")

        # Validate required parameters
        if not data_path or not split:
            raise ValueError("JSON file must contain 'data_path' and 'split' parameters.")

        # Load the dataset from disk
        hf_dataset = load_from_disk(data_path)[split]
        self.list_data_dict = []

        # Image grid is already suited for alignment training
        if image_grid == True:
            for sample in hf_dataset:
                prompt = sample.get("prompt", "")
                grid_index = find_text(sample.get("text", ""), "\n", 3)
                grid = sample.get("text", "")[0:grid_index]
                conversations = [["", grid]]
                data_dict = {
                    "image": sample.get("image", None),
                    "prompt": prompt,
                    "conversations": conversations
                }
                if self.debug:
                    data_dict["text"] = sample.get("text", "")
                if self.vision_token_ablation:
                    data_dict["grid"] = sample.get('grid', parse_grid(sample['text']))
                self.list_data_dict.append(data_dict)            

        else:
            count = 0
            if distinct_image == True:
                total = len(hf_dataset)
                subset = hf_dataset.select(range(0, int(total * num_distinct_img)))
                for sample in subset: 
                    prompt = sample.get("prompt", "")
                    if alignment == True:
                        for i in range(math.ceil(1/num_distinct_img)):
                            if count == total:
                                break       
                            conversations = []
                            response = ""
                            for entry in random.sample(sample["conversations"], num_distinct_questions):
                                for subentry in entry:
                                    response += subentry
                                response += "\n"
                            conversations.append(["", response])
                            data_dict = {
                                "image": sample.get("image", None),
                                "prompt": prompt,
                                "conversations": conversations
                            }
                            count += 1

                    else: 
                        for i in range(math.ceil(1/num_distinct_img)):
                            if count == total:
                                break 
                            conversations = random.sample(sample["conversations"], num_distinct_questions)
                            data_dict = {
                                "image": sample.get("image", None),
                                "prompt": prompt,
                                "conversations": conversations
                            }
                            count += 1
                    
                    if self.debug:
                        data_dict["text"] = sample.get("text", "")
                    if self.vision_token_ablation:
                        data_dict["grid"] = sample.get('grid', parse_grid(sample['text']))
                    self.list_data_dict.append(data_dict)  
            else:
                for sample in hf_dataset:
                    prompt = sample.get("prompt", "")
                    if alignment == True:
                        if sub_sampling == True:
                            conversations = []
                            response = ""
                            for entry in random.sample(sample["conversations"], num_samples):
                                for subentry in entry:
                                    response += subentry
                                response += "\n"
                            conversations.append(["", response]) 
                        else:
                            conversations = []
                            response = ""
                            for i in range(len(sample["conversations"])):
                                for j in range(len(sample["conversations"][i])):
                                    if j % 2 == 1:
                                        response += sample["conversations"][i][j] + "\n"
                                    else:
                                        response += sample["conversations"][i][j]
                            conversations.append(["", response])
                    else:
                        if sub_sampling == True:
                            conversations = random.sample(sample["conversations"], num_samples)
                        else:
                            conversations = sample.get("conversations", [])
                    data_dict = {
                        "image": sample.get("image", None),
                        "prompt": prompt,
                        "conversations": conversations
                        }
                    if self.debug:
                        data_dict["text"] = sample.get("text", "")
                    if self.vision_token_ablation:
                        data_dict["grid"] = sample.get('grid', parse_grid(sample['text']))
                    self.list_data_dict.append(data_dict)

        # Limit the dataset size if max_data_size is specified
        if max_data_size > 0:
            self.list_data_dict = self.list_data_dict[:max_data_size]

        logger.info(f"Dataset size: {len(self.list_data_dict)}")

        # Determine whether each sample is text-only
        self.is_text_only = [
            "image" not in source for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:
        """Retrieves the sample at index `i`.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Dict[str, List]: A dictionary containing the sample data.
        """
        sample = self.list_data_dict[i]
        item_dict = {
            "image": Image.open(sample["image"]).convert("RGB"),
            "prompt": sample["prompt"],
            "conversations": sample["conversations"]
        }
        if self.debug:
            item_dict["text"] = sample["text"]
        if self.vision_token_ablation:
            item_dict["grid"] = sample["grid"]
        return item_dict


if __name__ == "__main__":
    # Path to the JSON file holding parameters
    json_file_path = "/home/allanz/lmm/lmm_synthetic/mm_train/dataset_configs/test.json"

    # Create dataset instance using the JSON file
    dataset = LazySupervisedDataset(json_file_path)
    print(f"Dataset size: {len(dataset)}")