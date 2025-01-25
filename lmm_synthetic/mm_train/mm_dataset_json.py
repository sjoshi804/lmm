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

def load_params_from_json(json_file_path: str) -> Dict:
    """Load parameters from a JSON file."""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    with open(json_file_path, "r") as f:
        content = f.read()
        print(f"JSON content: {content}")  # Debug: Print file content
        params = json.loads(content)  # Use json.loads to parse string
    return params


class LazySupervisedDataset(Dataset):
    """Dataset for multimodal supervised fine-tuning

    Args:
        data_path (str): Path to the dataset.
        split (str): Dataset split (e.g., 'train', 'test').
        max_data_size (int, optional): Maximum number of data samples to load. Defaults to -1 (load all).
        vision_token_ablation (bool, optional): Whether to perform vision token ablation. Defaults to False.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        alignment (bool, optional): Whether to concatenate everything into reponse for alignment training
        image_grid (bool, optional): Whether to have dataset only include image and text grid
        sub_sampling (bool, optional): Whether to subsample the conversations
        num_samples (int, optional): Number of conversations to subsample
        distinct_image(bool, optional): Whether to limit the number of unique images to show 
        num_distinct_img (float, optional): Percent of unique images to show from each subset
        num_questions (int, optional): Number questions to show for each image

    """

    def __init__(
        self, 
        data_path: str, 
        split: str,
        max_data_size: int = -1,
        vision_token_ablation: bool = False,
        debug: bool = False,
        alignment: bool = False,
        image_grid: bool = False,
        sub_sampling: bool = False,
        num_samples: int = 3,
        distinct_image: bool = False,
        num_distinct_img: float = 0.5,
        num_distinct_questions: int = 2

    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.debug = debug
        self.vision_token_ablation = vision_token_ablation

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

def create_dataset_from_json(json_file_path: str):
    """Create a LazySupervisedDataset instance from a JSON file."""
    params = load_params_from_json(json_file_path)
    required_params = ["data_path", "split"]
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter '{param}' in JSON file.")
    return LazySupervisedDataset(**params)


#original_alignment = LazySupervisedDataset(r"/home/allanz/data/datasets/spuco/test/multimodal_dataset", "train", 10, False, True, True, False, False, 0, True, 0.5, 3)


if __name__ == "__main__":
    # Path to the JSON file holding parameters
    json_file_path = "/home/allanz/lmm/lmm_synthetic/mm_train/dataset_configs/test.json"

    # Create dataset instance using the JSON file
    dataset = create_dataset_from_json(json_file_path)
    print(f"Dataset size: {len(dataset)}")