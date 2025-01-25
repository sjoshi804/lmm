from typing import Dict, List

from datasets import load_from_disk
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import math
import json

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
        config_json: str

    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        with open(config_json, 'r') as f:
            config = json.load(f)
        

        self.debug = config.get("debug", False)
        self.vision_token_ablation = config.get("vision_token_ablation", False)
        data_path = config.get("data_path", None)
        split = config.get("split", None)
        max_data_size = config.get("max_data_size", -1)
        alignment = config.get("alignment", False)
        image_grid = config.get("image_grid", False)
        sub_sampling = config.get("sub_sampling", False)
        num_samples = config.get("num_samples", 3)
        distinct_image = config.get("distinct_image", False)
        num_distinct_img = config.get("num_distinct_img", 0.5)
        num_distinct_questions = config.get("num_distinct_questions", 2)

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
            if distinct_image == True:
                if alignment == True:
                    subset = hf_dataset.select(range(0, int(max_data_size * num_distinct_img)))
                    count = 0
                    for sample in subset:
                        prompt = sample.get("prompt", "")
                        for i in range(0, math.ceil(1/num_distinct_img)):
                            conversations = []
                            response = ""
                            for entry in random.sample(sample["conversations"], num_distinct_questions):
                                for subentry in entry:
                                    response += subentry
                            response += "\n"
                            conversations.append(response)
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
                            count += 1
                            if count == max_data_size:
                                break
                else:
                    subset = hf_dataset.select(range(0, int(max_data_size * num_distinct_img)))
                    count = 0
                    for sample in subset:
                        prompt = sample.get("prompt", "")
                        for i in range(0, math.ceil(1/num_distinct_img)):
                            conversations = random.sample(sample["conversations"], num_distinct_questions)
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
                            count += 1
                            if count == max_data_size:
                                break
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