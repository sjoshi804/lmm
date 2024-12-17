from torch.utils.data import Dataset
from typing import Dict, List
from PIL import Image
from datasets import load_from_disk
import torchvision.transforms as transforms

class LazySupervisedDataset(Dataset):
    """Dataset for multimodal supervised fine-tuning"""

    def __init__(
        self, 
        data_path: str, 
        split: str,
        max_data_size: int = -1,
        debug: bool = False
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.debug = debug
        hf_dataset = load_from_disk(data_path)[split]
        self.list_data_dict = []
        for sample in hf_dataset:
            prompt = sample.get("prompt", "")
            conversations = sample.get("conversations", [])
            self.list_data_dict.append(
            {
                "image": sample.get("image", None),
                "prompt": prompt,
                "conversations": conversations
            })
            if self.debug:
                self.list_data_dict[-1]["text"] = sample.get("text", "")
        if max_data_size > 0:
            self.list_data_dict = self.list_data_dict[:max_data_size]
        print("Dataset size:", len(self.list_data_dict))

        # Determine whether each sample is text-only
        self.is_text_only = [
            "image" not in source for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:
        sample = self.list_data_dict[i]
        item_dict = dict(
            image=Image.open(sample["image"]).convert("RGB"),
            prompt=sample["prompt"],
            conversations=sample["conversations"]
        )
        if self.debug:
            item_dict["text"] = sample["text"]
        return item_dict