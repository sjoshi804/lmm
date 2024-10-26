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
        image_transforms: transforms.Compose = None
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        
        hf_dataset = load_from_disk(data_path)[split]
        list_data_dict = []
        for sample in hf_dataset:
            prompt = sample.get("prompt", "")
            conversations = sample.get("conversations", [])
            list_data_dict.append(
            {
                "image": sample.get("image", None),
                "prompt": prompt,
                "conversations": conversations
            })
        

        # Determine whether each sample is text-only
        self.is_text_only = [
            "image" not in source for source in self.list_data_dict
        ]

        # Image transformation if needed
        if image_transforms is None:
            self.image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transforms = image_transforms

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:
        # Retrieve the sample data
        sample = self.list_data_dict[i]

        # Load image if available
        if not self.is_text_only[i]:
            if "image" in sample:
                image_path = sample["image"]
                image = Image.open(image_path).convert("RGB")
                image = self.image_transforms(image)  # Apply transformations
        else:
            image = None  # No image for text-only samples

        # Return the sample in the expected format
        return dict(
            images=image,
            conversations=sample["conversations"],
            prompt=sample["prompt"]
        )
