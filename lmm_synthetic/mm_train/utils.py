import enum
import json
import torch

from loguru import logger
from torch import nn
from torchvision import transforms
from transformers import CLIPModel, CLIPImageProcessor
from typing import Optional 

# Enum for vision token ablations
class VisionTokenAblations(enum.Enum):
    ONE_HOT_ENCODER = "one_hot_encoder"

def load_vision_encoder(encoder_name: str, config_kwargs: Optional[dict] = None):
    """
    Load the vision encoder based on the specified encoder name.

    Args:
        encoder_name (str): The name of the encoder to load.
        config_kwargs (dict): Additional configuration arguments.

    Returns:
        tuple: A tuple containing the encoder, transform function, and output dimension.
    """
    logger.info(f"Loading vision encoder: {encoder_name}")

    if encoder_name == 'clip':
        logger.info("Using CLIP model as the vision encoder")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP expects images of size 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
        output_dim = clip_model.vision_model.config.hidden_size

        # Define a wrapper around the CLIP vision encoder to process pre-transformed images
        class VisionEncoderWrapper(nn.Module):
            def __init__(self, clip_model):
                super(VisionEncoderWrapper, self).__init__()
                self.clip_vision_model = clip_model.vision_model

            def forward(self, images):
                # Expect images to be pre-transformed and batched [batch_size, 3, H, W]
                vision_outputs = self.clip_vision_model(pixel_values=images)
                return vision_outputs.last_hidden_state  # [batch_size, num_patches + 1, hidden_dim]

        encoder = VisionEncoderWrapper(clip_model)
    elif encoder_name in [e.value for e in VisionTokenAblations]:
        logger.info(f"Using {encoder_name} as the vision encoder")
        vocab = None
        with open(config_kwargs["dataset_config_path"], "r") as f:
            dataset_config = json.load(f)
            vocab = dataset_config["vocab"]
        encoder = nn.Identity()
        vocab_embedding_mapping = {}
        for object_num, object in enumerate(vocab):
            if encoder_name == VisionTokenAblations.ONE_HOT_ENCODER.value:
                vocab_embedding_mapping[object] = torch.eye(len(vocab))[object_num]
                output_dim = len(vocab)
            else:
                raise NotImplementedError(f"Unsupported encoder: {encoder_name}")
        
        def transform(grid):
            tokens = []
            for row in grid:
                for cell in row:
                    tokens.append(vocab_embedding_mapping[cell])
            return torch.stack(tokens, dim=0)
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    return encoder, transform, output_dim

def load_multimodal_projector(projector_type: str, input_dim: int, output_dim: int):
    """
    Load the multimodal projector based on the specified projector type.

    Args:
        projector_type (str): The type of the projector to load.
        input_dim (int): The input dimension of the projector.
        output_dim (int): The output dimension of the projector.

    Returns:
        nn.Module: The projector module.
    """
    logger.info(f"Loading multimodal projector: {projector_type}")

    if projector_type == 'linear':
        return nn.Linear(input_dim, output_dim)
    else:
        raise ValueError(f"Unsupported projector type: {projector_type}")
