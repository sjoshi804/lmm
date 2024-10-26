import torch
from torch import nn
from transformers import CLIPModel, CLIPImageProcessor
from torchvision import transforms
from PIL import Image

def load_vision_encoder(encoder_name: str):
    if encoder_name == 'clip':
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP expects images of size 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, 
                                 std=processor.image_std)
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
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    return encoder, transform, output_dim

def load_multimodal_projector(projector_type: str, input_dim: int, output_dim: int):
    if projector_type == 'linear':
        return nn.Linear(input_dim, output_dim)
    else:
        raise ValueError(f"Unsupported projector type: {projector_type}")
