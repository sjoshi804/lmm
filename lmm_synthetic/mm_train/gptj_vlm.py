import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, GPTJForCausalLM, GPTJConfig
from lmm_synthetic.mm_train.utils import load_vision_encoder, load_multimodal_projector
import safetensors.torch
import os 

class GPTJ_VLM_Config(PretrainedConfig):
    def __init__(self, vision_encoder_config=None, multimodal_projector_config=None, gptj_config=None, pad_token_id=None, pretrained_lm_path=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder_config = vision_encoder_config
        self.multimodal_projector_config = multimodal_projector_config
        self.gptj_config = GPTJConfig.from_dict(gptj_config) if isinstance(gptj_config, dict) else gptj_config
        self.pad_token_id = pad_token_id
        self.hidden_size = self.gptj_config.hidden_size if self.gptj_config is not None else None
        self.pretrained_lm_path = pretrained_lm_path # also the path to the pretrained tokenizer
        self.multimodal = True
        
class GPTJ_VLM(PreTrainedModel):
    config_class = GPTJ_VLM_Config

    def __init__(self, config: GPTJ_VLM_Config):
        super().__init__(config)
        self.gptj = GPTJForCausalLM(config.gptj_config)
        self.vision_encoder, self.image_transforms, self.vision_embed_dim = load_vision_encoder(config.vision_encoder_config)
        self.multimodal_projector = load_multimodal_projector(config.multimodal_projector_config, self.vision_embed_dim, self.gptj.config.hidden_size)
        self.config = config


    def forward(self, images, text_input_ids, label_ids):
        image_embeds = self.multimodal_projector(self.vision_encoder(images))
        text_embeds = self.gptj.transformer.wte(text_input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        label_ids = torch.cat([torch.full(image_embeds.size()[:2], -100, dtype=torch.long, device=inputs_embeds.device), label_ids], dim=1)
        attention_mask = torch.cat(
            [torch.ones(image_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device), text_input_ids.ne(self.config.pad_token_id)],
            dim=1
        )
        outputs = self.gptj(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
        )
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load the model and configuration.
        """
        config = GPTJ_VLM_Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        gptj_vlm = cls(config)
        gptj_vlm.load_state_dict(safetensors.torch.load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors")))
        return gptj_vlm

    def generate(self, images, text_input_ids, max_length=50, num_beams=1, **generate_kwargs):
        """
        Generate text based on image and text inputs.
        
        Args:
            images (torch.Tensor): Batch of images to process.
            text_input_ids (torch.Tensor): Batch of input text token IDs.
            max_length (int): Maximum length of the generated sequence.
            num_beams (int): Number of beams for beam search. Default is 1 (greedy search).
            generate_kwargs: Additional arguments for the `generate` method.
        
        Returns:
            torch.Tensor: Generated token IDs.
        """
        # Encode images to obtain embeddings
        image_embeds = self.multimodal_projector(self.vision_encoder(images))
        
        # Encode text input into embeddings
        text_embeds = self.gptj.transformer.wte(text_input_ids)
        
        # Concatenate image and text embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Create attention mask
        attention_mask = torch.cat(
            [torch.ones(image_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device), text_input_ids.ne(self.config.pad_token_id)],
            dim=1
        )
        
        # Generate outputs using the GPTJ model's generate method
        outputs = self.gptj.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **generate_kwargs
        )
        
        return outputs
    
class GPTJ_VLM_DataCollator:
    def __init__(self, tokenizer, image_transforms):
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        
    def __call__(self, examples):
        images = []
        text_input_ids_list = []
        label_ids_list = []

        for example in examples:
            images.append(self.image_transforms(example["image"]))  # Add batch dimension
            prompt = example["prompt"]
            conversations = example["conversations"]

            # Tokenize prompt and conversations
            text_input_ids = [] 
            label_ids = []

            # Tokenize prompt
            prompt_input_ids = self.tokenizer(prompt, return_tensors='pt', padding=True)['input_ids'].squeeze(0)
            text_input_ids.append(prompt_input_ids)
            label_ids.append(prompt_input_ids.clone())

            # Tokenize conversations
            for instr, resp in conversations:
                instr_input_ids = self.tokenizer(instr, return_tensors='pt', padding=True)['input_ids'].squeeze(0)
                resp_input_ids = self.tokenizer(resp, return_tensors='pt', padding=True)['input_ids'].squeeze(0)

                text_input_ids.extend([instr_input_ids, resp_input_ids])
                label_ids.extend([
                    torch.full(instr_input_ids.shape, -100, dtype=torch.long),  # Mask instruction in labels
                    resp_input_ids  # Keep response for label
                ])

            # Concatenate all input IDs and labels for this example
            text_input_ids = torch.cat(text_input_ids, dim=0)  # (seq_len)
            label_ids = torch.cat(label_ids, dim=0)  # (seq_len)

            # Append tokenized data
            text_input_ids_list.append(text_input_ids)
            label_ids_list.append(label_ids)

        # Pad sequences for batch processing
        images = torch.stack(images, dim=0)
        text_input_ids_padded = nn.utils.rnn.pad_sequence(
            text_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        label_ids_padded = nn.utils.rnn.pad_sequence(
            label_ids_list, batch_first=True, padding_value=-100
        )
        
        # Create batch dictionary
        batch = {
            'images': images,
            'text_input_ids': text_input_ids_padded,
            'label_ids': label_ids_padded,
        }
        
        return batch