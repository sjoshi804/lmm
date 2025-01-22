import difflib
import os

import safetensors.torch
import torch
from transformers import GPTJConfig, GPTJForCausalLM, PreTrainedModel, PretrainedConfig

from lmm_synthetic.mm_train.utils import load_multimodal_projector, load_vision_encoder


class GPTJ_VLM_Config(PretrainedConfig):
    """
    Configuration class for GPTJ_VLM model.
    """
    def __init__(self, vision_encoder_config=None, multimodal_projector_config=None, gptj_config=None, pad_token_id=None, pretrained_lm_path=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder_config = vision_encoder_config
        self.multimodal_projector_config = multimodal_projector_config
        self.gptj_config = GPTJConfig.from_dict(gptj_config) if isinstance(gptj_config, dict) else gptj_config
        self.pad_token_id = pad_token_id
        self.hidden_size = self.gptj_config.hidden_size if self.gptj_config is not None else None
        self.pretrained_lm_path = pretrained_lm_path  # also the path to the pretrained tokenizer
        self.multimodal = True
        self.kwargs = kwargs


class GPTJ_VLM(PreTrainedModel):
    """
    GPTJ_VLM model class.
    """
    config_class = GPTJ_VLM_Config

    def __init__(self, config: GPTJ_VLM_Config):
        super().__init__(config)
        self.gptj = GPTJForCausalLM(config.gptj_config)
        self.vision_encoder, self.image_transforms, self.vision_embed_dim = load_vision_encoder(config.vision_encoder_config, config.kwargs)
        self.multimodal_projector = load_multimodal_projector(config.multimodal_projector_config, self.vision_embed_dim, self.gptj.config.hidden_size)
        self.config = config

    def forward(self, images, text_input_ids, label_ids):
        """
        Forward pass for the model.
        
        Args:
            images (torch.Tensor): Batch of images to process.
            text_input_ids (torch.Tensor): Batch of input text token IDs.
            label_ids (torch.Tensor): Batch of label token IDs.
        
        Returns:
            torch.Tensor: Model outputs.
        """
        # Encode images to obtain embeddings
        image_embeds = self.multimodal_projector(self.vision_encoder(images))
        
        # Encode text input into embeddings
        text_embeds = self.gptj.transformer.wte(text_input_ids)
        
        # Concatenate image and text embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Create label IDs with -100 for image embeddings
        label_ids = torch.cat([torch.full(image_embeds.size()[:2], -100, dtype=torch.long, device=inputs_embeds.device), label_ids], dim=1)
        
        # Create attention mask
        attention_mask = torch.cat(
            [torch.ones(image_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device), text_input_ids.ne(self.config.pad_token_id)],
            dim=1
        )
        
        # Forward pass through GPTJ model
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
        
        Args:
            pretrained_model_name_or_path (str): Path to the pretrained model.
            model_args: Additional model arguments.
            kwargs: Additional keyword arguments.
        
        Returns:
            GPTJ_VLM: Loaded model.
        """
        config = GPTJ_VLM_Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        gptj_vlm = cls(config)
        gptj_vlm.load_state_dict(safetensors.torch.load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors")))
        return gptj_vlm

    def generate(self, images, text_input_ids, max_new_tokens=50, num_beams=1, **generate_kwargs):
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
        if images is not None:
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
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **generate_kwargs
            )
        else:
            # Generate outputs using the GPTJ model's generate method for text only
            outputs = self.gptj.generate(
                input_ids=text_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **generate_kwargs
            )
        
        return outputs


class GPTJ_VLM_DataCollator:
    """
    Data collator for GPTJ_VLM model.
    """
    def __init__(self, tokenizer, image_transforms, max_length=512, vision_token_ablation=False, debug=False):
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.max_length = max_length
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))  # Get the local rank from environment variables
        self.debug = debug and local_rank == 0
        self.vision_token_ablation = vision_token_ablation
        
    def __call__(self, examples):
        """
        Collate function to process a batch of examples.
        
        Args:
            examples (list): List of examples to process.
        
        Returns:
            dict: Batch dictionary containing images, text_input_ids, and label_ids.
        """
        images = []
        input_ids_list = []
        label_ids_list = []

        for example_num, example in enumerate(examples):
            debug = example_num == 0 and self.debug
            full_input = ""
            
            # Extract image, prompt, and conversations
            if self.vision_token_ablation:
                images.append(self.image_transforms(example["grid"]))
            else:
                images.append(self.image_transforms(example["image"])) 
            prompt = example["prompt"] + "\n"
            conversations = example["conversations"]
            
            if debug:
                full_input += prompt 
            
            # Tokenize prompt
            prompt_input_ids = self.tokenizer(prompt, return_tensors='pt', padding=True)['input_ids'].squeeze(0)
            input_ids = [prompt_input_ids]
            label_ids = [prompt_input_ids.clone()]

            # Tokenize conversations
            for conv_num, (instr, resp) in enumerate(conversations):
                instr = instr + " "
                if conv_num < len(conversations) - 1:
                    resp = resp + "\n" 
                
                if debug:
                    full_input += instr + resp 
                    
                instr_input_ids = self.tokenizer(instr, return_tensors='pt', padding=True)['input_ids'].squeeze(0)
                resp_input_ids = self.tokenizer(resp, return_tensors='pt', padding=True)['input_ids'].squeeze(0)

                input_ids.extend([instr_input_ids, resp_input_ids])
                label_ids.extend([
                    torch.full(instr_input_ids.shape, -100, dtype=torch.long),  # Mask instruction in labels
                    resp_input_ids  # Keep response for label
                ])
                
            if debug:
                print("Reconstructed Text", f"<start>{full_input}<end>")
                print("Original Text", f"<start>{example['text']}<end>") 
                diff = difflib.unified_diff(
                    example['text'].splitlines(keepends=True),
                    full_input.splitlines(keepends=True),
                    fromfile='original',
                    tofile='reconstructed'
                )
                print(''.join(diff))
                exit(0)

            # Concatenate all input IDs and labels for this example
            input_ids = torch.cat(input_ids, dim=0)  # (seq_len)
            label_ids = torch.cat(label_ids, dim=0)  # (seq_len)

            # Append tokenized data
            input_ids_list.append(input_ids)
            label_ids_list.append(label_ids)

        # Pad sequences for batch processing to max_length
        input_ids_padded = torch.full((len(input_ids_list), self.max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        label_ids_padded = torch.full((len(label_ids_list), self.max_length), -100, dtype=torch.long)

        for example_num, (input_ids, label_ids) in enumerate(zip(input_ids_list, label_ids_list)):
            seq_len = min(len(input_ids), self.max_length)
            input_ids_padded[example_num, :seq_len] = input_ids[:seq_len]
            label_ids_padded[example_num, :seq_len] = label_ids[:seq_len]
        
        # Create batch dictionary
        batch = {
            'images': torch.stack(images, dim=0),
            'text_input_ids': input_ids_padded,
            'label_ids': label_ids_padded,
        }
        
        return batch