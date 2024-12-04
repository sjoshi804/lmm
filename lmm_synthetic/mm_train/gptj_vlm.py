from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn

class GPTJ_VLM_Config(PretrainedConfig):
    def __init__(self, vision_encoder_config=None, multimodal_projector_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder_config = vision_encoder_config
        self.multimodal_projector_config = multimodal_projector_config

class GPTJ_VLM(PreTrainedModel):
    config_class = GPTJ_VLM_Config

    def __init__(self, config: GPTJ_VLM_Config, gptj, vision_encoder, multimodal_projector, tokenizer):
        super().__init__(config)
        self.gptj = gptj
        self.vision_encoder = vision_encoder
        self.multimodal_projector = multimodal_projector
        self.tokenizer = tokenizer  # Save tokenizer separately
        self.config = config

    def forward(self, images, text_input_ids, label_ids):
        image_embeds = self.multimodal_projector(self.vision_encoder(images))
        text_embeds = self.gptj.transformer.wte(text_input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        label_ids = torch.cat([torch.full(image_embeds.size()[:2], -100, dtype=torch.long, device=inputs_embeds.device), label_ids], dim=1)
        attention_mask = torch.cat(
            [torch.ones(image_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device), text_input_ids.ne(self.tokenizer.pad_token_id)],
            dim=1
        )
        outputs = self.gptj(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
        )
        return outputs

    def save_pretrained(self, save_directory):
        """
        Save the model, tokenizer, and configuration.
        """
        super().save_pretrained(save_directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
        # Save the vision encoder and multimodal projector configuration
        torch.save(self.vision_encoder.state_dict(), f"{save_directory}/vision_encoder.pt")
        torch.save(self.multimodal_projector.state_dict(), f"{save_directory}/multimodal_projector.pt")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load the model, tokenizer, and configuration.
        """
        config = GPTJ_VLM_Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        gptj = GPTJForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        vision_encoder = YourVisionEncoder()  # Initialize your vision encoder
        vision_encoder.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/vision_encoder.pt"))
        multimodal_projector = nn.Linear(vision_encoder.output_dim, gptj.config.n_embd)
        multimodal_projector.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/multimodal_projector.pt"))
        tokenizer = YourTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(config, gptj, vision_encoder, multimodal_projector, tokenizer)
