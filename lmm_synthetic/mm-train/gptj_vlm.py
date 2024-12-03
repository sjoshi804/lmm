import torch
import torch.nn as nn
from transformers import GPTJForCausalLM

class GPTJ_VLM(nn.Module):
    def __init__(self, gptj: GPTJForCausalLM, vision_encoder, multimodal_projector, tokenizer):
        super(GPTJ_VLM, self).__init__()
        self.gptj = gptj  # Pre-trained GPT-J model
        self.vision_encoder = vision_encoder  # Vision encoder model
        self.multimodal_projector = multimodal_projector  # Linear layer to project image features
        self.tokenizer = tokenizer  # GPT-J tokenizer
        self.config = self.gptj.config

    def forward(self, images, text_input_ids, label_ids):
        # Combine image embeddings and text input embeddings
        image_embeds = self.multimodal_projector(self.vision_encoder(images))  # Image embeddings
        text_embeds = self.gptj.transformer.wte(text_input_ids)  # Text embeddings from token IDs
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (batch_size, num_patches + seq_len, embedding_dim)
        label_ids = torch.cat([torch.full(image_embeds.size()[:2], -100, dtype=torch.long, device=inputs_embeds.device), label_ids], dim=1)
        # Create an attention mask combining image and text attention
        attention_mask = torch.cat(
            [torch.ones(image_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device), text_input_ids.ne(self.tokenizer.pad_token_id)],
            dim=1
        )

        # Forward pass through GPT-J
        outputs = self.gptj(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
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
