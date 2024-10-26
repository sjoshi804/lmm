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

    def forward(self, images, instructions, responses):
        batch_size = images.size(0)

        # Encode images and project to GPT-J embedding dimension
        image_features = self.vision_encoder(images)  # (batch_size, num_patches, feature_dim)
        multimodal_embeddings = self.multimodal_projector(image_features)  # (batch_size, num_patches, embedding_dim)

        inputs_embeds_list = []
        attention_masks = []
        labels_list = []

        for i in range(batch_size):
            # Process each sample individually
            image_embed = multimodal_embeddings[i]  # (num_patches, embedding_dim)
            image_embed = image_embed.unsqueeze(0)  # (1, num_patches, embedding_dim)

            # Build the textual input
            instr_input_ids = self.tokenizer(instructions[i], return_tensors='pt')['input_ids'] 
            resp_input_ids = self.tokenizer(responses[i], return_tensors='pt')['input_ids']

            # Tokenize the text sequence
            input_ids = torch.cat([instr_input_ids, resp_input_ids], dim=1)  # (1, seq_len)
            attention_mask = torch.ones_like(input_ids)  # (1, seq_len)

            # Get embeddings for the text input
            text_embeds = self.gptj.transformer.wte(input_ids)  # (1, seq_len, embedding_dim)

            # Combine image and text embeddings
            inputs_embeds = torch.cat([image_embed, text_embeds], dim=1)  # (1, num_patches + seq_len, embedding_dim)
            inputs_embeds_list.append(inputs_embeds)

            # Update attention mask to account for image embeddings
            image_attention_mask = torch.ones((1, image_embed.size(1)), dtype=torch.long)
            combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            attention_masks.append(combined_attention_mask)

            # Prepare labels (set labels for image embeddings to -100)
            image_labels = torch.full((1, image_embed.size(1)), -100, dtype=torch.long)
            instr_labels = torch.full(instr_input_ids.shape, -100, dtype=torch.long)
            resp_labels = resp_input_ids.clone()
            labels = torch.cat([image_labels, instr_labels, resp_labels], dim=1)
            labels_list.append(labels)

        # Pad sequences to the same length
        inputs_embeds_padded = nn.utils.rnn.pad_sequence(
            [embeds.squeeze(0) for embeds in inputs_embeds_list],
            batch_first=True,
        )
        attention_masks_padded = nn.utils.rnn.pad_sequence(
            [mask.squeeze(0) for mask in attention_masks],
            batch_first=True,
            padding_value=0,
        )
        labels_padded = nn.utils.rnn.pad_sequence(
            [lbl.squeeze(0) for lbl in labels_list],
            batch_first=True,
            padding_value=-100,
        )

        # Forward pass through GPT-J
        outputs = self.gptj(
            inputs_embeds=inputs_embeds_padded,
            attention_mask=attention_masks_padded,
            labels=labels_padded,
        )

        return outputs

class GPTJ_VLM_DataCollator:
    def __init__(self, vision_transform):
        self.vision_transform = vision_transform  # Preprocessing transformations for images

    def __call__(self, examples):
        images = []
        sys_prompts = []
        instr_resp_pairs_list = []

        for example in examples:
            # Apply vision transformations
            image = self.vision_transform(example['image'])
            images.append(image)

            sys_prompts.append(example['sys_prompt'])
            instr_resp_pairs_list.append(example['instr_resp_pairs'])

        images = torch.stack(images, dim=0)
        batch = {
            'images': images,
            'sys_prompts': sys_prompts,
            'instr_resp_pairs': instr_resp_pairs_list,
        }
        return batch
