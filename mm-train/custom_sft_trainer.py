from transformers import Trainer
import torch
import torch.nn.functional as F

class CustomSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get the image, system prompt, and instruction-response pairs
        images, sys_prompt, instr_resp_pairs = inputs['images'], inputs['sys_prompt'], inputs['instr_resp_pairs']

        # Forward pass through the model
        logits, labels = model(images, sys_prompt, instr_resp_pairs)

        # Shift the logits and labels for correct alignment as per your SFT setup
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Masking for <STOP> tokens and responses
        stop_token_id = self.tokenizer.convert_tokens_to_ids("<STOP>")
        mask = (shifted_labels == stop_token_id) | (shifted_labels > 0)

        # Compute the loss over masked tokens only
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        masked_loss = loss * mask.view(-1).float()
        masked_loss = masked_loss.sum() / mask.sum().float()

        return (masked_loss, logits) if return_outputs else masked_loss
