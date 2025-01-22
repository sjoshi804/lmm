from transformers import GPTJConfig, GPTJForCausalLM, GPT2TokenizerFast, Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer
from datasets import load_dataset, load_from_disk
from lmm_synthetic.lm_train.utils import ModelArguments, DataArguments, train_tokenizer
import os
import torch
from torch import nn
import difflib

class SFTDataCollator():
    def __init__(self, tokenizer, max_length=256, debug=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))  # Get the local rank from environment variables
        self.debug = debug and local_rank == 0
        
    def __call__(self, examples):
        input_ids_list = []
        label_ids_list = []

        for example_num, example in enumerate(examples):
            debug = example_num == 0 and self.debug
            full_input = ""
            
            # Extract prompt and conversations
            prompt = example["text"].split(".\n")[0] + ".\n"
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
            'input_ids': input_ids_padded,
            'labels': label_ids_padded,
        }
        return batch

def main():
    # Parse arguments
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False 
    
    # Load dataset
    if data_args.load_from_disk:
        dataset = load_from_disk(data_args.dataset_name)[data_args.split]
    else:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.split)
    
    # Load pretrained model or initialize a new model
    if model_args.model_name_or_path is None:
        config = GPTJConfig(
            vocab_size=model_args.vocab_size,
            n_positions=model_args.n_positions,
            n_embd=model_args.n_embd,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            rotary_dim=model_args.rotary_dim,
            intermediate_size=model_args.intermediate_size,
            activation_function='gelu_new',
            initializer_range=0.02,
        )
        model = GPTJForCausalLM(config)
    else:
        model = GPTJForCausalLM.from_pretrained(model_args.model_name_or_path)
        
    # Train or Load Tokenizer
    if model_args.model_name_or_path is None:
        if model_args.train_tokenizer:
            train_tokenizer(dataset, training_args.output_dir, model_args.vocab_size)
            tokenizer = GPT2TokenizerFast(
                vocab_file=os.path.join(training_args.output_dir, "vocab.json"), 
                merges_file=os.path.join(training_args.output_dir, "merges.txt")
            )
            tokenizer.save_pretrained(training_args.output_dir)
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define a custom data collator (pass the tokenizer to it)
    custom_data_collator = SFTDataCollator(
        tokenizer=tokenizer, 
        max_length=data_args.max_length,
        debug=data_args.debug_data
    )

    # Initialize Trainer for supervised fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
