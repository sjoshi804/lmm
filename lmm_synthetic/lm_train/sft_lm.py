from transformers import GPTJForCausalLM, GPT2TokenizerFast, Trainer, TrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass, field
from typing import Optional
import os
import re
import torch 

def replace_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '_', input_string)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="EleutherAI/gpt-j-6B", metadata={"help": "Path to the pretrained model or model identifier."})

@dataclass
class DataArguments:
    dataset_name: str = field(default="wikitext", metadata={"help": "The name of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The config name of the dataset to use."})
    load_from_disk: bool = field(default=False, metadata={"help": "Whether to load the dataset from disk."})
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length for the dataset."})
    split: Optional[str] = field(default="train", metadata={"help": "The dataset split to use."})
    cache_dir: str = field(default="/home/sjoshi/lmm/lm-train/tokenized_data/", metadata={"help": "Directory to cache the tokenized dataset."})

class SFTDataCollator():
    def __init__(self, tokenizer):      
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        input_ids = []
        labels = []
        attention_masks = []
        for example in examples:
            curr_input_ids = []
            curr_labels = []
            prompt_ids = self.tokenizer.encode(example["prompt"], add_special_tokens=False)
            curr_input_ids.append(prompt_ids)
            for instr, resp in example["conversations"]:
                instr_ids = self.tokenizer.encode(instr, add_special_tokens=False)
                resp_ids = self.tokenizer.encode(resp, add_special_tokens=False)
                curr_input_ids.extend([instr_ids, resp_ids])
                curr_labels.extend([torch.full(instr_ids.shape, -100), resp_ids])

                attention_masks.append([1] * len(input_ids[-1]))
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding="max_length", max_length=2048)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_masks
        )
    
def main():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False 
    
    data_args.cache_dir = os.path.join(data_args.cache_dir, replace_non_alphanumeric(f"{data_args.dataset_name}_config={data_args.dataset_config_name}_split={data_args.split}"))
    
    # Load dataset
    if data_args.load_from_disk:
        dataset = load_from_disk(data_args.dataset_name)[data_args.split]
    else:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.split)
    
    # Load pretrained model and tokenizer
    model = GPTJForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenized dataset saved to {data_args.cache_dir}")

    # Define a custom data collator (pass the tokenizer to it)
    custom_data_collator = SFTDataCollator(tokenizer=tokenizer)

    # Initialize Trainer for supervised fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()
