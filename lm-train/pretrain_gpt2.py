from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset, Dataset, load_from_disk
from dataclasses import dataclass, field
from typing import Optional
import os
import re

def replace_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '_', input_string)

@dataclass
class ModelArguments:
    vocab_size: int = field(default=50257, metadata={"help": "Vocabulary size of the model."})
    n_positions: int = field(default=1024, metadata={"help": "Number of positions (max sequence length)."})
    n_embd: int = field(default=768, metadata={"help": "Embedding size of the model."})
    n_layer: int = field(default=12, metadata={"help": "Number of layers in the model."})
    n_head: int = field(default=12, metadata={"help": "Number of attention heads."})
    intermediate_size: int = field(default=3072, metadata={"help": "Intermediate size of the feed-forward layers."})

@dataclass
class DataArguments:
    dataset_name: str = field(default="wikitext", metadata={"help": "The name of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The config name of the dataset to use."})
    load_from_disk: bool = field(default=False, metadata={"help": "Whether to load the dataset from disk."})
    max_length: int = field(default=1024, metadata={"help": "Maximum sequence length for the dataset."})
    split: Optional[str] = field(default="train", metadata={"help": "The dataset split to use."})
    cache_dir: str = field(default="/home/sjoshi/lmm/lm-train/tokenized_data/", metadata={"help": "Directory to cache the tokenized dataset."})

def main():
    # Setup HfArgumentParser for direct argument parsing
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Load the pretrained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    data_args.cache_dir = os.path.join(data_args.cache_dir, replace_non_alphanumeric(f"{data_args.dataset_name}_config={data_args.dataset_config_name}_split={data_args.split}"))
    # Check if the tokenized dataset already exists
    if os.path.exists(data_args.cache_dir):
        print(f"Loading tokenized dataset from {data_args.cache_dir}")
        tokenized_dataset = load_from_disk(data_args.cache_dir)
    else:
        # Load the dataset
        dataset = None
        if data_args.load_from_disk:
            dataset = load_from_disk(data_args.dataset_name)[data_args.split]
        else:
            dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.split)

        # Tokenize the dataset with labels
        def tokenize_function(examples):
            encoding = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_length,
            )
            # Set the input_ids as the labels for causal language modeling
            encoding["labels"] = encoding["input_ids"].copy()
            return encoding

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Save the tokenized dataset locally
        tokenized_dataset.save_to_disk(data_args.cache_dir)
        print(f"Tokenized dataset saved to {data_args.cache_dir}")

    # Initialize the model with the specified configuration
    config = GPT2Config(
        vocab_size=model_args.vocab_size,
        n_positions=model_args.n_positions,
        n_ctx=model_args.n_positions,
        n_embd=model_args.n_embd,
        n_layer=model_args.n_layer,
        n_head=model_args.n_head,
        intermediate_size=model_args.intermediate_size,
        activation_function='gelu_new',
        initializer_range=0.02,
    )
    model = GPT2LMHeadModel(config).to(training_args.device)

    # Define a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
