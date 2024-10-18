from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional

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
    max_length: int = field(default=1024, metadata={"help": "Maximum sequence length for the dataset."})

def main():
    # Setup HfArgumentParser for direct argument parsing
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Load the pretrained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split="train")

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
