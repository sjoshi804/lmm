from transformers import GPTJConfig, GPTJForCausalLM, GPT2TokenizerFast, Trainer, TrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer
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
    rotary_dim: int = field(default=64, metadata={"help": "Rotary dimension for rotary positional embeddings."})
    intermediate_size: int = field(default=3072, metadata={"help": "Intermediate size of the feed-forward layers."})

@dataclass
class DataArguments:
    dataset_name: str = field(default="wikitext", metadata={"help": "The name of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The config name of the dataset to use."})
    load_from_disk: bool = field(default=False, metadata={"help": "Whether to load the dataset from disk."})
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length for the dataset."})
    split: Optional[str] = field(default="train", metadata={"help": "The dataset split to use."})
    cache_dir: str = field(default="/home/sjoshi/lmm/lm-train/tokenized_data/", metadata={"help": "Directory to cache the tokenized dataset."})
    tokenizer_dir: str = field(default="/home/sjoshi/lmm/lm-train/tokenizer/", metadata={"help": "Directory to save/load the trained tokenizer."})

def train_tokenizer(dataset, tokenizer_dir, vocab_size):
    tokenizer = ByteLevelBPETokenizer()
    dataset_texts = (sample["text"] for sample in dataset)
    tokenizer.train_from_iterator(dataset_texts, vocab_size=vocab_size, min_frequency=2)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
    print(f"Tokenizer saved to {tokenizer_dir}")

def main():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()
    
    data_args.tokenizer_dir = os.path.join(data_args.tokenizer_dir, replace_non_alphanumeric(f"{data_args.dataset_name}_config={data_args.dataset_config_name}"))
    tokenizer_files = {
        "vocab_file": os.path.join(data_args.tokenizer_dir, "vocab.json"),
        "merges.txt": os.path.join(data_args.tokenizer_dir, "merges.txt")
    }

    if data_args.load_from_disk:
        dataset = load_from_disk(data_args.dataset_name)[data_args.split]
    else:        
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.split)
    train_tokenizer(dataset, data_args.tokenizer_dir, model_args.vocab_size)
    tokenizer = GPT2TokenizerFast(vocab_file=tokenizer_files["vocab_file"], merges_file=tokenizer_files["merges.txt"])

    tokenizer.pad_token = tokenizer.eos_token
    data_args.cache_dir = os.path.join(data_args.cache_dir, replace_non_alphanumeric(f"{data_args.dataset_name}_config={data_args.dataset_config_name}_split={data_args.split}"))
    
    def tokenize_function(examples):
        encoding = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_length,
        )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.save_to_disk(data_args.cache_dir)
    print(f"Tokenized dataset saved to {data_args.cache_dir}")

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()
