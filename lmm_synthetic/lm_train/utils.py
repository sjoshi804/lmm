from dataclasses import dataclass, field
from typing import Optional
import re
import os
from tokenizers import ByteLevelBPETokenizer

def replace_non_alphanumeric(input_string: str) -> str:
    """
    Replace all non-alphanumeric characters in the input string with underscores.

    Args:
        input_string (str): The input string to process.

    Returns:
        str: The processed string with non-alphanumeric characters replaced by underscores.
    """
    return re.sub(r'[^a-zA-Z0-9]', '_', input_string)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to the model configuration.
    """
    model_name_or_path: str = field(default=None, metadata={"help": "Path to the pretrained model or model identifier."})
    vocab_size: int = field(default=50257, metadata={"help": "Vocabulary size of the model."})
    n_positions: int = field(default=1024, metadata={"help": "Number of positions (max sequence length)."})
    n_embd: int = field(default=768, metadata={"help": "Embedding size of the model."})
    n_layer: int = field(default=12, metadata={"help": "Number of layers in the model."})
    n_head: int = field(default=12, metadata={"help": "Number of attention heads."})
    rotary_dim: int = field(default=64, metadata={"help": "Rotary dimension for rotary positional embeddings."})
    intermediate_size: int = field(default=3072, metadata={"help": "Intermediate size of the feed-forward layers."})
    train_tokenizer: bool = field(default=False, metadata={"help": "Whether to train the tokenizer."})

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data configuration.
    """
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The config name of the dataset to use."})
    load_from_disk: bool = field(default=False, metadata={"help": "Whether to load the dataset from disk."})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length for the dataset."})
    split: Optional[str] = field(default="train", metadata={"help": "The dataset split to use."})

def train_tokenizer(dataset, tokenizer_dir: str, vocab_size: int):
    """
    Train a ByteLevelBPETokenizer on the provided dataset and save it to the specified directory.

    Args:
        dataset: The dataset to train the tokenizer on.
        tokenizer_dir (str): The directory to save the trained tokenizer.
        vocab_size (int): The vocabulary size for the tokenizer.
    """
    tokenizer = ByteLevelBPETokenizer()
    dataset_texts = (sample["text"] for sample in dataset)
    tokenizer.train_from_iterator(dataset_texts, vocab_size=vocab_size, min_frequency=2)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
    print(f"Tokenizer saved to {tokenizer_dir}")