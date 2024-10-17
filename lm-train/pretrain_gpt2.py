from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional
import hydra
from omegaconf import DictConfig

@dataclass
class ModelArguments:
    vocab_size: Optional[int] = 50257
    n_positions: Optional[int] = 1024
    n_embd: Optional[int] = 768
    n_layer: Optional[int] = 12
    n_head: Optional[int] = 12
    intermediate_size: Optional[int] = 3072

@dataclass
class DataArguments:
    dataset_name: Optional[str] = "wikitext"
    dataset_config_name: Optional[str] = "wikitext-2-raw-v1"
    max_length: Optional[int] = 1024

@dataclass
class TrainingHyperparameters:
    output_dir: Optional[str] = "./gpt2-model"
    num_train_epochs: Optional[int] = 3
    per_device_train_batch_size: Optional[int] = 1
    gradient_accumulation_steps: Optional[int] = 1
    logging_dir: Optional[str] = "./logs"
    logging_steps: Optional[int] = 500
    save_total_limit: Optional[int] = 3
    save_steps: Optional[int] = 1000
    deepspeed_config: Optional[str] = "ds_config.json"

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Parse model, data, and training arguments
    model_args = ModelArguments(**cfg.model)
    data_args = DataArguments(**cfg.data)
    training_args = TrainingHyperparameters(**cfg.training)

    # Load the pretrained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split="train")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=data_args.max_length)

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
    model = GPT2LMHeadModel(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        logging_dir=training_args.logging_dir,
        logging_steps=training_args.logging_steps,
        save_total_limit=training_args.save_total_limit,
        save_steps=training_args.save_steps,
        deepspeed=training_args.deepspeed_config,
        fp16=True,
    )

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