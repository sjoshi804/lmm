from transformers import GPTJConfig, GPTJForCausalLM, GPT2TokenizerFast, Trainer, TrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk
from lmm_synthetic.lm_train.utils import ModelArguments, DataArguments, train_tokenizer, replace_non_alphanumeric
import os

def main():
    # Parse arguments
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Load dataset
    if data_args.load_from_disk:
        dataset = load_from_disk(data_args.dataset_name)[data_args.split]
    else:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.split)
    
    # Train or load tokenizer
    if model_args.train_tokenizer:
        train_tokenizer(dataset, training_args.output_dir, model_args.vocab_size)
        tokenizer = GPT2TokenizerFast(
            vocab_file=os.path.join(training_args.output_dir, "vocab.json"),
            merges_file=os.path.join(training_args.output_dir, "merges.txt")
        )
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
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

    # Configure model
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

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()
