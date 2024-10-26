import os
from dataclasses import dataclass, field
from transformers import GPTJForCausalLM, HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer
from utils import prepare_multimodal_data, load_vision_encoder, load_multimodal_projector
from gptj_vlm import GPTJ_VLM, GPTJ_VLM_DataCollator
from mm_datasets import LazySupervisedDataset 

# Define data-specific arguments
@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the dataset."}
    )
    split: str = field(
        metadata={"help": "Dataset split to use (e.g. 'train', 'validation', 'test')."}
    )

# Define model-specific arguments
@dataclass
class ModelArguments:
    gptj_model_path: str = field(
        metadata={"help": "Path to the pretrained GPT-J model."}
    )
    freeze_lm: bool = field(
        default=False, metadata={"help": "Whether to freeze the language model."}
    )
    freeze_multimodal_projector: bool = field(
        default=False, metadata={"help": "Whether to freeze the multimodal projector."}
    )
    freeze_vision_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision encoder."}
    )

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load the GPT-J model
    gptj = GPTJForCausalLM.from_pretrained(model_args.gptj_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.gptj_model_path)
    lang_embedding_size = gptj.config.hidden_size

    # Load the vision encoder and multimodal projector
    vision_encoder, image_transforms, vision_embedding_size = load_vision_encoder(model_args.vision_encoder)
    multimodal_projector = load_multimodal_projector(model_args.multimodal_projector, vision_embedding_size, lang_embedding_size)

    # Freeze components based on arguments
    if model_args.freeze_lm:
        for param in gptj.parameters():
            param.requires_grad = False
    if model_args.freeze_vision_encoder:
        for param in vision_encoder.parameters():
            param.requires_grad = False
    if model_args.freeze_multimodal_projector:
        for param in multimodal_projector.parameters():
            param.requires_grad = False

    # Initialize the VLM model
    model = GPTJ_VLM(gptj, vision_encoder, multimodal_projector, tokenizer)

    # Prepare the dataset using the function in utils.py
    dataset = LazySupervisedDataset(data_args.data_path, data_args.split, image_transforms)

    # Initialize the Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=GPTJ_VLM_DataCollator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
