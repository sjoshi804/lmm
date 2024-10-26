import os
from dataclasses import dataclass, field
from transformers import GPTJForCausalLM, HfArgumentParser, TrainingArguments
from utils import prepare_multimodal_data, load_vision_encoder, load_multimodal_projector
from gptj_vlm import GPTJ_VLM
from custom_sft_trainer import CustomSFTTrainer

# Define data-specific arguments
@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the dataset."}
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

    # Load the vision encoder and multimodal projector
    vision_encoder = load_vision_encoder()
    multimodal_projector = load_multimodal_projector()
    
    # Load the GPT-J model
    gptj = GPTJForCausalLM.from_pretrained(model_args.gptj_model_path)

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
    model = GPTJ_VLM(gptj, vision_encoder, multimodal_projector)

    # Prepare the dataset using the function in utils.py
    dataset = prepare_multimodal_data(data_args.data_path)

    # Initialize the Custom SFT Trainer
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=None,
        tokenizer=gptj.tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
