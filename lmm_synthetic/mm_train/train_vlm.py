import os
from dataclasses import dataclass, field
from transformers import GPTJForCausalLM, HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer
from utils import load_vision_encoder, load_multimodal_projector
from gptj_vlm import GPTJ_VLM, GPTJ_VLM_Config, GPTJ_VLM_DataCollator
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
    max_data_size: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to load."})

# Define model-specific arguments
@dataclass
class ModelArguments:
    gptj_model_path: str = field(
        metadata={"help": "Path to the pretrained GPT-J model."}
    )
    freeze_lm: bool = field(
        default=True, metadata={"help": "Whether to freeze the language model."}
    )
    freeze_multimodal_projector: bool = field(
        default=False, metadata={"help": "Whether to freeze the multimodal projector."}
    )
    freeze_vision_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the vision encoder."}
    )
    vision_encoder_config: str = field(
        default="clip", metadata={"help": "Type of vision encoder to use (e.g. 'clip')."}
    )
    multimodal_projector_config: str = field(
        default="linear", metadata={"help": "Type of multimodal projector to use (e.g. 'linear')."}
    )

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    
    # Load the GPT-J model
    gptj = GPTJForCausalLM.from_pretrained(model_args.gptj_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.gptj_model_path)
    model = GPTJ_VLM(
        config=GPTJ_VLM_Config(
            model_args.vision_encoder_config, 
            model_args.multimodal_projector_config, 
            gptj.config, 
            tokenizer.pad_token_id,
            model_args.gptj_model_path
        ),
    )
    model.gptj = gptj
    
    # Freeze components based on arguments
    if model_args.freeze_lm:
        for param in model.gptj.parameters():
            param.requires_grad = False
    if model_args.freeze_vision_encoder:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    if model_args.freeze_multimodal_projector:
        for param in model.multimodal_projector.parameters():
            param.requires_grad = False
            

    # Prepare the dataset using the function in utils.py
    dataset = LazySupervisedDataset(data_args.data_path, data_args.split, data_args.max_data_size)

    # Initialize the Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=GPTJ_VLM_DataCollator(tokenizer, model.image_transforms),
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
