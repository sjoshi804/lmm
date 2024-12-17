import os
from dataclasses import dataclass, field
from transformers import (
    GPTJForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AdamW,
    get_scheduler
)
from gptj_vlm import GPTJ_VLM, GPTJ_VLM_Config, GPTJ_VLM_DataCollator
from mm_datasets import LazySupervisedDataset

# Define data-specific arguments
@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the dataset."})
    split: str = field(metadata={"help": "Dataset split to use (e.g., 'train', 'validation', 'test')."})
    max_data_size: int = field(default=-1, metadata={"help": "Maximum number of samples to load."})
    debug_data: bool = field(default=False, metadata={"help": "Whether to run in data debug mode."})
    
# Define model-specific arguments
@dataclass
class ModelArguments:
    gptj_model_path: str = field(metadata={"help": "Path to the pretrained GPT-J model."})
    vision_encoder_config: str = field(default="clip", metadata={"help": "Type of vision encoder to use (e.g., 'clip')."})
    multimodal_projector_config: str = field(default="linear", metadata={"help": "Type of multimodal projector to use (e.g., 'linear')."})

# Define custom training arguments for freeze and LR scaling
@dataclass
class CustomTrainingArguments(TrainingArguments):
    freeze_lm: bool = field(default=True, metadata={"help": "Whether to freeze the language model."})
    freeze_multimodal_projector: bool = field(default=False, metadata={"help": "Whether to freeze the multimodal projector."})
    freeze_vision_encoder: bool = field(default=True, metadata={"help": "Whether to freeze the vision encoder."})
    lr_scale_lm: float = field(default=1.0, metadata={"help": "Learning rate scaling factor for the language model."})
    lr_scale_vision_encoder: float = field(default=1.0, metadata={"help": "Learning rate scaling factor for the vision encoder."})

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Construct parameter groups with scaled learning rates based on training arguments
        optimizer_grouped_parameters = [
            {"params": [], "lr": self.args.learning_rate * self.args.lr_scale_lm},
            {"params": [], "lr": self.args.learning_rate * self.args.lr_scale_vision_encoder},
            {"params": [], "lr": self.args.learning_rate}
        ]

        for name, param in self.model.named_parameters():
            # If it's frozen, skip adding it to the optimizer
            if not param.requires_grad:
                continue

            # Assign params to the appropriate LR group
            if "gptj" in name and not self.args.freeze_lm:
                optimizer_grouped_parameters[0]["params"].append(param)
            elif "vision_encoder" in name and not self.args.freeze_vision_encoder:
                optimizer_grouped_parameters[1]["params"].append(param)
            else:
                optimizer_grouped_parameters[2]["params"].append(param)

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def log(self, logs: dict):
        """
        Override the default logging behavior to include learning rates for all parameter groups.
        """
        # Add learning rates for each parameter group
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            last_lrs = self.lr_scheduler.get_last_lr()
            if len(last_lrs) == 1:
                logs["lr_mm_proj"] = last_lrs[0]
            elif len(last_lrs) == 2:
                if self.args.freeze_vision_encoder:
                    logs["lr_lm"] = last_lrs[0]
                    logs["lr_mm_proj"] = last_lrs[1]
                elif self.args.freeze_lm:
                    logs["lr_vision_encoder"] = last_lrs[0]
                    logs["lr_mm_proj"] = last_lrs[1]
            elif len(last_lrs) == 3:
                logs["lr_lm"] = last_lrs[0]
                logs["lr_vision_encoder"] = last_lrs[1]
                logs["lr_mm_proj"] = last_lrs[2]
            del logs["learning_rate"]
        
        # Call the parent method to continue normal logging behavior
        super().log(logs)

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    # Load model and tokenizer
    gptj = GPTJForCausalLM.from_pretrained(model_args.gptj_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.gptj_model_path)
    model = GPTJ_VLM(
        config=GPTJ_VLM_Config(
            vision_encoder_config=model_args.vision_encoder_config,
            multimodal_projector_config=model_args.multimodal_projector_config,
            gptj_config=gptj.config,
            pad_token_id=tokenizer.pad_token_id,
            pretrained_lm_path=model_args.gptj_model_path
        ),
    )
    model.gptj = gptj

    # Freeze components based on custom training arguments
    if training_args.freeze_lm:
        for param in model.gptj.parameters():
            param.requires_grad = False
    if training_args.freeze_vision_encoder:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    if training_args.freeze_multimodal_projector:
        for param in model.multimodal_projector.parameters():
            param.requires_grad = False

    # Prepare dataset
    dataset = LazySupervisedDataset(data_args.data_path, data_args.split, data_args.max_data_size, debug=data_args.debug_data)
    data_collator = GPTJ_VLM_DataCollator(tokenizer, model.image_transforms, debug=data_args.debug_data)

    # Initialize custom trainer (no need to pass model_args here since we only need training_args)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
