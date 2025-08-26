import logging
import pathlib
from dataclasses import dataclass, field
from lmfusion_qwen3 import LMFusionQwen3ForCausalLM
from lmfusion_qwen3_config import LMFusionQwen3Config
from dataset import LMFusionQwen3Dataset, DataCollatorForLMFusionQwen3Dataset
import torch

from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to the pretrained model used for initializing weights, e.g., 'Qwen/Qwen2-7B-Instruct'"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )

@dataclass
class LMFusionQwen3TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    training_phase: str = field(
        default="pretrain",
        metadata={"help": "Specifies the training phase: 'pretrain' or 'finetune'."}
    )
    timestep_sampling_strategy: str = field(
        default="uniform",
        metadata={"help": "In the finetuning stage, noise is added to the target images. \
                  The noise is sampled from a specific distribution, typically a standard \
                  Gaussian distribution (N(0, I)), depending on the requirements of the d\
                  iffusion or generative training framework.Options: 'uniform', 'cosine', etc."}
    )
    remove_unused_columns: bool = False
   
def train():

    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LMFusionQwen3TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    model = LMFusionQwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    model.build_vae(vae_model_name_or_path=model.config.vae_model_name_or_path)
    model.config.model_name_or_path=model_args.model_name_or_path
    model.config.training_phase = training_args.training_phase
    model.config.timestep_sampling_strategy = training_args.timestep_sampling_strategy

    logging.info(f"Model {model_args.model_name_or_path} loaded.")

    # Freeze the appropriate weights
    logger.info("Setting up parameter freezing...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith("model.layers."):
            if "image" in name:
                param.requires_grad = True
        elif name.startswith("model.unet_") or name.startswith("model.norm.weight") or name.startswith("time_embedding"):
            param.requires_grad = True

    # # Print the number of trainable parameters for verification
    # # With DeepSpeed ZeRO-3, model parameters cannot be obtained via model.parameters()
    # trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # total_params = sum([p.numel() for p in model.parameters()])
    # logger.info(f"Trainable params: {trainable_params/1e9:.1f} B || Total params: {total_params/1e9:.1f} B || Trainable %: {100 * trainable_params / total_params:.2f}")

    train_dataset = LMFusionQwen3Dataset(
        data_path=data_args.data_path,
        training_phase=training_args.training_phase
    )

    data_collator = DataCollatorForLMFusionQwen3Dataset()

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No evaluation set is provided
        tokenizer=model.tokenizer,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("Starting training from scratch")
        trainer.train()


    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(output_dir=training_args.output_dir)
    trainer.save_state()

if __name__ == "__main__":
    train()
