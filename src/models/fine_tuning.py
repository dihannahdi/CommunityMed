"""
QLoRA Fine-tuning for MedGemma
Implements parameter-efficient fine-tuning using LoRA adapters
"""

import os
import torch
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import yaml

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from loguru import logger


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 50
    save_strategy: str = "epoch"
    bf16: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class FineTuner:
    """
    Fine-tuning pipeline for MedGemma using QLoRA
    Designed for medical imaging tasks like TB detection from chest X-rays
    """
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        output_dir: str = "./outputs/medgemma-finetuned",
    ):
        """
        Initialize the fine-tuner
        
        Args:
            model: Pre-loaded MedGemma model (quantized)
            processor: Model processor/tokenizer
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.trainer = None
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Fine-tuner initialized. Output: {output_dir}")
        
    def get_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: str = "all-linear",
    ) -> LoraConfig:
        """
        Get LoRA configuration for parameter-efficient fine-tuning
        
        Args:
            r: LoRA rank (dimension of low-rank matrices)
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to
            
        Returns:
            LoraConfig object
        """
        return LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=r,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["lm_head", "embed_tokens"],
        )
    
    def get_training_config(
        self,
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ) -> SFTConfig:
        """
        Get SFTConfig for training
        
        Args:
            config: TrainingConfig dataclass
            **kwargs: Override specific settings
            
        Returns:
            SFTConfig object
        """
        config = config or TrainingConfig(output_dir=self.output_dir)
        
        return SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=kwargs.get("num_train_epochs", config.num_train_epochs),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", config.per_device_train_batch_size),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", config.per_device_eval_batch_size),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", config.gradient_accumulation_steps),
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=kwargs.get("logging_steps", config.logging_steps),
            save_strategy=config.save_strategy,
            eval_strategy="steps",
            eval_steps=kwargs.get("eval_steps", config.eval_steps),
            learning_rate=kwargs.get("learning_rate", config.learning_rate),
            bf16=config.bf16,
            max_grad_norm=config.max_grad_norm,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type="linear",
            push_to_hub=config.push_to_hub,
            report_to="tensorboard",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            label_names=["labels"],
        )
    
    def create_collate_fn(self) -> Callable:
        """
        Create custom data collator for image-text data
        
        Returns:
            Collate function for DataLoader
        """
        processor = self.processor
        
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Custom collator for multimodal data"""
            texts = []
            images = []
            
            for example in examples:
                # Handle image
                if "image" in example and example["image"] is not None:
                    img = example["image"]
                    if hasattr(img, "convert"):
                        images.append([img.convert("RGB")])
                    else:
                        images.append([img])
                else:
                    images.append(None)
                
                # Handle text (apply chat template)
                if "messages" in example:
                    text = processor.apply_chat_template(
                        example["messages"],
                        add_generation_prompt=False,
                        tokenize=False,
                    ).strip()
                elif "text" in example:
                    text = example["text"]
                else:
                    text = ""
                    
                texts.append(text)
            
            # Filter out None images
            valid_images = [img for img in images if img is not None]
            
            # Tokenize texts and process images
            if valid_images:
                batch = processor(
                    text=texts,
                    images=valid_images,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                batch = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                )
            
            # Create labels with masked tokens
            labels = batch["input_ids"].clone()
            
            # Mask special tokens
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            # Mask image tokens if present
            if hasattr(processor.tokenizer, "special_tokens_map"):
                boi_token = processor.tokenizer.special_tokens_map.get("boi_token")
                if boi_token:
                    boi_id = processor.tokenizer.convert_tokens_to_ids(boi_token)
                    labels[labels == boi_id] = -100
            
            # Mask image embedding positions (common token ID)
            labels[labels == 262144] = -100
            
            batch["labels"] = labels
            return batch
            
        return collate_fn
    
    def prepare_dataset(
        self,
        train_data: Dataset,
        eval_data: Optional[Dataset] = None,
        format_fn: Optional[Callable] = None,
    ) -> tuple:
        """
        Prepare datasets for training
        
        Args:
            train_data: Training dataset
            eval_data: Evaluation dataset (optional)
            format_fn: Function to format each example
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if format_fn:
            train_data = train_data.map(format_fn)
            if eval_data:
                eval_data = eval_data.map(format_fn)
                
        return train_data, eval_data
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        lora_config: Optional[LoraConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        """
        Run fine-tuning
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            lora_config: LoRA configuration (uses defaults if None)
            training_config: Training configuration (uses defaults if None)
        """
        logger.info("Starting fine-tuning...")
        
        # Get configs
        lora_config = lora_config or self.get_lora_config()
        sft_config = self.get_training_config(training_config)
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            processing_class=self.processor,
            data_collator=self.create_collate_fn(),
        )
        
        logger.info("Trainer initialized. Starting training...")
        
        # Train
        self.trainer.train()
        
        logger.success("Training complete!")
        
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the fine-tuned model
        
        Args:
            path: Save path (uses output_dir if None)
            
        Returns:
            Path where model was saved
        """
        save_path = path or self.output_dir
        
        if self.trainer:
            self.trainer.save_model(save_path)
        else:
            self.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def push_to_hub(self, repo_id: str, private: bool = False) -> str:
        """
        Push fine-tuned model to HuggingFace Hub
        
        Args:
            repo_id: Repository ID on HuggingFace
            private: Whether to make the repo private
            
        Returns:
            URL of the pushed model
        """
        if self.trainer:
            self.trainer.push_to_hub()
        else:
            self.model.push_to_hub(repo_id, private=private)
            self.processor.push_to_hub(repo_id, private=private)
            
        url = f"https://huggingface.co/{repo_id}"
        logger.success(f"Model pushed to {url}")
        return url


def create_tb_format_fn(prompt_template: str = None):
    """
    Create formatting function for TB detection dataset
    
    Args:
        prompt_template: Custom prompt template
        
    Returns:
        Formatting function
    """
    default_prompt = (
        "You are an expert radiologist. Analyze this chest X-ray and determine:\n"
        "1. Is there evidence of tuberculosis?\n"
        "2. What is your confidence level (Low/Medium/High)?\n"
        "3. What other findings are present?\n"
        "Provide a structured analysis."
    )
    
    prompt = prompt_template or default_prompt
    
    def format_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        """Format single example for TB detection"""
        # Determine label
        label = example.get("label", example.get("diagnosis", "unknown"))
        if isinstance(label, int):
            label_map = {0: "Normal", 1: "Tuberculosis", 2: "Other abnormality"}
            label = label_map.get(label, f"Class {label}")
        
        # Create messages
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"**Radiological Analysis:**\n\n"
                               f"**Primary Finding:** {label}\n"
                               f"**Confidence:** High\n\n"
                               f"**Recommendations:**\n"
                               f"- {'Refer for sputum testing and clinical evaluation' if 'tuberculosis' in label.lower() else 'No immediate intervention required'}\n"
                               f"- Follow standard TB screening protocol",
                    },
                ],
            },
        ]
        
        return example
        
    return format_fn


if __name__ == "__main__":
    # Example usage
    print("Fine-tuning module loaded successfully!")
    print("Use FineTuner class to fine-tune MedGemma models.")
    
    # Example workflow:
    # 1. loader = MedGemmaLoader()
    # 2. model, processor = loader.load_multimodal_model("medgemma-4b-it")
    # 3. finetuner = FineTuner(model, processor)
    # 4. finetuner.train(train_dataset, eval_dataset)
    # 5. finetuner.save_model()
