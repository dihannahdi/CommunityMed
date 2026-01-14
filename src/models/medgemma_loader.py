"""
MedGemma Model Loader - HAI-DEF Foundation Models
Handles loading MedGemma and other HAI-DEF models with proper quantization
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import login
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    model_id: str
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    attn_implementation: str = "eager"
    use_quantization: bool = True
    quantization_bits: int = 4


class MedGemmaLoader:
    """
    Loader for MedGemma and HAI-DEF models
    Supports both multimodal (4B-IT) and text-only (27B-text) variants
    """
    
    # Available HAI-DEF models
    AVAILABLE_MODELS = {
        "medgemma-4b-it": "google/medgemma-4b-it",
        "medgemma-27b-text-it": "google/medgemma-27b-text-it",
        "medgemma-4b-pt": "google/medgemma-4b-pt",
        "medgemma-1.5-4b-it": "google/medgemma-1.5-4b-it",
        "medgemma-27b-it": "google/medgemma-27b-it",
    }
    
    # Model types
    MULTIMODAL_MODELS = ["medgemma-4b-it", "medgemma-4b-pt", "medgemma-1.5-4b-it", "medgemma-27b-it"]
    TEXT_ONLY_MODELS = ["medgemma-27b-text-it"]
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the loader
        
        Args:
            hf_token: HuggingFace token for gated model access
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if self.hf_token:
            login(token=self.hf_token)
            logger.info("Logged in to HuggingFace Hub")
        else:
            logger.warning("No HF_TOKEN provided. Ensure you're logged in via CLI.")
            
        self._check_gpu_support()
        
    def _check_gpu_support(self):
        """Check GPU capabilities"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Models will run on CPU (very slow).")
            self.device = "cpu"
            return
            
        self.device = "cuda"
        capability = torch.cuda.get_device_capability()
        
        if capability[0] < 8:
            logger.warning(f"GPU compute capability {capability} < 8.0. BF16 may not be supported.")
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16
            logger.info(f"GPU supports BF16 (compute capability {capability})")
            
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Available GPU memory: {gpu_mem:.1f} GB")
        
    def get_quantization_config(self, bits: int = 4) -> BitsAndBytesConfig:
        """
        Get BitsAndBytes quantization configuration for QLoRA
        
        Args:
            bits: Quantization bits (4 or 8)
            
        Returns:
            BitsAndBytesConfig for model loading
        """
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_storage=self.torch_dtype,
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    
    def load_multimodal_model(
        self,
        model_name: str = "medgemma-4b-it",
        use_quantization: bool = True,
        quantization_bits: int = 4,
    ) -> Tuple[Any, Any]:
        """
        Load a multimodal MedGemma model (supports images + text)
        
        Args:
            model_name: Model identifier
            use_quantization: Whether to use quantization
            quantization_bits: Bits for quantization (4 or 8)
            
        Returns:
            Tuple of (model, processor)
        """
        if model_name not in self.MULTIMODAL_MODELS:
            raise ValueError(f"Model {model_name} is not multimodal. Use load_text_model instead.")
            
        model_id = self.AVAILABLE_MODELS.get(model_name, model_name)
        logger.info(f"Loading multimodal model: {model_id}")
        
        # Model kwargs
        model_kwargs = {
            "attn_implementation": "eager",
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
        }
        
        # Add quantization if requested
        if use_quantization:
            model_kwargs["quantization_config"] = self.get_quantization_config(quantization_bits)
            logger.info(f"Using {quantization_bits}-bit quantization")
        
        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_kwargs,
            token=self.hf_token,
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=self.hf_token,
        )
        
        # Set padding side to right for training
        processor.tokenizer.padding_side = "right"
        
        logger.success(f"Loaded {model_name} with {model.num_parameters()/1e9:.2f}B parameters")
        
        return model, processor
    
    def load_text_model(
        self,
        model_name: str = "medgemma-27b-text-it",
        use_quantization: bool = True,
        quantization_bits: int = 4,
    ) -> Tuple[Any, Any]:
        """
        Load a text-only MedGemma model
        
        Args:
            model_name: Model identifier
            use_quantization: Whether to use quantization
            quantization_bits: Bits for quantization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name not in self.TEXT_ONLY_MODELS:
            logger.warning(f"Model {model_name} may be multimodal. Consider load_multimodal_model.")
            
        model_id = self.AVAILABLE_MODELS.get(model_name, model_name)
        logger.info(f"Loading text model: {model_id}")
        
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
        }
        
        if use_quantization:
            model_kwargs["quantization_config"] = self.get_quantization_config(quantization_bits)
            logger.info(f"Using {quantization_bits}-bit quantization")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
            token=self.hf_token,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.hf_token,
        )
        tokenizer.padding_side = "right"
        
        logger.success(f"Loaded {model_name} with {model.num_parameters()/1e9:.2f}B parameters")
        
        return model, tokenizer
    
    def load_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load model(s) from YAML configuration file
        
        Args:
            config_path: Path to model_config.yaml
            
        Returns:
            Dictionary of loaded models and processors
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        loaded_models = {}
        
        for model_key, model_config in config.get("models", {}).items():
            model_name = model_config.get("name")
            model_type = model_config.get("type")
            use_quant = model_config.get("quantization", {}).get("enabled", True)
            
            try:
                if model_type == "multimodal" or model_type == "base":
                    model, processor = self.load_multimodal_model(
                        model_name=model_name.split("/")[-1],
                        use_quantization=use_quant,
                    )
                else:
                    model, processor = self.load_text_model(
                        model_name=model_name.split("/")[-1],
                        use_quantization=use_quant,
                    )
                    
                loaded_models[model_key] = {
                    "model": model,
                    "processor": processor,
                    "config": model_config,
                }
                
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")
                
        return loaded_models


def get_model_info(model_name: str = "medgemma-4b-it") -> Dict[str, Any]:
    """
    Get information about a MedGemma model without loading it
    
    Args:
        model_name: Model identifier
        
    Returns:
        Dictionary with model information
    """
    info = {
        "medgemma-4b-it": {
            "parameters": "4.3B",
            "type": "multimodal",
            "capabilities": ["radiology", "dermatology", "pathology", "ophthalmology"],
            "min_gpu_memory": "8GB (quantized), 16GB (full)",
            "recommended_for": ["chest_xray", "skin_lesion", "fundus"],
        },
        "medgemma-27b-text-it": {
            "parameters": "27B",
            "type": "text-only",
            "capabilities": ["clinical_reasoning", "diagnosis", "treatment_planning"],
            "min_gpu_memory": "24GB (quantized), 64GB (full)",
            "recommended_for": ["clinical_synthesis", "differential_diagnosis"],
        },
        "medgemma-4b-pt": {
            "parameters": "4.3B",
            "type": "base/pretrained",
            "capabilities": ["fine_tuning", "embedding_extraction"],
            "min_gpu_memory": "8GB (quantized)",
            "recommended_for": ["custom_fine_tuning", "domain_adaptation"],
        },
    }
    
    return info.get(model_name, {"error": f"Unknown model: {model_name}"})


if __name__ == "__main__":
    # Test model loading
    loader = MedGemmaLoader()
    
    # Print available models
    print("Available HAI-DEF Models:")
    for name, path in loader.AVAILABLE_MODELS.items():
        info = get_model_info(name)
        print(f"  - {name}: {info.get('parameters', 'N/A')} ({info.get('type', 'N/A')})")
    
    # Test loading (uncomment to actually load)
    # model, processor = loader.load_multimodal_model("medgemma-4b-it")
    # print(f"Model loaded successfully!")
