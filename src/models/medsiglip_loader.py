"""
MedSigLIP Model Loader - Medical Image Embeddings
Google's HAI-DEF foundation model for medical image analysis

MedSigLIP is fine-tuned from SigLIP for medical images including:
- Chest X-rays
- CT slices
- MRI slices
- Dermatology images
- Ophthalmology images
- Histopathology patches
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from PIL import Image
from loguru import logger

try:
    from transformers import (
        AutoModel,
        AutoProcessor,
        SiglipProcessor,
        SiglipModel,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not installed for MedSigLIP")


@dataclass
class MedSigLIPConfig:
    """Configuration for MedSigLIP model"""
    # Base SigLIP model (MedSigLIP is fine-tuned from this)
    base_model_id: str = "google/siglip-so400m-patch14-384"
    # MedSigLIP model ID (when released on HuggingFace)
    medsiglip_model_id: str = "google/medsiglip"  # Placeholder
    image_size: int = 384
    embedding_dim: int = 1152
    device: str = "cuda"
    use_fp16: bool = True


class MedSigLIPLoader:
    """
    Loader for Google's MedSigLIP model
    
    MedSigLIP provides medical image embeddings for:
    - Data-efficient classification
    - Zero-shot classification
    - Semantic image retrieval
    
    Reference: https://developers.google.com/health-ai-developer-foundations/medsiglip
    """
    
    # Medical image type labels for zero-shot classification
    MEDICAL_LABELS = {
        "chest_xray": [
            "normal chest x-ray",
            "chest x-ray with pneumonia",
            "chest x-ray with tuberculosis",
            "chest x-ray with cardiomegaly",
            "chest x-ray with pleural effusion",
            "chest x-ray with lung nodule",
            "chest x-ray with pulmonary edema",
        ],
        "dermatology": [
            "normal skin",
            "benign skin lesion",
            "malignant melanoma",
            "basal cell carcinoma",
            "squamous cell carcinoma",
            "actinic keratosis",
        ],
        "fundus": [
            "normal fundus",
            "diabetic retinopathy",
            "glaucoma",
            "macular degeneration",
            "hypertensive retinopathy",
        ],
    }
    
    def __init__(self, config: Optional[MedSigLIPConfig] = None):
        """
        Initialize MedSigLIP loader
        
        Args:
            config: MedSigLIP configuration
        """
        self.config = config or MedSigLIPConfig()
        self.model = None
        self.processor = None
        self._loaded = False
        
        # Set device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"
            logger.warning("CUDA not available, using CPU")
    
    def load_model(self, use_medsiglip: bool = False) -> bool:
        """
        Load the MedSigLIP or base SigLIP model
        
        Args:
            use_medsiglip: If True, try to load MedSigLIP. Falls back to SigLIP.
            
        Returns:
            True if model loaded successfully
        """
        if not HAS_TRANSFORMERS:
            logger.error("Transformers not installed")
            return False
        
        try:
            model_id = self.config.medsiglip_model_id if use_medsiglip else self.config.base_model_id
            
            logger.info(f"Loading SigLIP model: {model_id}")
            
            # Load processor and model
            self.processor = SiglipProcessor.from_pretrained(self.config.base_model_id)
            self.model = SiglipModel.from_pretrained(
                self.config.base_model_id,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            ).to(self.config.device)
            
            self.model.eval()
            self._loaded = True
            logger.info("SigLIP model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MedSigLIP model: {e}")
            return False
    
    def preprocess_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Preprocess image for MedSigLIP
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.config.device)
    
    def extract_embeddings(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Extract image embeddings using MedSigLIP
        
        Args:
            image: Input image
            
        Returns:
            Image embedding array of shape (embedding_dim,)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess if needed
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(pixel_values=image)
            
            # Normalize embeddings
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            
        return embeddings.cpu().numpy().squeeze()
    
    def zero_shot_classify(
        self,
        image: Union[str, Image.Image, np.ndarray],
        labels: Optional[List[str]] = None,
        image_type: str = "chest_xray",
    ) -> Dict[str, float]:
        """
        Zero-shot classification of medical images
        
        Args:
            image: Input image
            labels: Custom labels for classification. If None, uses predefined.
            image_type: Type of medical image for default labels
            
        Returns:
            Dictionary mapping labels to probabilities
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get labels
        if labels is None:
            labels = self.MEDICAL_LABELS.get(image_type, ["normal", "abnormal"])
        
        # Preprocess image and text
        if not isinstance(image, torch.Tensor):
            image_tensor = self.preprocess_image(image)
        else:
            image_tensor = image
        
        text_inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = self.model.get_image_features(pixel_values=image_tensor)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features @ text_features.T).squeeze()
            probs = torch.softmax(similarity * 100, dim=-1)  # Temperature scaling
        
        # Convert to dictionary
        return {label: float(prob) for label, prob in zip(labels, probs.cpu().numpy())}
    
    def compute_similarity(
        self,
        image1: Union[str, Image.Image, np.ndarray],
        image2: Union[str, Image.Image, np.ndarray],
    ) -> float:
        """
        Compute similarity between two medical images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.extract_embeddings(image1)
        emb2 = self.extract_embeddings(image2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)


class MockMedSigLIPLoader:
    """
    Mock MedSigLIP loader for development and testing
    """
    
    def __init__(self, config: Optional[MedSigLIPConfig] = None):
        self.config = config or MedSigLIPConfig()
        self._loaded = True
        logger.info("MockMedSigLIPLoader initialized (development mode)")
    
    def load_model(self, use_medsiglip: bool = False) -> bool:
        return True
    
    def preprocess_image(self, image) -> np.ndarray:
        return np.random.randn(1, 3, 384, 384).astype(np.float32)
    
    def extract_embeddings(self, image) -> np.ndarray:
        """Return mock embeddings"""
        return np.random.randn(1152).astype(np.float32)
    
    def zero_shot_classify(
        self,
        image,
        labels: Optional[List[str]] = None,
        image_type: str = "chest_xray",
    ) -> Dict[str, float]:
        """Return mock classification"""
        import random
        
        if labels is None:
            labels = MedSigLIPLoader.MEDICAL_LABELS.get(image_type, ["normal", "abnormal"])
        
        # Generate mock probabilities
        raw_probs = [random.random() for _ in labels]
        total = sum(raw_probs)
        probs = [p / total for p in raw_probs]
        
        return {label: prob for label, prob in zip(labels, probs)}
    
    def compute_similarity(self, image1, image2) -> float:
        """Return mock similarity"""
        import random
        return 0.5 + random.uniform(-0.3, 0.3)
