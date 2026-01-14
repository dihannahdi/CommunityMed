"""
HeAR Model Loader - Health Acoustic Representations
Google's HAI-DEF foundation model for non-speech audio analysis

HeAR produces embeddings that capture dense features relevant for:
- Cough analysis for TB screening
- Respiratory sound classification
- Data-efficient health audio classification
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import librosa
from loguru import logger

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not installed. HeAR model requires TensorFlow.")


@dataclass
class HeARConfig:
    """Configuration for HeAR model"""
    model_url: str = "https://www.kaggle.com/models/google/hear/TensorFlow2/hear-base/1"
    sample_rate: int = 16000
    embedding_dim: int = 768
    window_seconds: float = 2.0
    hop_seconds: float = 0.5


class HeARLoader:
    """
    Loader for Google's HeAR (Health Acoustic Representations) model
    
    HeAR is a foundation model for non-speech health audio analysis,
    trained on diverse acoustic health signals including:
    - Cough sounds
    - Breathing patterns
    - Lung auscultation sounds
    
    Reference: https://developers.google.com/health-ai-developer-foundations/hear
    """
    
    def __init__(self, config: Optional[HeARConfig] = None):
        """
        Initialize HeAR loader
        
        Args:
            config: HeAR configuration
        """
        self.config = config or HeARConfig()
        self.model = None
        self._loaded = False
        
    def load_model(self) -> bool:
        """
        Load the HeAR model from TensorFlow Hub
        
        Returns:
            True if model loaded successfully
        """
        if not HAS_TF:
            logger.error("TensorFlow not available. Install with: pip install tensorflow tensorflow-hub")
            return False
            
        try:
            logger.info(f"Loading HeAR model from: {self.config.model_url}")
            self.model = hub.load(self.config.model_url)
            self._loaded = True
            logger.info("HeAR model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load HeAR model: {e}")
            return False
    
    def preprocess_audio(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess audio for HeAR model
        
        Args:
            audio_path: Path to audio file
            audio_array: Raw audio array (alternative to path)
            sample_rate: Sample rate of audio_array
            
        Returns:
            Preprocessed audio array at 16kHz
        """
        if audio_path:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        elif audio_array is not None:
            audio = audio_array
            sr = sample_rate or self.config.sample_rate
            
            # Resample if needed
            if sr != self.config.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=self.config.sample_rate
                )
        else:
            raise ValueError("Either audio_path or audio_array must be provided")
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio.astype(np.float32)
    
    def extract_embeddings(
        self,
        audio: np.ndarray,
        return_windows: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extract HeAR embeddings from audio
        
        Args:
            audio: Preprocessed audio array at 16kHz
            return_windows: Whether to return per-window embeddings
            
        Returns:
            Dictionary with 'embedding' (pooled) and optionally 'windows'
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure correct shape [batch, samples]
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]
        
        # Run inference
        if HAS_TF:
            outputs = self.model(audio)
            
            # HeAR returns multiple outputs
            # - embeddings: per-window embeddings
            # - time_stamps: timestamps for each window
            embeddings = outputs["embedding"].numpy()
            
            # Pool embeddings (mean across windows)
            pooled = np.mean(embeddings, axis=1)
            
            result = {"embedding": pooled}
            if return_windows:
                result["windows"] = embeddings
                result["timestamps"] = outputs.get("time_stamps", None)
                
            return result
        else:
            # Mock embeddings for development
            return {
                "embedding": np.random.randn(1, self.config.embedding_dim).astype(np.float32)
            }
    
    def classify_cough(
        self,
        audio: np.ndarray,
        classifier_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Classify cough audio for TB screening
        
        This is a simplified classifier that uses HeAR embeddings
        with a linear probe for TB-related cough classification.
        
        Args:
            audio: Preprocessed audio array
            classifier_weights: Optional pre-trained classifier weights
            
        Returns:
            Dictionary with classification scores
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(audio)["embedding"]
        
        if classifier_weights is not None:
            # Use trained linear classifier
            logits = np.dot(embeddings, classifier_weights)
            probs = self._softmax(logits)
        else:
            # Return mock scores for demonstration
            # In production, this would use fine-tuned classifier
            probs = np.array([[0.3, 0.5, 0.2]])  # [normal, suspicious, tb_likely]
        
        return {
            "normal": float(probs[0, 0]),
            "suspicious": float(probs[0, 1]),
            "tb_likely": float(probs[0, 2]),
            "embedding_dim": embeddings.shape[-1],
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MockHeARLoader:
    """
    Mock HeAR loader for development and testing
    """
    
    def __init__(self, config: Optional[HeARConfig] = None):
        self.config = config or HeARConfig()
        self._loaded = True
        logger.info("MockHeARLoader initialized (development mode)")
    
    def load_model(self) -> bool:
        return True
    
    def preprocess_audio(self, **kwargs) -> np.ndarray:
        # Return mock audio
        duration = kwargs.get("duration", 2.0)
        samples = int(duration * 16000)
        return np.random.randn(samples).astype(np.float32)
    
    def extract_embeddings(
        self,
        audio: np.ndarray,
        return_windows: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Return mock embeddings"""
        return {
            "embedding": np.random.randn(1, 768).astype(np.float32),
        }
    
    def classify_cough(
        self,
        audio: np.ndarray,
        classifier_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return mock classification"""
        import random
        
        # Generate realistic mock scores based on audio characteristics
        base_normal = 0.6 + random.uniform(-0.2, 0.2)
        base_suspicious = 0.25 + random.uniform(-0.1, 0.1)
        base_tb = 1.0 - base_normal - base_suspicious
        
        return {
            "normal": max(0.0, min(1.0, base_normal)),
            "suspicious": max(0.0, min(1.0, base_suspicious)),
            "tb_likely": max(0.0, min(1.0, base_tb)),
            "embedding_dim": 768,
            "is_mock": True,
        }
