"""
Tests for HAI-DEF Foundation Model Loaders

This module tests the HeAR and MedSigLIP loaders that are critical
for the MedGemma Impact Challenge submission.

Test Coverage:
- HeAR: Audio embedding extraction, cough classification
- MedSigLIP: Image embedding extraction, zero-shot classification
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import io
import tempfile
import os

# Import the loaders
from src.models.hear_loader import HeARLoader, MockHeARLoader, HeARConfig
from src.models.medsiglip_loader import MedSigLIPLoader, MockMedSigLIPLoader, MedSigLIPConfig


class TestHeARLoader:
    """Tests for HeAR (Health Acoustic Representations) loader."""
    
    def test_config_defaults(self):
        """Test HeARConfig has correct defaults."""
        config = HeARConfig()
        
        assert config.model_path == "https://www.kaggle.com/models/google/hear/TensorFlow2/hear-base/1"
        assert config.sample_rate == 16000
        assert config.embedding_dim == 768
        assert config.segment_duration == 4.0
    
    def test_mock_loader_initialization(self):
        """Test MockHeARLoader initializes correctly."""
        loader = MockHeARLoader()
        
        assert loader.config is not None
        assert loader.config.embedding_dim == 768
        assert loader._loaded is True
    
    def test_mock_extract_embeddings(self):
        """Test mock embedding extraction."""
        loader = MockHeARLoader()
        
        # Create mock audio (16kHz, 4 seconds)
        audio = np.random.randn(64000).astype(np.float32)
        
        embeddings = loader.extract_embeddings(audio)
        
        assert embeddings.shape == (1, 768)
        assert embeddings.dtype == np.float32
    
    def test_mock_classify_cough(self):
        """Test mock cough classification for TB screening."""
        loader = MockHeARLoader()
        
        audio = np.random.randn(64000).astype(np.float32)
        
        result = loader.classify_cough(audio)
        
        assert "label" in result
        assert "confidence" in result
        assert "embeddings" in result
        assert result["label"] in ["healthy", "tb_suspected", "other_respiratory"]
        assert 0 <= result["confidence"] <= 1
    
    def test_mock_batch_extraction(self):
        """Test batch embedding extraction."""
        loader = MockHeARLoader()
        
        # Create batch of audio samples
        batch = [np.random.randn(64000).astype(np.float32) for _ in range(5)]
        
        embeddings = loader.extract_embeddings_batch(batch)
        
        assert embeddings.shape == (5, 768)
    
    def test_embedding_dimension_consistency(self):
        """Test that embeddings always have consistent dimensions."""
        loader = MockHeARLoader()
        
        # Test with different audio lengths
        short_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        long_audio = np.random.randn(160000).astype(np.float32)  # 10 seconds
        
        short_emb = loader.extract_embeddings(short_audio)
        long_emb = loader.extract_embeddings(long_audio)
        
        assert short_emb.shape == long_emb.shape == (1, 768)
    
    def test_invalid_audio_handling(self):
        """Test handling of invalid audio input."""
        loader = MockHeARLoader()
        
        # Empty audio should still work (return zero embeddings)
        empty_audio = np.array([], dtype=np.float32)
        
        embeddings = loader.extract_embeddings(empty_audio)
        
        assert embeddings.shape == (1, 768)


class TestMedSigLIPLoader:
    """Tests for MedSigLIP (Medical Image Embeddings) loader."""
    
    def test_config_defaults(self):
        """Test MedSigLIPConfig has correct defaults."""
        config = MedSigLIPConfig()
        
        assert config.base_model == "google/siglip-so400m-patch14-384"
        assert config.embedding_dim == 1152
        assert config.image_size == 384
    
    def test_mock_loader_initialization(self):
        """Test MockMedSigLIPLoader initializes correctly."""
        loader = MockMedSigLIPLoader()
        
        assert loader.config is not None
        assert loader.config.embedding_dim == 1152
        assert loader._loaded is True
    
    def test_mock_extract_embeddings(self):
        """Test mock image embedding extraction."""
        loader = MockMedSigLIPLoader()
        
        # Create test image
        image = Image.new("RGB", (384, 384), color="white")
        
        embeddings = loader.extract_embeddings(image)
        
        assert embeddings.shape == (1, 1152)
        assert isinstance(embeddings, np.ndarray)
    
    def test_zero_shot_classification(self):
        """Test zero-shot classification for medical images."""
        loader = MockMedSigLIPLoader()
        
        image = Image.new("RGB", (384, 384), color="gray")
        labels = ["normal", "pneumonia", "tuberculosis"]
        
        result = loader.zero_shot_classify(image, labels)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 3
        
        # Check probabilities sum to 1
        probs = [p["probability"] for p in result["predictions"]]
        assert abs(sum(probs) - 1.0) < 0.01
    
    def test_compute_similarity(self):
        """Test image similarity computation."""
        loader = MockMedSigLIPLoader()
        
        image1 = Image.new("RGB", (384, 384), color="white")
        image2 = Image.new("RGB", (384, 384), color="black")
        
        similarity = loader.compute_similarity(image1, image2)
        
        assert 0 <= similarity <= 1
    
    def test_batch_embedding_extraction(self):
        """Test batch image embedding extraction."""
        loader = MockMedSigLIPLoader()
        
        images = [Image.new("RGB", (384, 384), color="white") for _ in range(4)]
        
        embeddings = loader.extract_embeddings_batch(images)
        
        assert embeddings.shape == (4, 1152)
    
    def test_different_image_sizes(self):
        """Test handling of different input image sizes."""
        loader = MockMedSigLIPLoader()
        
        # Small image
        small = Image.new("RGB", (100, 100), color="white")
        # Large image
        large = Image.new("RGB", (1000, 1000), color="white")
        
        small_emb = loader.extract_embeddings(small)
        large_emb = loader.extract_embeddings(large)
        
        # Both should produce same embedding dimensions
        assert small_emb.shape == large_emb.shape == (1, 1152)
    
    def test_medical_labels_available(self):
        """Test that predefined medical labels are available."""
        loader = MockMedSigLIPLoader()
        
        assert hasattr(loader, 'medical_labels')
        assert "chest_xray" in loader.medical_labels
        assert "dermatology" in loader.medical_labels
        
        # Check TB-related labels exist
        chest_labels = loader.medical_labels["chest_xray"]
        assert "tuberculosis" in chest_labels or "tb" in [l.lower() for l in chest_labels]


class TestIntegration:
    """Integration tests for HAI-DEF model loaders."""
    
    def test_hear_medgemma_pipeline(self):
        """Test HeAR -> MedGemma pipeline for TB screening."""
        hear_loader = MockHeARLoader()
        
        # Simulate cough audio analysis
        audio = np.random.randn(64000).astype(np.float32)
        cough_result = hear_loader.classify_cough(audio)
        
        # Verify output can be used for downstream processing
        assert "embeddings" in cough_result
        assert cough_result["embeddings"].shape[-1] == 768
        
        # Embeddings should be suitable for MedGemma context
        context = f"Cough analysis: {cough_result['label']} (confidence: {cough_result['confidence']:.2f})"
        assert isinstance(context, str)
    
    def test_medsiglip_medgemma_pipeline(self):
        """Test MedSigLIP -> MedGemma pipeline for X-ray analysis."""
        siglip_loader = MockMedSigLIPLoader()
        
        # Simulate chest X-ray analysis
        image = Image.new("RGB", (384, 384), color="gray")
        
        # Get embeddings
        embeddings = siglip_loader.extract_embeddings(image)
        
        # Zero-shot classification
        labels = ["normal", "pneumonia", "tuberculosis", "other"]
        classification = siglip_loader.zero_shot_classify(image, labels)
        
        # Verify output format
        assert embeddings.shape == (1, 1152)
        assert len(classification["predictions"]) == 4
    
    def test_multimodal_screening(self):
        """Test combined audio + image screening workflow."""
        hear = MockHeARLoader()
        siglip = MockMedSigLIPLoader()
        
        # Audio analysis
        audio = np.random.randn(64000).astype(np.float32)
        audio_result = hear.classify_cough(audio)
        
        # Image analysis
        image = Image.new("RGB", (384, 384), color="gray")
        image_result = siglip.zero_shot_classify(
            image, 
            ["normal", "tuberculosis", "pneumonia"]
        )
        
        # Combine results (simplified fusion)
        audio_tb_score = 0.8 if audio_result["label"] == "tb_suspected" else 0.2
        image_tb_score = next(
            (p["probability"] for p in image_result["predictions"] 
             if p["label"] == "tuberculosis"),
            0.0
        )
        
        # Weighted combination
        combined_score = 0.4 * audio_tb_score + 0.6 * image_tb_score
        
        assert 0 <= combined_score <= 1


class TestEdgeCases:
    """Edge case tests for robustness."""
    
    def test_hear_with_silence(self):
        """Test HeAR with silent audio."""
        loader = MockHeARLoader()
        
        silent_audio = np.zeros(64000, dtype=np.float32)
        result = loader.extract_embeddings(silent_audio)
        
        assert result.shape == (1, 768)
    
    def test_medsiglip_with_blank_image(self):
        """Test MedSigLIP with blank/uniform image."""
        loader = MockMedSigLIPLoader()
        
        blank = Image.new("RGB", (384, 384), color="white")
        result = loader.extract_embeddings(blank)
        
        assert result.shape == (1, 1152)
    
    def test_hear_with_noise(self):
        """Test HeAR with pure noise."""
        loader = MockHeARLoader()
        
        noise = np.random.randn(64000).astype(np.float32) * 0.1
        result = loader.classify_cough(noise)
        
        # Should still return valid structure
        assert "label" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
