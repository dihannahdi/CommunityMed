"""
Models module for CommunityMed AI

Provides loaders for Google's HAI-DEF (Health AI Developer Foundations) models:
- MedGemma: Medical vision-language models
- HeAR: Health Acoustic Representations for audio analysis
- MedSigLIP: Medical image embeddings

Reference: https://developers.google.com/health-ai-developer-foundations
"""

from .medgemma_loader import MedGemmaLoader, get_model_info
from .fine_tuning import FineTuner, TrainingConfig, create_tb_format_fn

# HeAR for audio analysis (Novel Task)
try:
    from .hear_loader import HeARLoader, MockHeARLoader, HeARConfig
    HAS_HEAR = True
except ImportError:
    HAS_HEAR = False
    HeARLoader = None
    MockHeARLoader = None
    HeARConfig = None

# MedSigLIP for image embeddings
try:
    from .medsiglip_loader import MedSigLIPLoader, MockMedSigLIPLoader, MedSigLIPConfig
    HAS_MEDSIGLIP = True
except ImportError:
    HAS_MEDSIGLIP = False
    MedSigLIPLoader = None
    MockMedSigLIPLoader = None
    MedSigLIPConfig = None

__all__ = [
    # MedGemma
    "MedGemmaLoader",
    "get_model_info",
    # Fine-tuning
    "FineTuner",
    "TrainingConfig",
    "create_tb_format_fn",
    # HeAR (audio)
    "HeARLoader",
    "MockHeARLoader", 
    "HeARConfig",
    "HAS_HEAR",
    # MedSigLIP (embeddings)
    "MedSigLIPLoader",
    "MockMedSigLIPLoader",
    "MedSigLIPConfig",
    "HAS_MEDSIGLIP",
]
