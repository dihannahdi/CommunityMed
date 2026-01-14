"""
CommunityMed AI - MedGemma Impact Challenge
An AI-powered diagnostic assistant for Community Health Workers
"""

__version__ = "1.0.0"
__author__ = "CommunityMed AI Team"
__license__ = "CC BY 4.0"

from .models import MedGemmaLoader, FineTuner, Quantizer
from .agents import Orchestrator, RadiologyAgent, ClinicalAgent, TriageAgent

__all__ = [
    "MedGemmaLoader",
    "FineTuner",
    "Quantizer",
    "Orchestrator",
    "RadiologyAgent",
    "ClinicalAgent",
    "TriageAgent",
]
