"""
CommunityMed AI - Agents Module
Multi-agent system for TB screening and community health
"""

from .orchestrator import (
    Orchestrator,
    PatientCase,
    AgentResult,
    AgentType,
    TriageLevel,
)
from .radiology_agent import RadiologyAgent, MockRadiologyAgent
from .clinical_agent import ClinicalAgent, MockClinicalAgent
from .triage_agent import TriageAgent, MockTriageAgent
from .audio_agent import AudioAgent, MockAudioAgent


__all__ = [
    # Main orchestrator
    "Orchestrator",
    "PatientCase",
    "AgentResult",
    "AgentType",
    "TriageLevel",
    # Radiology
    "RadiologyAgent",
    "MockRadiologyAgent",
    # Clinical
    "ClinicalAgent",
    "MockClinicalAgent",
    # Triage
    "TriageAgent",
    "MockTriageAgent",
    # Audio
    "AudioAgent",
    "MockAudioAgent",
]
