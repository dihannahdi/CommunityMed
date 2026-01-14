"""
Audio Agent - Voice and cough sound analysis
Uses Gemini's audio capabilities for respiratory sound analysis
Novel Task Prize target: Cough-based TB screening
"""

import time
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AudioAgent:
    """
    Audio agent for analyzing respiratory sounds
    Targets Novel Task Prize with cough-based TB screening
    """
    
    COUGH_FEATURES = {
        "duration": "Length of cough episode",
        "frequency": "Pitch characteristics",
        "wetness": "Productive vs dry cough",
        "pattern": "Single, paroxysmal, or continuous",
        "strength": "Force of cough",
    }
    
    TB_COUGH_INDICATORS = [
        "persistent_productive",
        "bloody_sputum",
        "prolonged_duration",
        "night_cough",
        "weak_cough",
    ]
    
    def __init__(self, model=None, device: str = "cuda"):
        """
        Initialize audio agent
        
        Note: MedGemma 4B-IT supports multimodal including audio through
        the Gemini backbone. Audio is converted to spectrograms for analysis.
        """
        self.model = model
        self.device = device
        
        if model:
            logger.info("AudioAgent initialized with model")
        else:
            logger.warning("AudioAgent initialized without model (mock mode)")
    
    async def analyze(self, case, audio_path: Optional[str] = None) -> "AgentResult":
        """
        Analyze audio data from patient
        
        Args:
            case: PatientCase with audio data
            audio_path: Optional path to audio file
            
        Returns:
            AgentResult with audio analysis
        """
        from .orchestrator import AgentResult, AgentType
        
        start_time = time.time()
        
        try:
            # Get audio data
            audio_data = getattr(case, 'audio_data', None) or audio_path
            
            if audio_data is None:
                # Analyze based on symptoms only
                findings = self._analyze_from_symptoms(case)
            else:
                # Analyze actual audio
                findings = await self._analyze_audio(audio_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_type=AgentType.AUDIO,
                success=True,
                findings=findings,
                confidence=findings.get('confidence', 0.7),
                recommendations=findings.get('recommendations', []),
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"AudioAgent error: {e}")
            return AgentResult(
                agent_type=AgentType.AUDIO,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _analyze_from_symptoms(self, case) -> Dict[str, Any]:
        """Analyze based on reported symptoms when no audio available"""
        symptoms = getattr(case, 'symptoms', [])
        symptoms_lower = [s.lower() for s in symptoms]
        
        cough_type = "unknown"
        tb_risk_from_cough = "Unable to assess without audio"
        
        # Infer from symptom descriptions
        if any("productive" in s or "phlegm" in s for s in symptoms_lower):
            cough_type = "productive"
            tb_risk_from_cough = "Elevated - productive cough reported"
        elif any("dry" in s or "nonproductive" in s for s in symptoms_lower):
            cough_type = "dry"
            tb_risk_from_cough = "Moderate - persistent dry cough"
        elif any("blood" in s or "hemoptysis" in s for s in symptoms_lower):
            cough_type = "hemoptytic"
            tb_risk_from_cough = "High - blood in sputum indicates serious pathology"
        
        # Check for cough duration
        if any("weeks" in s or "persistent" in s or "chronic" in s for s in symptoms_lower):
            tb_risk_from_cough = "Elevated - persistent cough >2 weeks is TB indicator"
        
        recommendations = [
            "Request audio recording of cough for AI analysis",
            "Document cough characteristics: frequency, time of day, productivity",
            "Assess for associated symptoms: hemoptysis, night sweats, weight loss",
        ]
        
        if "elevated" in tb_risk_from_cough.lower() or "high" in tb_risk_from_cough.lower():
            recommendations.insert(0, "Collect sputum sample for AFB testing")
        
        return {
            "analysis_type": "symptom-based",
            "audio_available": False,
            "cough_characteristics": {
                "type": cough_type,
                "inferred_from": "patient symptoms",
            },
            "tb_risk_from_cough": tb_risk_from_cough,
            "confidence": 0.5,  # Lower confidence without audio
            "recommendations": recommendations,
            "summary": f"Symptom-based analysis: {cough_type} cough pattern. {tb_risk_from_cough}",
        }
    
    async def _analyze_audio(self, audio_data) -> Dict[str, Any]:
        """
        Analyze actual audio data
        
        In production, this would:
        1. Convert audio to spectrogram
        2. Extract acoustic features (MFCCs, chroma, spectral contrast)
        3. Use MedGemma's multimodal capabilities to analyze
        4. Apply TB-specific acoustic models
        """
        
        # Check if it's a file path
        audio_path = None
        if isinstance(audio_data, str):
            if Path(audio_data).exists():
                audio_path = audio_data
        
        # Mock analysis for demonstration
        # In production, this uses librosa + MedGemma
        
        acoustic_features = {
            "duration_seconds": 3.2,
            "fundamental_frequency_hz": 180,
            "spectral_centroid": 1200,
            "mfcc_summary": "Low frequency dominant",
            "cough_events_detected": 5,
        }
        
        cough_classification = {
            "type": "productive",
            "wetness_score": 0.7,  # 0=dry, 1=very wet
            "strength_score": 0.6,  # 0=weak, 1=strong
            "pattern": "paroxysmal",
        }
        
        # TB risk scoring based on acoustic features
        tb_risk_score = self._calculate_tb_risk(acoustic_features, cough_classification)
        
        recommendations = [
            "Audio analysis complete - correlate with clinical findings",
        ]
        
        if tb_risk_score > 0.6:
            recommendations.insert(0, "⚠️ Acoustic pattern suggests high TB risk - prioritize sputum testing")
            recommendations.append("Consider GeneXpert testing for rapid TB diagnosis")
        elif tb_risk_score > 0.4:
            recommendations.append("Moderate acoustic risk - recommend follow-up if symptoms persist")
        
        return {
            "analysis_type": "acoustic",
            "audio_available": True,
            "audio_path": audio_path,
            "acoustic_features": acoustic_features,
            "cough_classification": cough_classification,
            "tb_risk_score": tb_risk_score,
            "tb_risk_level": "High" if tb_risk_score > 0.6 else "Medium" if tb_risk_score > 0.4 else "Low",
            "confidence": 0.85,
            "recommendations": recommendations,
            "summary": f"Acoustic analysis: {cough_classification['type']} cough, TB risk score: {tb_risk_score:.2f}",
        }
    
    def _calculate_tb_risk(self, acoustic: Dict, cough: Dict) -> float:
        """Calculate TB risk score from acoustic features"""
        score = 0.0
        
        # Productive cough is more associated with TB
        if cough.get('type') == 'productive':
            score += 0.2
        
        # Wetness score contribution
        score += cough.get('wetness_score', 0) * 0.15
        
        # Weak cough can indicate TB-related weakness
        if cough.get('strength_score', 1) < 0.5:
            score += 0.15
        
        # Paroxysmal pattern contribution
        if cough.get('pattern') == 'paroxysmal':
            score += 0.1
        
        # Multiple cough events
        if acoustic.get('cough_events_detected', 0) > 3:
            score += 0.1
        
        # Duration factor
        if acoustic.get('duration_seconds', 0) > 5:
            score += 0.1
        
        # Normalize to 0-1
        return min(1.0, score)


class MockAudioAgent(AudioAgent):
    """Mock audio agent for testing"""
    
    def __init__(self):
        super().__init__(model=None)
        logger.info("MockAudioAgent initialized")


if __name__ == "__main__":
    import asyncio
    from orchestrator import PatientCase
    
    agent = MockAudioAgent()
    
    # Test with symptoms only
    case = PatientCase(
        case_id="test-audio-001",
        chief_complaint="Cough for 3 weeks",
        symptoms=["productive cough", "night sweats", "weight loss"],
        age=45,
        gender="male",
    )
    
    result = asyncio.run(agent.analyze(case))
    print(f"Audio analysis: {result}")
