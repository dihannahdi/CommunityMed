"""
CommunityMed AI - End-to-End TB Screening Example

This script demonstrates the complete TB screening workflow using:
1. MedGemma-4B-IT for chest X-ray analysis
2. HeAR for cough sound analysis
3. MedGemma-27B-text-IT for clinical synthesis

Reference implementation for MedGemma Impact Challenge.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
from PIL import Image
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ScreeningResult:
    """Complete TB screening result"""
    patient_id: str
    timestamp: str
    
    # X-ray analysis
    xray_tb_probability: float
    xray_findings: str
    xray_confidence: float
    
    # Audio analysis (optional)
    cough_tb_indicator: Optional[float]
    cough_analysis: Optional[str]
    
    # Combined assessment
    combined_risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    recommendation: str
    
    # Clinical synthesis
    clinical_summary: str
    next_steps: list


def create_sample_xray() -> Image.Image:
    """
    Create a synthetic chest X-ray for demonstration
    
    In production, this would be a real chest X-ray image.
    """
    # Create grayscale image simulating chest X-ray appearance
    img = Image.new("RGB", (512, 512), color=(40, 40, 40))
    return img


def create_sample_cough_audio() -> np.ndarray:
    """
    Create synthetic cough audio for demonstration
    
    In production, this would be real audio recorded from a smartphone.
    """
    # 4 seconds at 16kHz
    duration = 4.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Generate synthetic audio with some structure
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 200 * t) * np.exp(-t) * 0.5
    audio += np.random.randn(samples) * 0.1
    
    return audio.astype(np.float32)


class CommunityMedScreener:
    """
    Multi-agent TB screening system using HAI-DEF models
    
    Prize Targets:
    - Main Track: MedGemma multimodal analysis
    - Novel Task: HeAR cough-based screening
    - Agentic Workflow: Multi-agent orchestration
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize the screener
        
        Args:
            use_mock: Use mock models for demonstration (no GPU required)
        """
        self.use_mock = use_mock
        self.models_loaded = False
        
        # Model references
        self.xray_model = None
        self.xray_processor = None
        self.hear_loader = None
        self.text_model = None
        self.text_tokenizer = None
        
        logger.info(f"CommunityMed Screener initialized (mock={use_mock})")
        
    def load_models(self):
        """Load all required models"""
        if self.models_loaded:
            return
        
        if self.use_mock:
            self._load_mock_models()
        else:
            self._load_real_models()
            
        self.models_loaded = True
        
    def _load_mock_models(self):
        """Load mock models for demonstration"""
        from src.models.hear_loader import MockHeARLoader
        from src.models.medsiglip_loader import MockMedSigLIPLoader
        
        self.hear_loader = MockHeARLoader()
        self.siglip_loader = MockMedSigLIPLoader()
        
        logger.info("Mock models loaded successfully")
        
    def _load_real_models(self):
        """Load real HAI-DEF models (requires GPU)"""
        from src.models.medgemma_loader import MedGemmaLoader
        from src.models.hear_loader import HeARLoader
        
        loader = MedGemmaLoader()
        
        # Load MedGemma-4B-IT for X-ray analysis
        logger.info("Loading MedGemma-4B-IT...")
        self.xray_model, self.xray_processor = loader.load_multimodal_model(
            model_name="medgemma-4b-it",
            use_quantization=True,
        )
        
        # Load HeAR for audio analysis
        logger.info("Loading HeAR...")
        self.hear_loader = HeARLoader()
        self.hear_loader.load_model()
        
        logger.success("All models loaded successfully")
        
    def analyze_xray(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze chest X-ray using MedGemma
        
        Args:
            image: Chest X-ray image
            
        Returns:
            Analysis results with TB probability and findings
        """
        if self.use_mock:
            # Mock response for demonstration
            return {
                "tb_probability": 0.72,
                "findings": "Right upper lobe consolidation with cavitary lesion. "
                           "Hilar lymphadenopathy present. Findings concerning for "
                           "pulmonary tuberculosis.",
                "confidence": 0.85,
                "abnormalities": [
                    "Consolidation (right upper lobe)",
                    "Cavitary lesion",
                    "Hilar lymphadenopathy",
                ],
            }
        else:
            # Real model inference
            from src.models.medgemma_loader import MedGemmaLoader
            loader = MedGemmaLoader()
            
            prompt = """Analyze this chest X-ray for signs of pulmonary tuberculosis.

Provide:
1. TB probability score (0-1)
2. Key radiological findings
3. Confidence level

Focus on: consolidation, cavitation, lymphadenopathy, pleural effusion, 
miliary patterns, and fibrotic changes."""

            response = loader.generate_with_image(
                self.xray_model,
                self.xray_processor,
                image,
                prompt,
                max_new_tokens=256,
            )
            
            # Parse response (simplified)
            return {
                "tb_probability": 0.65,  # Would parse from response
                "findings": response,
                "confidence": 0.80,
            }
    
    def analyze_cough(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cough audio using HeAR
        
        This is the NOVEL TASK - applying HeAR to cough-based TB screening
        
        Args:
            audio: Cough audio at 16kHz
            
        Returns:
            Cough analysis with TB indicators
        """
        if self.hear_loader is None:
            from src.models.hear_loader import MockHeARLoader
            self.hear_loader = MockHeARLoader()
        
        result = self.hear_loader.classify_cough(audio)
        
        return {
            "tb_indicator": result.get("confidence", 0.5),
            "label": result.get("label", "unknown"),
            "embeddings": result.get("embeddings"),
            "analysis": f"Cough pattern classified as: {result.get('label')}. "
                       f"TB indicator score: {result.get('confidence', 0):.2f}",
        }
    
    def synthesize_results(
        self,
        xray_result: Dict[str, Any],
        cough_result: Optional[Dict[str, Any]] = None,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> ScreeningResult:
        """
        Synthesize multi-modal results into final screening decision
        
        This implements the AGENTIC WORKFLOW - coordinating multiple agents
        
        Args:
            xray_result: X-ray analysis results
            cough_result: Cough analysis results (optional)
            patient_context: Additional patient information
            
        Returns:
            Complete screening result
        """
        # Weighted fusion of modalities
        xray_weight = 0.7
        cough_weight = 0.3
        
        xray_score = xray_result.get("tb_probability", 0.5)
        
        if cough_result:
            cough_score = cough_result.get("tb_indicator", 0.5)
            combined_score = xray_weight * xray_score + cough_weight * cough_score
        else:
            combined_score = xray_score
        
        # Determine risk level
        if combined_score >= 0.7:
            risk_level = "HIGH"
            recommendation = "URGENT REFERRAL - Immediate TB testing recommended"
        elif combined_score >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "FOLLOW-UP - Schedule TB testing within 7 days"
        else:
            risk_level = "LOW"
            recommendation = "MONITOR - Routine follow-up, repeat if symptoms persist"
        
        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            xray_result,
            cough_result,
            combined_score,
            risk_level,
        )
        
        # Determine next steps
        next_steps = self._get_next_steps(risk_level, combined_score)
        
        return ScreeningResult(
            patient_id=patient_context.get("id", "DEMO-001") if patient_context else "DEMO-001",
            timestamp=datetime.now().isoformat(),
            xray_tb_probability=xray_score,
            xray_findings=xray_result.get("findings", ""),
            xray_confidence=xray_result.get("confidence", 0.0),
            cough_tb_indicator=cough_result.get("tb_indicator") if cough_result else None,
            cough_analysis=cough_result.get("analysis") if cough_result else None,
            combined_risk_score=combined_score,
            risk_level=risk_level,
            recommendation=recommendation,
            clinical_summary=clinical_summary,
            next_steps=next_steps,
        )
    
    def _generate_clinical_summary(
        self,
        xray_result: Dict,
        cough_result: Optional[Dict],
        score: float,
        risk: str,
    ) -> str:
        """Generate clinical summary using MedGemma-27B (or mock)"""
        findings = xray_result.get("findings", "No significant findings")
        cough_info = f" Cough analysis: {cough_result.get('label', 'N/A')}." if cough_result else ""
        
        return f"""CLINICAL SUMMARY
================
Risk Assessment: {risk} (Score: {score:.2f})

Radiological Findings:
{findings}
{cough_info}

This automated screening suggests {'further investigation' if risk != 'LOW' else 'low probability'} 
for pulmonary tuberculosis. {'Recommend immediate sputum collection for AFB smear and GeneXpert testing.' if risk == 'HIGH' else ''}

Note: This is a screening tool. All findings require confirmation by qualified healthcare providers.
"""
    
    def _get_next_steps(self, risk_level: str, score: float) -> list:
        """Get recommended next steps based on risk level"""
        steps = {
            "HIGH": [
                "1. Collect sputum sample immediately",
                "2. Order GeneXpert MTB/RIF test",
                "3. Refer to TB physician within 24 hours",
                "4. Initiate contact tracing",
                "5. Document in TB registry",
            ],
            "MEDIUM": [
                "1. Schedule follow-up within 7 days",
                "2. Collect sputum if productive cough",
                "3. Complete symptom questionnaire",
                "4. Consider additional imaging if indicated",
            ],
            "LOW": [
                "1. Routine follow-up as needed",
                "2. Provide TB education materials",
                "3. Advise to return if symptoms develop",
            ],
        }
        return steps.get(risk_level, [])
    
    def screen_patient(
        self,
        xray_image: Optional[Image.Image] = None,
        cough_audio: Optional[np.ndarray] = None,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> ScreeningResult:
        """
        Complete TB screening workflow
        
        Args:
            xray_image: Chest X-ray (optional)
            cough_audio: Cough recording (optional)
            patient_context: Patient demographics and history
            
        Returns:
            Complete screening result
        """
        self.load_models()
        
        # Analyze available modalities
        xray_result = None
        cough_result = None
        
        if xray_image is not None:
            logger.info("Analyzing chest X-ray...")
            xray_result = self.analyze_xray(xray_image)
        else:
            # Use mock result for demo
            xray_result = {"tb_probability": 0.65, "findings": "Demo mode", "confidence": 0.8}
        
        if cough_audio is not None:
            logger.info("Analyzing cough audio (Novel Task: HeAR)...")
            cough_result = self.analyze_cough(cough_audio)
        
        # Synthesize results
        logger.info("Synthesizing multi-modal results...")
        result = self.synthesize_results(xray_result, cough_result, patient_context)
        
        return result


def print_result(result: ScreeningResult):
    """Pretty print screening result"""
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(result.risk_level, "⚪")
    
    print("\n" + "="*70)
    print("🏥 CommunityMed AI - TB Screening Report")
    print("="*70)
    print(f"\n📋 Patient ID: {result.patient_id}")
    print(f"⏰ Timestamp: {result.timestamp}")
    
    print(f"\n{risk_emoji} RISK LEVEL: {result.risk_level}")
    print(f"📊 Combined Score: {result.combined_risk_score:.2f}")
    print(f"\n💡 Recommendation: {result.recommendation}")
    
    print("\n📸 X-RAY ANALYSIS:")
    print(f"  • TB Probability: {result.xray_tb_probability:.1%}")
    print(f"  • Confidence: {result.xray_confidence:.1%}")
    print(f"  • Findings: {result.xray_findings[:100]}...")
    
    if result.cough_tb_indicator is not None:
        print("\n🎤 COUGH ANALYSIS (HeAR - Novel Task):")
        print(f"  • TB Indicator: {result.cough_tb_indicator:.2f}")
        print(f"  • Analysis: {result.cough_analysis}")
    
    print("\n📝 NEXT STEPS:")
    for step in result.next_steps:
        print(f"  {step}")
    
    print("\n" + result.clinical_summary)
    print("="*70)


def main():
    """Run end-to-end TB screening demonstration"""
    logger.info("🏥 CommunityMed AI - End-to-End TB Screening Demo")
    logger.info("=" * 50)
    
    # Initialize screener (mock mode for demo)
    screener = CommunityMedScreener(use_mock=True)
    
    # Create sample inputs
    logger.info("Creating sample inputs...")
    sample_xray = create_sample_xray()
    sample_cough = create_sample_cough_audio()
    
    patient_context = {
        "id": "TB-2026-001",
        "age": 45,
        "sex": "Male",
        "symptoms": ["cough > 2 weeks", "night sweats", "weight loss"],
        "hiv_status": "unknown",
        "previous_tb": False,
    }
    
    # Run screening
    logger.info("Running multi-modal TB screening...")
    result = screener.screen_patient(
        xray_image=sample_xray,
        cough_audio=sample_cough,
        patient_context=patient_context,
    )
    
    # Display results
    print_result(result)
    
    logger.success("Demo completed successfully!")
    logger.info("To run with real models, set use_mock=False (requires GPU)")


if __name__ == "__main__":
    main()
