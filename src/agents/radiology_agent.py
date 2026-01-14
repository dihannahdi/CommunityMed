"""
Radiology Agent - Chest X-ray and Medical Image Analysis
Uses MedGemma-4B-IT for multimodal radiology interpretation
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from PIL import Image
import torch

from loguru import logger


@dataclass
class RadiologyFinding:
    """Structured radiology finding"""
    finding: str
    location: str
    confidence: float
    severity: str  # "normal", "mild", "moderate", "severe"


class RadiologyAgent:
    """
    Specialist agent for radiology image analysis
    Focuses on chest X-ray interpretation for TB, pneumonia, and other conditions
    """
    
    # System prompt for radiology analysis
    SYSTEM_PROMPT = """You are an expert radiologist assistant specializing in chest X-ray interpretation 
for community health settings in resource-limited areas. Your analysis should be:

1. ACCURATE: Focus on findings visible in the image
2. STRUCTURED: Provide organized, actionable findings
3. CONSERVATIVE: When uncertain, recommend further evaluation
4. EDUCATIONAL: Help CHWs understand the findings

Focus areas:
- Tuberculosis (pulmonary TB patterns, cavitations, lymphadenopathy)
- Pneumonia (lobar consolidation, interstitial patterns, pleural effusion)
- Cardiomegaly and heart failure signs
- Other pulmonary abnormalities

Always provide:
1. Primary finding with confidence level
2. Secondary findings if present
3. Clear recommendation for the CHW"""

    # Prompt template for chest X-ray analysis
    XRAY_PROMPT = """Analyze this chest X-ray image and provide a structured assessment.

Patient context:
- Age: {age}
- Gender: {gender}
- Chief complaint: {complaint}
- Symptoms: {symptoms}

Please provide:
1. **Primary Finding**: Main observation with confidence (Low/Medium/High)
2. **TB Assessment**: Likelihood of tuberculosis and supporting features
3. **Other Findings**: Additional abnormalities detected
4. **Technical Quality**: Image quality assessment
5. **Recommendations**: Specific actions for the community health worker"""

    def __init__(
        self,
        model=None,
        processor=None,
        device: str = "cuda",
    ):
        """
        Initialize radiology agent
        
        Args:
            model: MedGemma multimodal model
            processor: Model processor
            device: Device to run inference on
        """
        self.model = model
        self.processor = processor
        self.device = device if torch.cuda.is_available() else "cpu"
        
        if model is not None:
            logger.info("RadiologyAgent initialized with model")
        else:
            logger.warning("RadiologyAgent initialized without model (mock mode)")
    
    async def analyze(self, case) -> "AgentResult":
        """
        Analyze radiology images in a patient case
        
        Args:
            case: PatientCase object with images
            
        Returns:
            AgentResult with findings
        """
        from .orchestrator import AgentResult, AgentType
        
        start_time = time.time()
        
        try:
            # Get images from case
            images = case.images if hasattr(case, 'images') else []
            
            if not images:
                return AgentResult(
                    agent_type=AgentType.RADIOLOGY,
                    success=False,
                    error="No images provided for analysis",
                )
            
            # Analyze each image
            all_findings = []
            for idx, image in enumerate(images):
                finding = await self._analyze_single_image(
                    image=image,
                    age=getattr(case, 'age', None),
                    gender=getattr(case, 'gender', None),
                    complaint=getattr(case, 'chief_complaint', ''),
                    symptoms=getattr(case, 'symptoms', []),
                )
                all_findings.append(finding)
            
            # Synthesize findings
            primary_finding = all_findings[0] if all_findings else {}
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_type=AgentType.RADIOLOGY,
                success=True,
                findings={
                    "primary_finding": primary_finding.get("primary", "Analysis complete"),
                    "tb_likelihood": primary_finding.get("tb_likelihood", "Low"),
                    "confidence": primary_finding.get("confidence", 0.0),
                    "all_findings": all_findings,
                    "summary": primary_finding.get("summary", ""),
                },
                confidence=primary_finding.get("confidence", 0.85),
                recommendations=primary_finding.get("recommendations", []),
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"RadiologyAgent error: {e}")
            return AgentResult(
                agent_type=AgentType.RADIOLOGY,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def _analyze_single_image(
        self,
        image,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        complaint: str = "",
        symptoms: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single radiology image
        
        Args:
            image: PIL Image or path
            age: Patient age
            gender: Patient gender
            complaint: Chief complaint
            symptoms: List of symptoms
            
        Returns:
            Dictionary with findings
        """
        symptoms = symptoms or []
        
        # Prepare image
        if isinstance(image, str):
            image = Image.open(image)
        if hasattr(image, 'convert'):
            image = image.convert("RGB")
        
        # Format prompt
        prompt = self.XRAY_PROMPT.format(
            age=age or "Unknown",
            gender=gender or "Unknown",
            complaint=complaint or "Not specified",
            symptoms=", ".join(symptoms) if symptoms else "None reported",
        )
        
        if self.model is None:
            # Mock response for testing
            return self._mock_analysis(complaint, symptoms)
        
        # Create message for model
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # Process with model
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured findings"""
        # Extract key information from response
        result = {
            "raw_response": response,
            "primary": "",
            "tb_likelihood": "Low",
            "confidence": 0.85,
            "summary": "",
            "recommendations": [],
        }
        
        response_lower = response.lower()
        
        # Determine TB likelihood
        if "high likelihood" in response_lower or "highly suggestive" in response_lower:
            result["tb_likelihood"] = "High"
            result["confidence"] = 0.9
        elif "moderate" in response_lower or "possible" in response_lower:
            result["tb_likelihood"] = "Medium"
            result["confidence"] = 0.75
        elif "tuberculosis" in response_lower:
            result["tb_likelihood"] = "Medium"
            result["confidence"] = 0.7
        else:
            result["tb_likelihood"] = "Low"
            result["confidence"] = 0.85
        
        # Extract primary finding
        if "primary finding" in response_lower:
            lines = response.split("\n")
            for line in lines:
                if "primary" in line.lower():
                    result["primary"] = line.split(":", 1)[-1].strip() if ":" in line else line
                    break
        
        # Generate recommendations based on findings
        if result["tb_likelihood"] == "High":
            result["recommendations"] = [
                "Urgent referral for sputum AFB testing",
                "Isolate patient if possible",
                "Contact tracing recommended",
                "Physician review required",
            ]
        elif result["tb_likelihood"] == "Medium":
            result["recommendations"] = [
                "Recommend sputum testing within 24-48 hours",
                "Clinical correlation advised",
                "Follow up in 1-2 days",
            ]
        else:
            result["recommendations"] = [
                "Routine follow-up",
                "Return if symptoms worsen",
            ]
        
        result["summary"] = response[:500] if len(response) > 500 else response
        
        return result
    
    def _mock_analysis(self, complaint: str, symptoms: List[str]) -> Dict[str, Any]:
        """Generate mock analysis for testing"""
        # Simple keyword-based mock
        complaint_lower = complaint.lower()
        symptoms_lower = " ".join(symptoms).lower()
        
        if "tb" in complaint_lower or "tuberculosis" in complaint_lower:
            tb_likelihood = "High"
        elif "cough" in symptoms_lower and "fever" in symptoms_lower:
            tb_likelihood = "Medium"
        else:
            tb_likelihood = "Low"
        
        return {
            "primary": "Mock analysis - normal chest X-ray appearance",
            "tb_likelihood": tb_likelihood,
            "confidence": 0.75,
            "summary": f"Mock radiology analysis based on: {complaint}",
            "recommendations": [
                "This is a mock analysis for testing",
                "Real model would provide detailed findings",
            ],
            "raw_response": "Mock response",
        }


class MockRadiologyAgent(RadiologyAgent):
    """Mock radiology agent for testing without model"""
    
    def __init__(self):
        super().__init__(model=None, processor=None)
        logger.info("MockRadiologyAgent initialized")
    
    async def analyze(self, case) -> "AgentResult":
        """Generate mock analysis"""
        from .orchestrator import AgentResult, AgentType
        
        mock_findings = self._mock_analysis(
            getattr(case, 'chief_complaint', ''),
            getattr(case, 'symptoms', []),
        )
        
        return AgentResult(
            agent_type=AgentType.RADIOLOGY,
            success=True,
            findings={
                "primary_finding": mock_findings["primary"],
                "tb_likelihood": mock_findings["tb_likelihood"],
                "confidence": mock_findings["confidence"],
                "summary": mock_findings["summary"],
            },
            confidence=mock_findings["confidence"],
            recommendations=mock_findings["recommendations"],
            processing_time_ms=50.0,
        )


if __name__ == "__main__":
    # Test radiology agent
    import asyncio
    from orchestrator import PatientCase
    
    agent = MockRadiologyAgent()
    
    case = PatientCase(
        case_id="test-001",
        chief_complaint="Persistent cough for 3 weeks",
        symptoms=["cough", "fever", "night sweats"],
        age=35,
        gender="male",
    )
    
    result = asyncio.run(agent.analyze(case))
    print(f"Analysis result: {result}")
