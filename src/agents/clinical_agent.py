"""
Clinical Reasoning Agent - Medical synthesis and differential diagnosis
Uses MedGemma-27B-text for deep clinical reasoning
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from loguru import logger


class ClinicalAgent:
    """
    Clinical reasoning agent for symptom analysis and diagnosis
    Uses MedGemma-27B-text for comprehensive clinical synthesis
    """
    
    SYSTEM_PROMPT = """You are an expert clinical reasoning assistant helping Community Health Workers 
in resource-limited settings. Your role is to:

1. Synthesize patient symptoms, history, and any imaging findings
2. Generate prioritized differential diagnoses with probabilities
3. Identify red flags requiring immediate attention
4. Provide evidence-based management recommendations

Focus on conditions prevalent in community health settings:
- Respiratory infections (TB, pneumonia, bronchitis)
- Tropical diseases (malaria, dengue, typhoid)
- Maternal and child health conditions
- Non-communicable diseases (hypertension, diabetes)
- Common infectious diseases

Always prioritize patient safety and recommend physician consultation when appropriate."""

    CLINICAL_PROMPT = """Based on the following patient information, provide clinical analysis:

**Patient Demographics:**
- Age: {age}
- Gender: {gender}

**Chief Complaint:** {complaint}

**Reported Symptoms:** {symptoms}

**Duration:** {duration}

**Medical History:** {history}

**Imaging Findings (if available):** {imaging}

Please provide:
1. **Most Likely Diagnosis** with confidence level
2. **Differential Diagnoses** (top 3 with probabilities)
3. **Red Flags** requiring immediate attention
4. **Recommended Investigations** appropriate for community setting
5. **Management Recommendations** for the CHW
6. **When to Refer** - specific criteria for escalation"""

    def __init__(self, model=None, processor=None, device: str = "cuda"):
        """Initialize clinical reasoning agent"""
        self.model = model
        self.processor = processor
        self.device = device
        
        if model:
            logger.info("ClinicalAgent initialized with model")
        else:
            logger.warning("ClinicalAgent initialized without model (mock mode)")
    
    async def analyze(self, case) -> "AgentResult":
        """
        Perform clinical reasoning on patient case
        
        Args:
            case: PatientCase with symptoms and findings
            
        Returns:
            AgentResult with clinical analysis
        """
        from .orchestrator import AgentResult, AgentType
        
        start_time = time.time()
        
        try:
            # Extract case information
            age = getattr(case, 'age', None)
            gender = getattr(case, 'gender', None)
            complaint = getattr(case, 'chief_complaint', '')
            symptoms = getattr(case, 'symptoms', [])
            history = getattr(case, 'medical_history', '')
            
            # Get imaging findings from other agents
            imaging_findings = ""
            if hasattr(case, 'agent_results'):
                radiology_result = case.agent_results.get('radiology')
                if radiology_result and hasattr(radiology_result, 'findings'):
                    imaging_findings = str(radiology_result.findings.get('summary', ''))
            
            # Perform analysis
            analysis = await self._perform_analysis(
                age=age,
                gender=gender,
                complaint=complaint,
                symptoms=symptoms,
                history=history,
                imaging=imaging_findings,
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_type=AgentType.CLINICAL,
                success=True,
                findings={
                    "primary_finding": analysis.get("primary_diagnosis", ""),
                    "differential": analysis.get("differential", []),
                    "red_flags": analysis.get("red_flags", []),
                    "confidence": analysis.get("confidence", 0.0),
                    "summary": analysis.get("summary", ""),
                },
                confidence=analysis.get("confidence", 0.8),
                recommendations=analysis.get("recommendations", []),
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"ClinicalAgent error: {e}")
            return AgentResult(
                agent_type=AgentType.CLINICAL,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def _perform_analysis(
        self,
        age: Optional[int],
        gender: Optional[str],
        complaint: str,
        symptoms: List[str],
        history: str,
        imaging: str,
    ) -> Dict[str, Any]:
        """Perform clinical analysis"""
        
        if self.model is None:
            return self._mock_analysis(complaint, symptoms)
        
        # Format prompt
        prompt = self.CLINICAL_PROMPT.format(
            age=age or "Unknown",
            gender=gender or "Unknown",
            complaint=complaint or "Not specified",
            symptoms=", ".join(symptoms) if symptoms else "None",
            duration="As reported in chief complaint",
            history=history or "Not provided",
            imaging=imaging or "Not available",
        )
        
        # Create messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        # Generate with model
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
        )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured findings"""
        return {
            "primary_diagnosis": "Parsed from response",
            "differential": [],
            "red_flags": [],
            "confidence": 0.8,
            "recommendations": ["Based on model analysis"],
            "summary": response[:500],
        }
    
    def _mock_analysis(self, complaint: str, symptoms: List[str]) -> Dict[str, Any]:
        """Generate mock clinical analysis"""
        complaint_lower = complaint.lower()
        symptoms_lower = " ".join(symptoms).lower()
        
        # Simple rule-based mock
        if "cough" in symptoms_lower and "fever" in symptoms_lower:
            if "night sweats" in symptoms_lower or "weight loss" in symptoms_lower:
                return {
                    "primary_diagnosis": "Suspected Pulmonary Tuberculosis",
                    "differential": [
                        {"diagnosis": "Pulmonary TB", "probability": 0.6},
                        {"diagnosis": "Community-acquired Pneumonia", "probability": 0.25},
                        {"diagnosis": "Bronchitis", "probability": 0.15},
                    ],
                    "red_flags": ["Constitutional symptoms suggest TB", "Needs immediate evaluation"],
                    "confidence": 0.75,
                    "recommendations": [
                        "Urgent sputum AFB test required",
                        "Isolate patient if possible",
                        "Refer to TB treatment center",
                        "Contact tracing for household members",
                    ],
                    "summary": "Clinical presentation highly suggestive of pulmonary TB with constitutional symptoms.",
                }
            else:
                return {
                    "primary_diagnosis": "Respiratory Infection",
                    "differential": [
                        {"diagnosis": "Viral URI", "probability": 0.5},
                        {"diagnosis": "Bacterial pneumonia", "probability": 0.3},
                        {"diagnosis": "Bronchitis", "probability": 0.2},
                    ],
                    "red_flags": [],
                    "confidence": 0.7,
                    "recommendations": [
                        "Symptomatic treatment",
                        "Monitor for worsening",
                        "Return if fever persists >3 days",
                    ],
                    "summary": "Likely respiratory infection requiring symptomatic management.",
                }
        
        return {
            "primary_diagnosis": "Requires further evaluation",
            "differential": [],
            "red_flags": [],
            "confidence": 0.5,
            "recommendations": ["Complete clinical assessment recommended"],
            "summary": f"Mock analysis for: {complaint}",
        }


class MockClinicalAgent(ClinicalAgent):
    """Mock clinical agent for testing"""
    
    def __init__(self):
        super().__init__(model=None, processor=None)
        logger.info("MockClinicalAgent initialized")
    
    async def analyze(self, case) -> "AgentResult":
        """Generate mock clinical analysis"""
        from .orchestrator import AgentResult, AgentType
        
        mock = self._mock_analysis(
            getattr(case, 'chief_complaint', ''),
            getattr(case, 'symptoms', []),
        )
        
        return AgentResult(
            agent_type=AgentType.CLINICAL,
            success=True,
            findings={
                "primary_finding": mock["primary_diagnosis"],
                "differential": mock["differential"],
                "red_flags": mock["red_flags"],
                "confidence": mock["confidence"],
                "summary": mock["summary"],
            },
            confidence=mock["confidence"],
            recommendations=mock["recommendations"],
            processing_time_ms=75.0,
        )


if __name__ == "__main__":
    import asyncio
    from orchestrator import PatientCase
    
    agent = MockClinicalAgent()
    
    case = PatientCase(
        case_id="test-clinical-001",
        chief_complaint="Cough and fever for 2 weeks",
        symptoms=["cough", "fever", "night sweats", "weight loss"],
        age=40,
        gender="female",
    )
    
    result = asyncio.run(agent.analyze(case))
    print(f"Clinical analysis: {result}")
