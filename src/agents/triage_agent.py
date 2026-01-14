"""
Triage Agent - Risk stratification and action prioritization
Synthesizes all agent findings to determine urgency and next steps
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass

from loguru import logger


class TriageAgent:
    """
    Triage agent for final risk stratification
    Combines findings from all specialist agents to determine patient priority
    """
    
    TRIAGE_CRITERIA = {
        "EMERGENCY": {
            "keywords": ["severe", "critical", "emergency", "life-threatening", "acute respiratory distress"],
            "conditions": ["suspected MI", "respiratory failure", "sepsis", "massive hemoptysis"],
            "action": "Immediate referral - emergency transport",
        },
        "URGENT": {
            "keywords": ["urgent", "high likelihood TB", "consolidation", "effusion"],
            "conditions": ["suspected TB", "pneumonia", "severe dehydration"],
            "action": "Same-day referral to health facility",
        },
        "PRIORITY": {
            "keywords": ["abnormal", "concerning", "moderate", "possible TB"],
            "conditions": ["persistent cough >2 weeks", "unexplained weight loss"],
            "action": "Referral within 24-48 hours",
        },
        "STANDARD": {
            "keywords": ["mild", "stable", "improving"],
            "conditions": ["common cold", "minor infection"],
            "action": "Community-level management with follow-up",
        },
        "ADVICE": {
            "keywords": ["normal", "healthy", "prevention"],
            "conditions": ["health education", "routine checkup"],
            "action": "Health education and prevention counseling",
        },
    }
    
    def __init__(self, model=None, processor=None, device: str = "cuda"):
        """Initialize triage agent"""
        self.model = model
        self.processor = processor
        self.device = device
        
        if model:
            logger.info("TriageAgent initialized with model")
        else:
            logger.warning("TriageAgent initialized without model (rule-based mode)")
    
    async def analyze(self, case) -> "AgentResult":
        """
        Perform triage assessment based on all available findings
        
        Args:
            case: PatientCase with agent_results from other agents
            
        Returns:
            AgentResult with triage level and recommendations
        """
        from .orchestrator import AgentResult, AgentType, TriageLevel
        
        start_time = time.time()
        
        try:
            # Gather all findings
            all_findings = self._gather_findings(case)
            
            # Determine triage level
            triage_level, reasoning = self._determine_triage(all_findings, case)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(triage_level, all_findings)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_type=AgentType.TRIAGE,
                success=True,
                findings={
                    "triage_level": triage_level,
                    "reasoning": reasoning,
                    "input_findings": all_findings,
                    "summary": f"Patient triaged as {triage_level}: {reasoning}",
                },
                confidence=0.9,
                recommendations=recommendations,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"TriageAgent error: {e}")
            return AgentResult(
                agent_type=AgentType.TRIAGE,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _gather_findings(self, case) -> Dict[str, Any]:
        """Gather findings from all previous agents"""
        findings = {
            "chief_complaint": getattr(case, 'chief_complaint', ''),
            "symptoms": getattr(case, 'symptoms', []),
            "age": getattr(case, 'age', None),
            "radiology": None,
            "clinical": None,
        }
        
        agent_results = getattr(case, 'agent_results', {})
        
        # Extract radiology findings
        if 'radiology' in agent_results:
            rad_result = agent_results['radiology']
            if hasattr(rad_result, 'findings'):
                findings['radiology'] = rad_result.findings
        
        # Extract clinical findings
        if 'clinical' in agent_results:
            clin_result = agent_results['clinical']
            if hasattr(clin_result, 'findings'):
                findings['clinical'] = clin_result.findings
        
        return findings
    
    def _determine_triage(self, findings: Dict[str, Any], case) -> tuple:
        """
        Determine triage level based on findings
        
        Returns:
            Tuple of (triage_level, reasoning)
        """
        # Convert all findings to searchable text
        findings_text = str(findings).lower()
        
        # Check for emergency indicators
        for level, criteria in self.TRIAGE_CRITERIA.items():
            keywords = criteria["keywords"]
            if any(kw in findings_text for kw in keywords):
                reasoning = f"Matched {level} criteria: keyword indicators present"
                return level, reasoning
        
        # Check radiology-specific findings
        if findings.get('radiology'):
            tb_likelihood = findings['radiology'].get('tb_likelihood', 'Low')
            if tb_likelihood == 'High':
                return "URGENT", "High TB likelihood on chest X-ray"
            elif tb_likelihood == 'Medium':
                return "PRIORITY", "Moderate TB likelihood - needs further evaluation"
        
        # Check clinical findings
        if findings.get('clinical'):
            red_flags = findings['clinical'].get('red_flags', [])
            if red_flags:
                return "URGENT", f"Clinical red flags identified: {red_flags[0]}"
        
        # Check symptoms
        symptoms = findings.get('symptoms', [])
        if len(symptoms) >= 4:
            return "PRIORITY", "Multiple symptoms requiring evaluation"
        
        # Default based on chief complaint
        complaint = findings.get('chief_complaint', '').lower()
        if any(word in complaint for word in ['severe', 'acute', 'sudden']):
            return "URGENT", "Concerning symptom description"
        elif complaint:
            return "PRIORITY", "Symptoms requiring medical evaluation"
        
        return "STANDARD", "No urgent findings identified"
    
    def _generate_recommendations(self, triage_level: str, findings: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on triage level"""
        
        base_recommendations = {
            "EMERGENCY": [
                "🚨 EMERGENCY: Arrange immediate transport to nearest hospital",
                "Do NOT delay - patient requires emergency care",
                "Stay with patient and monitor vital signs",
                "Call ahead to receiving facility",
                "Document and send case summary immediately",
            ],
            "URGENT": [
                "⚠️ URGENT: Patient needs same-day medical evaluation",
                "Arrange transport to health facility today",
                "Complete all documentation before referral",
                "Collect sputum sample if TB suspected",
                "Provide first aid and supportive care while waiting",
            ],
            "PRIORITY": [
                "📋 PRIORITY: Schedule appointment within 24-48 hours",
                "Document all symptoms and findings",
                "Provide symptom relief if appropriate",
                "Educate patient on warning signs",
                "Ensure patient has transport for appointment",
            ],
            "STANDARD": [
                "✅ STANDARD: Can be managed at community level",
                "Provide appropriate health education",
                "Prescribe OTC medications if indicated",
                "Schedule routine follow-up in 1 week",
                "Advise return if symptoms worsen",
            ],
            "ADVICE": [
                "💡 ADVICE: Health education and counseling only",
                "Discuss preventive health measures",
                "Address any questions or concerns",
                "Provide wellness resources",
            ],
        }
        
        recommendations = base_recommendations.get(triage_level, [])
        
        # Add condition-specific recommendations
        if findings.get('radiology'):
            tb = findings['radiology'].get('tb_likelihood', 'Low')
            if tb in ['High', 'Medium']:
                recommendations.append("TB-specific: Collect sputum for AFB testing")
                recommendations.append("Initiate infection control precautions")
        
        return recommendations


class MockTriageAgent(TriageAgent):
    """Mock triage agent for testing"""
    
    def __init__(self):
        super().__init__(model=None, processor=None)
        logger.info("MockTriageAgent initialized")


if __name__ == "__main__":
    import asyncio
    from orchestrator import PatientCase
    
    agent = MockTriageAgent()
    
    case = PatientCase(
        case_id="test-triage-001",
        chief_complaint="Severe chest pain",
        symptoms=["chest pain", "shortness of breath", "sweating"],
        age=55,
        gender="male",
    )
    
    result = asyncio.run(agent.analyze(case))
    print(f"Triage result: {result}")
