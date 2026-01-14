"""
Orchestrator Agent - Central coordinator for multi-agent workflow
Routes patient cases to appropriate specialist agents and synthesizes results
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from loguru import logger


class TriageLevel(Enum):
    """Triage classification levels"""
    EMERGENCY = "EMERGENCY"     # Red - Immediate life-threatening
    URGENT = "URGENT"           # Orange - Serious, same-day care
    PRIORITY = "PRIORITY"       # Yellow - 24-48 hours
    STANDARD = "STANDARD"       # Green - Routine care
    ADVICE = "ADVICE"           # Blue - Health education only


class AgentType(Enum):
    """Types of specialist agents"""
    RADIOLOGY = "radiology"
    CLINICAL = "clinical"
    AUDIO = "audio"
    TRIAGE = "triage"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"


@dataclass
class PatientCase:
    """Represents a patient case for analysis"""
    case_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Input data
    chief_complaint: str = ""
    symptoms: List[str] = field(default_factory=list)
    medical_history: str = ""
    age: Optional[int] = None
    gender: Optional[str] = None
    
    # Multimodal inputs
    images: List[Any] = field(default_factory=list)  # X-rays, photos
    image_types: List[str] = field(default_factory=list)  # "chest_xray", "skin", etc.
    audio_recordings: List[Any] = field(default_factory=list)  # Lung sounds
    
    # Analysis results
    agent_results: Dict[str, Any] = field(default_factory=dict)
    final_triage: Optional[TriageLevel] = None
    recommendations: List[str] = field(default_factory=list)
    needs_physician_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "case_id": self.case_id,
            "timestamp": self.timestamp.isoformat(),
            "chief_complaint": self.chief_complaint,
            "symptoms": self.symptoms,
            "age": self.age,
            "gender": self.gender,
            "has_images": len(self.images) > 0,
            "image_types": self.image_types,
            "has_audio": len(self.audio_recordings) > 0,
            "triage_level": self.final_triage.value if self.final_triage else None,
            "needs_physician_review": self.needs_physician_review,
        }


@dataclass
class AgentResult:
    """Result from a specialist agent"""
    agent_type: AgentType
    success: bool
    findings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class Orchestrator:
    """
    Central orchestrator for the multi-agent diagnostic workflow
    Coordinates specialist agents and synthesizes results for CHWs
    """
    
    def __init__(
        self,
        radiology_agent=None,
        clinical_agent=None,
        audio_agent=None,
        triage_agent=None,
    ):
        """
        Initialize orchestrator with specialist agents
        
        Args:
            radiology_agent: Agent for image analysis (X-rays, etc.)
            clinical_agent: Agent for clinical reasoning
            audio_agent: Agent for audio analysis (lung sounds)
            triage_agent: Agent for risk stratification
        """
        self.agents = {
            AgentType.RADIOLOGY: radiology_agent,
            AgentType.CLINICAL: clinical_agent,
            AgentType.AUDIO: audio_agent,
            AgentType.TRIAGE: triage_agent,
        }
        
        self.case_history: List[PatientCase] = []
        logger.info("Orchestrator initialized with agents: " + 
                   str([k.value for k, v in self.agents.items() if v is not None]))
    
    def route_case(self, case: PatientCase) -> List[AgentType]:
        """
        Determine which agents should analyze this case
        
        Args:
            case: Patient case to route
            
        Returns:
            List of agent types to use
        """
        agents_to_use = []
        
        # Always use clinical reasoning
        agents_to_use.append(AgentType.CLINICAL)
        
        # Route based on available data
        if case.images:
            for img_type in case.image_types:
                if img_type in ["chest_xray", "xray", "radiograph"]:
                    agents_to_use.append(AgentType.RADIOLOGY)
                elif img_type in ["skin", "lesion", "rash"]:
                    agents_to_use.append(AgentType.DERMATOLOGY)
                elif img_type in ["fundus", "retina", "eye"]:
                    agents_to_use.append(AgentType.OPHTHALMOLOGY)
                else:
                    # Default to radiology for unknown image types
                    agents_to_use.append(AgentType.RADIOLOGY)
        
        # Add audio analysis if lung sounds available
        if case.audio_recordings:
            agents_to_use.append(AgentType.AUDIO)
        
        # Always finish with triage
        agents_to_use.append(AgentType.TRIAGE)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in agents_to_use:
            if agent not in seen:
                seen.add(agent)
                unique_agents.append(agent)
                
        logger.info(f"Case {case.case_id} routed to: {[a.value for a in unique_agents]}")
        return unique_agents
    
    async def analyze_case(self, case: PatientCase) -> PatientCase:
        """
        Analyze a patient case using appropriate specialist agents
        
        Args:
            case: Patient case to analyze
            
        Returns:
            Updated case with analysis results
        """
        logger.info(f"Starting analysis for case {case.case_id}")
        
        # Determine which agents to use
        agents_to_use = self.route_case(case)
        
        # Process with each agent
        for agent_type in agents_to_use:
            agent = self.agents.get(agent_type)
            
            if agent is None:
                logger.warning(f"Agent {agent_type.value} not available, using fallback")
                result = self._fallback_analysis(agent_type, case)
            else:
                try:
                    result = await agent.analyze(case)
                except Exception as e:
                    logger.error(f"Agent {agent_type.value} failed: {e}")
                    result = AgentResult(
                        agent_type=agent_type,
                        success=False,
                        error=str(e),
                    )
            
            case.agent_results[agent_type.value] = result
            
            # Update recommendations from each agent
            if result.success and result.recommendations:
                case.recommendations.extend(result.recommendations)
        
        # Determine final triage level
        case.final_triage = self._determine_triage(case)
        
        # Check if physician review needed
        case.needs_physician_review = self._needs_physician_review(case)
        
        # Store in history
        self.case_history.append(case)
        
        logger.success(f"Case {case.case_id} analyzed. Triage: {case.final_triage.value}")
        return case
    
    def _fallback_analysis(self, agent_type: AgentType, case: PatientCase) -> AgentResult:
        """
        Provide fallback analysis when agent is not available
        
        Args:
            agent_type: Type of agent that was requested
            case: Patient case
            
        Returns:
            Fallback result
        """
        return AgentResult(
            agent_type=agent_type,
            success=True,
            findings={"note": "Agent not available, manual review recommended"},
            confidence=0.0,
            recommendations=["Manual review by healthcare provider recommended"],
        )
    
    def _determine_triage(self, case: PatientCase) -> TriageLevel:
        """
        Determine final triage level based on all agent results
        
        Args:
            case: Case with agent results
            
        Returns:
            Final triage level
        """
        # Get triage agent result if available
        triage_result = case.agent_results.get(AgentType.TRIAGE.value)
        if triage_result and triage_result.success:
            level = triage_result.findings.get("triage_level")
            if level:
                try:
                    return TriageLevel(level)
                except ValueError:
                    pass
        
        # Fallback: analyze based on findings
        emergency_keywords = ["emergency", "critical", "severe", "acute", "life-threatening"]
        urgent_keywords = ["urgent", "serious", "concerning", "positive", "abnormal"]
        
        all_findings = json.dumps(case.agent_results).lower()
        
        if any(kw in all_findings for kw in emergency_keywords):
            return TriageLevel.EMERGENCY
        elif any(kw in all_findings for kw in urgent_keywords):
            return TriageLevel.URGENT
        elif case.images or case.symptoms:
            return TriageLevel.PRIORITY
        else:
            return TriageLevel.STANDARD
    
    def _needs_physician_review(self, case: PatientCase) -> bool:
        """
        Determine if case needs physician review
        
        Args:
            case: Analyzed case
            
        Returns:
            True if physician review is needed
        """
        # Always flag emergency and urgent cases
        if case.final_triage in [TriageLevel.EMERGENCY, TriageLevel.URGENT]:
            return True
        
        # Flag low-confidence results
        for result in case.agent_results.values():
            if isinstance(result, AgentResult) and result.confidence < 0.7:
                return True
        
        # Flag if any agent failed
        for result in case.agent_results.values():
            if isinstance(result, AgentResult) and not result.success:
                return True
        
        return False
    
    def get_case_summary(self, case: PatientCase) -> Dict[str, Any]:
        """
        Generate a summary suitable for CHW display
        
        Args:
            case: Analyzed case
            
        Returns:
            Summary dictionary
        """
        # Color coding for triage
        triage_colors = {
            TriageLevel.EMERGENCY: "#FF0000",  # Red
            TriageLevel.URGENT: "#FF8C00",     # Orange
            TriageLevel.PRIORITY: "#FFD700",   # Yellow
            TriageLevel.STANDARD: "#32CD32",   # Green
            TriageLevel.ADVICE: "#1E90FF",     # Blue
        }
        
        return {
            "case_id": case.case_id,
            "timestamp": case.timestamp.isoformat(),
            "triage": {
                "level": case.final_triage.value if case.final_triage else "UNKNOWN",
                "color": triage_colors.get(case.final_triage, "#808080"),
            },
            "needs_physician_review": case.needs_physician_review,
            "key_findings": self._extract_key_findings(case),
            "recommendations": list(set(case.recommendations)),  # Deduplicate
            "next_steps": self._get_next_steps(case),
        }
    
    def _extract_key_findings(self, case: PatientCase) -> List[str]:
        """Extract key findings from all agent results"""
        findings = []
        
        for agent_type, result in case.agent_results.items():
            if isinstance(result, AgentResult) and result.success:
                if "primary_finding" in result.findings:
                    findings.append(f"{agent_type}: {result.findings['primary_finding']}")
                elif "summary" in result.findings:
                    findings.append(f"{agent_type}: {result.findings['summary']}")
                    
        return findings
    
    def _get_next_steps(self, case: PatientCase) -> List[str]:
        """Get actionable next steps for CHW"""
        steps = []
        
        if case.final_triage == TriageLevel.EMERGENCY:
            steps.append("🚨 IMMEDIATE REFERRAL to nearest health facility")
            steps.append("Arrange emergency transport if available")
            steps.append("Contact physician/supervisor immediately")
        elif case.final_triage == TriageLevel.URGENT:
            steps.append("⚠️ SAME-DAY referral to health facility")
            steps.append("Document all symptoms and findings")
            steps.append("Provide first aid if applicable")
        elif case.final_triage == TriageLevel.PRIORITY:
            steps.append("📋 Schedule appointment within 24-48 hours")
            steps.append("Monitor for worsening symptoms")
            steps.append("Provide self-care instructions")
        else:
            steps.append("✅ Can be managed at community level")
            steps.append("Provide health education")
            steps.append("Schedule routine follow-up if needed")
        
        if case.needs_physician_review:
            steps.append("📱 Send case for physician review via app")
            
        return steps


# Convenience function for creating orchestrator
def create_orchestrator(
    medgemma_model=None,
    medgemma_processor=None,
    use_mock_agents: bool = False,
) -> Orchestrator:
    """
    Factory function to create orchestrator with agents
    
    Args:
        medgemma_model: Loaded MedGemma model
        medgemma_processor: Model processor
        use_mock_agents: Use mock agents for testing
        
    Returns:
        Configured Orchestrator instance
    """
    if use_mock_agents:
        # Import mock agents for testing
        from .radiology_agent import MockRadiologyAgent
        from .clinical_agent import MockClinicalAgent
        from .triage_agent import MockTriageAgent
        
        return Orchestrator(
            radiology_agent=MockRadiologyAgent(),
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
        )
    else:
        # Import real agents
        from .radiology_agent import RadiologyAgent
        from .clinical_agent import ClinicalAgent
        from .triage_agent import TriageAgent
        
        return Orchestrator(
            radiology_agent=RadiologyAgent(medgemma_model, medgemma_processor),
            clinical_agent=ClinicalAgent(medgemma_model, medgemma_processor),
            triage_agent=TriageAgent(medgemma_model, medgemma_processor),
        )


if __name__ == "__main__":
    # Test the orchestrator
    import uuid
    
    # Create mock orchestrator
    orchestrator = Orchestrator()
    
    # Create test case
    case = PatientCase(
        case_id=str(uuid.uuid4())[:8],
        chief_complaint="Persistent cough for 3 weeks",
        symptoms=["cough", "fever", "night sweats", "weight loss"],
        age=35,
        gender="male",
        image_types=["chest_xray"],
    )
    
    print(f"Test case created: {case.case_id}")
    print(f"Chief complaint: {case.chief_complaint}")
    print(f"Would route to: {[a.value for a in orchestrator.route_case(case)]}")
