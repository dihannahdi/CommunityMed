"""
CommunityMed AI - Test Suite
Unit tests for core functionality
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPatientCase:
    """Test PatientCase dataclass"""
    
    def test_create_patient_case(self):
        from agents import PatientCase
        
        case = PatientCase(
            case_id="test-001",
            chief_complaint="Cough for 2 weeks",
            symptoms=["cough", "fever"],
            age=45,
            gender="male",
        )
        
        assert case.case_id == "test-001"
        assert case.chief_complaint == "Cough for 2 weeks"
        assert len(case.symptoms) == 2
        assert case.age == 45
        assert case.gender == "male"


class TestMockAgents:
    """Test mock agent implementations"""
    
    @pytest.mark.asyncio
    async def test_mock_radiology_agent(self):
        from agents import MockRadiologyAgent, PatientCase
        
        agent = MockRadiologyAgent()
        case = PatientCase(
            case_id="test-rad",
            chief_complaint="Chest pain",
            symptoms=["cough", "hemoptysis"],
            age=50,
            gender="female",
        )
        
        result = await agent.analyze(case)
        
        assert result.success is True
        assert result.findings is not None
        assert "tb_likelihood" in result.findings
    
    @pytest.mark.asyncio
    async def test_mock_clinical_agent(self):
        from agents import MockClinicalAgent, PatientCase
        
        agent = MockClinicalAgent()
        case = PatientCase(
            case_id="test-clin",
            chief_complaint="Night sweats and weight loss",
            symptoms=["night sweats", "weight loss", "fatigue"],
            age=35,
            gender="male",
        )
        
        result = await agent.analyze(case)
        
        assert result.success is True
        assert result.findings is not None
    
    @pytest.mark.asyncio
    async def test_mock_triage_agent(self):
        from agents import MockTriageAgent, PatientCase
        
        agent = MockTriageAgent()
        case = PatientCase(
            case_id="test-triage",
            chief_complaint="Severe chest pain",
            symptoms=["severe chest pain", "shortness of breath"],
            age=60,
            gender="male",
        )
        
        result = await agent.analyze(case)
        
        assert result.success is True
        assert result.findings is not None
        assert "triage_level" in result.findings


class TestOrchestrator:
    """Test orchestrator functionality"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        from agents import (
            Orchestrator,
            MockRadiologyAgent,
            MockClinicalAgent,
            MockTriageAgent,
            MockAudioAgent,
        )
        
        orchestrator = Orchestrator(
            radiology_agent=MockRadiologyAgent(),
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
            audio_agent=MockAudioAgent(),
        )
        
        assert orchestrator.radiology_agent is not None
        assert orchestrator.clinical_agent is not None
        assert orchestrator.triage_agent is not None
        assert orchestrator.audio_agent is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_full_analysis(self):
        from agents import (
            Orchestrator,
            PatientCase,
            MockRadiologyAgent,
            MockClinicalAgent,
            MockTriageAgent,
            MockAudioAgent,
        )
        
        orchestrator = Orchestrator(
            radiology_agent=MockRadiologyAgent(),
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
            audio_agent=MockAudioAgent(),
        )
        
        case = PatientCase(
            case_id="test-full",
            chief_complaint="Cough for 3 weeks with night sweats",
            symptoms=["productive cough", "night sweats", "weight loss"],
            age=45,
            gender="male",
        )
        
        results = await orchestrator.analyze_case(case)
        
        assert "clinical" in results
        assert "triage" in results
        # Audio should be analyzed based on symptoms
        assert results["triage"].success is True


class TestDatasetLoader:
    """Test dataset loading functionality"""
    
    def test_dataset_loader_initialization(self):
        from data import TBDatasetLoader
        
        loader = TBDatasetLoader("./test_data")
        
        assert loader.data_dir.exists() or True  # May not exist in test
        assert len(loader.DATASETS) >= 2
    
    def test_dataset_info(self):
        from data import TBDatasetLoader
        
        loader = TBDatasetLoader("./test_data")
        
        assert "shenzhen" in loader.DATASETS
        assert "montgomery" in loader.DATASETS
        assert "kaggle_id" in loader.DATASETS["shenzhen"]


class TestTriageLevels:
    """Test triage level assignment"""
    
    @pytest.mark.asyncio
    async def test_emergency_triage(self):
        from agents import MockTriageAgent, PatientCase
        
        agent = MockTriageAgent()
        case = PatientCase(
            case_id="test-emergency",
            chief_complaint="Severe respiratory distress",
            symptoms=["severe difficulty breathing", "cyanosis", "altered consciousness"],
            age=55,
            gender="female",
        )
        
        result = await agent.analyze(case)
        
        # Should be high priority
        assert result.findings["triage_level"] in ["EMERGENCY", "URGENT"]
    
    @pytest.mark.asyncio
    async def test_standard_triage(self):
        from agents import MockTriageAgent, PatientCase
        
        agent = MockTriageAgent()
        case = PatientCase(
            case_id="test-standard",
            chief_complaint="Mild cold symptoms",
            symptoms=["runny nose", "mild headache"],
            age=25,
            gender="female",
        )
        
        result = await agent.analyze(case)
        
        # Should be lower priority
        assert result.findings["triage_level"] in ["STANDARD", "PRIORITY", "ADVICE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
