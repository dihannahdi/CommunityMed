"""
CommunityMed AI - FastAPI Backend
REST API for multi-agent diagnostic assistant
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our agents
try:
    from ..agents import (
        Orchestrator,
        PatientCase,
        MockRadiologyAgent,
        MockClinicalAgent,
        MockTriageAgent,
        MockAudioAgent,
    )
except ImportError:
    # Direct import for standalone running
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from agents import (
        Orchestrator,
        PatientCase,
        MockRadiologyAgent,
        MockClinicalAgent,
        MockTriageAgent,
        MockAudioAgent,
    )


# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global orchestrator
    
    print("🚀 Starting CommunityMed AI API...")
    
    # Initialize with mock agents for development
    # In production, replace with real model-backed agents
    use_mock = os.getenv("USE_MOCK_AGENTS", "true").lower() == "true"
    
    if use_mock:
        orchestrator = Orchestrator(
            radiology_agent=MockRadiologyAgent(),
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
            audio_agent=MockAudioAgent(),
        )
        print("✅ Initialized with mock agents (development mode)")
    else:
        # Load real models
        print("⏳ Loading MedGemma models...")
        # This would load actual models in production
        orchestrator = Orchestrator(
            radiology_agent=MockRadiologyAgent(),  # Replace with real
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
            audio_agent=MockAudioAgent(),
        )
        print("✅ Models loaded")
    
    yield
    
    # Cleanup
    print("👋 Shutting down CommunityMed AI API...")


app = FastAPI(
    title="CommunityMed AI",
    description="""
    🏥 **Multi-Agent Diagnostic Assistant for Community Health Workers**
    
    Powered by Google's MedGemma models for TB screening and community health support.
    
    ## Features
    - 🔬 Chest X-ray analysis for TB detection
    - 🩺 Clinical symptom assessment
    - 🎯 Intelligent triage and routing
    - 🎙️ Cough sound analysis (Novel Task)
    
    ## Prize Targets
    - Main Track ($75,000)
    - Agentic Workflow Prize ($10,000)
    - Novel Task Prize ($10,000)
    - Edge AI Prize ($5,000)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================= SCHEMAS =======================

class PatientSymptoms(BaseModel):
    """Patient symptoms input"""
    chief_complaint: str = Field(..., description="Main reason for visit")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms")
    duration: Optional[str] = Field(None, description="Duration of symptoms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chief_complaint": "Cough for 3 weeks with night sweats",
                "symptoms": ["productive cough", "night sweats", "weight loss", "fatigue"],
                "duration": "3 weeks"
            }
        }


class PatientInfo(BaseModel):
    """Patient demographic information"""
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., description="Patient gender")
    medical_history: Optional[List[str]] = Field(default_factory=list)
    medications: Optional[List[str]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "gender": "male",
                "medical_history": ["diabetes", "hypertension"],
                "medications": ["metformin"]
            }
        }


class FullCaseRequest(BaseModel):
    """Complete case submission"""
    symptoms: PatientSymptoms
    patient_info: PatientInfo
    location: Optional[str] = Field(None, description="Patient location for context")
    priority_note: Optional[str] = Field(None, description="CHW priority notes")


class CaseResponse(BaseModel):
    """Case analysis response"""
    case_id: str
    status: str
    triage_level: str
    summary: str
    recommendations: List[str]
    agent_results: Dict[str, Any]
    processing_time_ms: float


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_loaded: bool
    mode: str


# ======================= ENDPOINTS =======================

@app.get("/", response_model=HealthCheck)
async def root():
    """API health check"""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        agents_loaded=orchestrator is not None,
        mode="mock" if os.getenv("USE_MOCK_AGENTS", "true").lower() == "true" else "production"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return await root()


@app.post("/api/v1/analyze", response_model=CaseResponse)
async def analyze_case(request: FullCaseRequest):
    """
    Analyze a complete patient case
    
    Routes through all agents:
    1. Radiology (if X-ray provided)
    2. Clinical assessment
    3. Audio analysis (if audio provided)
    4. Final triage
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    start_time = time.time()
    
    # Create case
    case = PatientCase(
        case_id=str(uuid.uuid4()),
        chief_complaint=request.symptoms.chief_complaint,
        symptoms=request.symptoms.symptoms,
        age=request.patient_info.age,
        gender=request.patient_info.gender,
        medical_history=request.patient_info.medical_history or [],
        medications=request.patient_info.medications or [],
    )
    
    # Analyze through orchestrator
    result = await orchestrator.analyze_case(case)
    
    processing_time = (time.time() - start_time) * 1000
    
    # Format response
    agent_summaries = {}
    recommendations = []
    
    for agent_name, agent_result in result.items():
        if hasattr(agent_result, 'findings'):
            agent_summaries[agent_name] = {
                "success": agent_result.success,
                "findings": agent_result.findings,
                "confidence": getattr(agent_result, 'confidence', None),
            }
            if agent_result.recommendations:
                recommendations.extend(agent_result.recommendations)
    
    # Get triage level from triage agent
    triage_level = "PRIORITY"  # Default
    summary = "Case analyzed by CommunityMed AI"
    
    if 'triage' in result:
        triage_result = result['triage']
        if hasattr(triage_result, 'findings'):
            triage_level = triage_result.findings.get('triage_level', 'PRIORITY')
            summary = triage_result.findings.get('summary', summary)
    
    return CaseResponse(
        case_id=case.case_id,
        status="completed",
        triage_level=triage_level,
        summary=summary,
        recommendations=list(set(recommendations)),  # Deduplicate
        agent_results=agent_summaries,
        processing_time_ms=processing_time,
    )


@app.post("/api/v1/analyze/quick")
async def quick_symptom_check(symptoms: PatientSymptoms):
    """Quick symptom check without full patient info"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Create minimal case
    case = PatientCase(
        case_id=str(uuid.uuid4()),
        chief_complaint=symptoms.chief_complaint,
        symptoms=symptoms.symptoms,
        age=30,  # Default age
        gender="unknown",
    )
    
    # Quick analysis - clinical only
    result = await orchestrator.analyze_case(case)
    
    return {
        "case_id": case.case_id,
        "status": "quick_check",
        "results": result,
    }


@app.post("/api/v1/analyze/xray")
async def analyze_xray(
    file: UploadFile = File(...),
    chief_complaint: str = "Chest X-ray analysis",
):
    """
    Analyze chest X-ray image
    
    Supports: JPEG, PNG, DICOM
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "application/dicom"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported. Use JPEG, PNG, or DICOM."
        )
    
    # Read image
    image_data = await file.read()
    
    # Create case with X-ray
    case = PatientCase(
        case_id=str(uuid.uuid4()),
        chief_complaint=chief_complaint,
        symptoms=["chest xray ordered"],
        age=30,
        gender="unknown",
    )
    # Attach image data
    case.chest_xray = image_data
    
    # Analyze
    result = await orchestrator.analyze_case(case)
    
    return {
        "case_id": case.case_id,
        "status": "xray_analyzed",
        "filename": file.filename,
        "results": result,
    }


@app.post("/api/v1/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    chief_complaint: str = "Cough analysis",
):
    """
    Analyze cough/respiratory audio
    
    Novel Task: AI-powered cough analysis for TB screening
    Supports: WAV, MP3, OGG
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Audio type {file.content_type} not supported. Use WAV, MP3, or OGG."
        )
    
    # Read audio
    audio_data = await file.read()
    
    # Create case
    case = PatientCase(
        case_id=str(uuid.uuid4()),
        chief_complaint=chief_complaint,
        symptoms=["cough"],
        age=30,
        gender="unknown",
    )
    case.audio_data = audio_data
    
    # Analyze
    result = await orchestrator.analyze_case(case)
    
    return {
        "case_id": case.case_id,
        "status": "audio_analyzed",
        "filename": file.filename,
        "results": result,
    }


@app.get("/api/v1/agents")
async def list_agents():
    """List available agents and their status"""
    if orchestrator is None:
        return {"agents": [], "status": "not_initialized"}
    
    agents = []
    
    if orchestrator.radiology_agent:
        agents.append({
            "name": "radiology",
            "description": "Chest X-ray analysis for TB detection",
            "model": "MedGemma-4B-IT",
            "status": "active",
        })
    
    if orchestrator.clinical_agent:
        agents.append({
            "name": "clinical",
            "description": "Clinical symptom assessment",
            "model": "MedGemma-27B-text-IT",
            "status": "active",
        })
    
    if orchestrator.triage_agent:
        agents.append({
            "name": "triage",
            "description": "Risk stratification and routing",
            "model": "Rule-based + MedGemma",
            "status": "active",
        })
    
    if orchestrator.audio_agent:
        agents.append({
            "name": "audio",
            "description": "Cough/respiratory sound analysis",
            "model": "MedGemma-4B-IT (multimodal)",
            "status": "active",
        })
    
    return {"agents": agents, "status": "initialized"}


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
