"""
CommunityMed AI - Gradio Demo Interface
Interactive web UI for TB screening demonstration
"""

import os
import asyncio
from typing import Optional

try:
    import gradio as gr
except ImportError:
    gr = None
    print("Warning: gradio not installed")

# Import our agents
try:
    from src.agents import (
        Orchestrator,
        PatientCase,
        MockRadiologyAgent,
        MockClinicalAgent,
        MockTriageAgent,
        MockAudioAgent,
    )
except ImportError:
    from agents import (
        Orchestrator,
        PatientCase,
        MockRadiologyAgent,
        MockClinicalAgent,
        MockTriageAgent,
        MockAudioAgent,
    )


# Global orchestrator
orchestrator: Optional[Orchestrator] = None


def initialize_agents():
    """Initialize agents for demo"""
    global orchestrator
    
    if orchestrator is None:
        orchestrator = Orchestrator(
            radiology_agent=MockRadiologyAgent(),
            clinical_agent=MockClinicalAgent(),
            triage_agent=MockTriageAgent(),
            audio_agent=MockAudioAgent(),
        )
    
    return orchestrator


def analyze_symptoms(
    chief_complaint: str,
    symptoms: str,
    age: int,
    gender: str,
    duration: str,
) -> tuple:
    """
    Analyze patient symptoms
    
    Returns: (summary, triage, recommendations, details)
    """
    if not chief_complaint:
        return "Please enter chief complaint", "", "", ""
    
    orchestrator = initialize_agents()
    
    # Parse symptoms
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    
    # Create case
    case = PatientCase(
        case_id="demo-001",
        chief_complaint=chief_complaint,
        symptoms=symptom_list,
        age=age,
        gender=gender,
    )
    
    # Analyze
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(orchestrator.analyze_case(case))
    loop.close()
    
    # Format output
    summary = "## Analysis Summary\n\n"
    triage_level = "PRIORITY"
    recommendations = []
    details = ""
    
    for agent_name, result in results.items():
        if hasattr(result, 'findings') and result.findings:
            details += f"### {agent_name.title()} Agent\n"
            
            if isinstance(result.findings, dict):
                summary_text = result.findings.get('summary', str(result.findings))
                details += f"{summary_text}\n\n"
                
                if agent_name == 'triage':
                    triage_level = result.findings.get('triage_level', 'PRIORITY')
            else:
                details += f"{result.findings}\n\n"
            
            if hasattr(result, 'recommendations') and result.recommendations:
                recommendations.extend(result.recommendations)
    
    summary += f"**Chief Complaint:** {chief_complaint}\n\n"
    summary += f"**Symptoms:** {', '.join(symptom_list) if symptom_list else 'None specified'}\n\n"
    summary += f"**Patient:** {age} year old {gender}\n\n"
    
    # Format triage with color
    triage_colors = {
        "EMERGENCY": "🔴",
        "URGENT": "🟠",
        "PRIORITY": "🟡",
        "STANDARD": "🟢",
        "ADVICE": "🔵",
    }
    triage_emoji = triage_colors.get(triage_level, "⚪")
    triage_display = f"{triage_emoji} **{triage_level}**"
    
    # Format recommendations
    rec_text = "\n".join([f"• {r}" for r in recommendations[:8]])  # Limit to 8
    
    return summary, triage_display, rec_text, details


def analyze_xray(image, chief_complaint: str = "Chest X-ray analysis"):
    """Analyze chest X-ray image"""
    if image is None:
        return "Please upload a chest X-ray image", "", ""
    
    orchestrator = initialize_agents()
    
    # Create case with image
    case = PatientCase(
        case_id="xray-demo",
        chief_complaint=chief_complaint,
        symptoms=["chest xray ordered"],
        age=40,
        gender="unknown",
    )
    case.chest_xray = image
    
    # Analyze
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(orchestrator.analyze_case(case))
    loop.close()
    
    # Extract radiology findings
    rad_result = results.get('radiology')
    
    if rad_result and hasattr(rad_result, 'findings'):
        findings = rad_result.findings
        
        summary = f"""## Chest X-ray Analysis

**TB Likelihood:** {findings.get('tb_likelihood', 'Unknown')}

**Key Findings:**
{chr(10).join(['• ' + f for f in findings.get('findings', ['No specific findings'])])}

**Confidence:** {findings.get('confidence', 0):.0%}
"""
        
        recommendations = "\n".join([
            f"• {r}" for r in findings.get('recommendations', [])
        ])
        
        return summary, recommendations, findings.get('summary', '')
    
    return "Analysis failed", "", ""


def analyze_audio(audio, chief_complaint: str = "Cough analysis"):
    """Analyze cough audio"""
    if audio is None:
        return "Please record or upload cough audio", "", ""
    
    orchestrator = initialize_agents()
    
    # Create case
    case = PatientCase(
        case_id="audio-demo",
        chief_complaint=chief_complaint,
        symptoms=["cough"],
        age=40,
        gender="unknown",
    )
    case.audio_data = audio
    
    # Analyze
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(orchestrator.analyze_case(case))
    loop.close()
    
    # Extract audio findings
    audio_result = results.get('audio')
    
    if audio_result and hasattr(audio_result, 'findings'):
        findings = audio_result.findings
        
        summary = f"""## Cough Audio Analysis

**Analysis Type:** {findings.get('analysis_type', 'Unknown')}

**TB Risk from Cough:** {findings.get('tb_risk_from_cough', 'Unable to assess')}

**Confidence:** {findings.get('confidence', 0):.0%}
"""
        
        recommendations = "\n".join([
            f"• {r}" for r in findings.get('recommendations', [])
        ])
        
        return summary, recommendations, findings.get('summary', '')
    
    return "Audio analysis not available", "", ""


def create_demo():
    """Create Gradio demo interface"""
    
    if gr is None:
        print("Gradio not installed. Run: pip install gradio")
        return None
    
    with gr.Blocks(
        title="CommunityMed AI",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown("""
        # 🏥 CommunityMed AI
        ### Multi-Agent Diagnostic Assistant for Community Health Workers
        
        Powered by Google's **MedGemma** models for TB screening and community health support.
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Symptom Analysis
            with gr.TabItem("🩺 Symptom Analysis"):
                gr.Markdown("### Enter patient symptoms for AI-powered assessment")
                
                with gr.Row():
                    with gr.Column():
                        chief_complaint = gr.Textbox(
                            label="Chief Complaint",
                            placeholder="e.g., Cough for 3 weeks with night sweats",
                            lines=2,
                        )
                        symptoms = gr.Textbox(
                            label="Symptoms (comma-separated)",
                            placeholder="e.g., productive cough, night sweats, weight loss, fatigue",
                            lines=2,
                        )
                        
                        with gr.Row():
                            age = gr.Slider(
                                label="Age",
                                minimum=0,
                                maximum=120,
                                value=40,
                                step=1,
                            )
                            gender = gr.Dropdown(
                                label="Gender",
                                choices=["male", "female", "other"],
                                value="male",
                            )
                        
                        duration = gr.Textbox(
                            label="Duration",
                            placeholder="e.g., 3 weeks",
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        summary_output = gr.Markdown(label="Summary")
                        triage_output = gr.Markdown(label="Triage Level")
                
                gr.Markdown("### Recommendations")
                recommendations_output = gr.Markdown()
                
                gr.Markdown("### Detailed Analysis")
                details_output = gr.Markdown()
                
                analyze_btn.click(
                    fn=analyze_symptoms,
                    inputs=[chief_complaint, symptoms, age, gender, duration],
                    outputs=[summary_output, triage_output, recommendations_output, details_output],
                )
                
                # Examples
                gr.Examples(
                    examples=[
                        ["Cough for 3 weeks with night sweats", "productive cough, night sweats, weight loss, fatigue, fever", 45, "male", "3 weeks"],
                        ["Chest pain and difficulty breathing", "sharp chest pain, shortness of breath, dry cough", 60, "female", "2 days"],
                        ["Mild cold symptoms", "runny nose, sneezing, mild headache", 25, "female", "3 days"],
                    ],
                    inputs=[chief_complaint, symptoms, age, gender, duration],
                )
            
            # Tab 2: X-ray Analysis
            with gr.TabItem("📷 Chest X-ray Analysis"):
                gr.Markdown("### Upload chest X-ray for TB screening")
                
                with gr.Row():
                    with gr.Column():
                        xray_image = gr.Image(
                            label="Chest X-ray",
                            type="filepath",
                        )
                        xray_complaint = gr.Textbox(
                            label="Clinical Context",
                            placeholder="e.g., Rule out TB",
                            value="Chest X-ray analysis for TB screening",
                        )
                        xray_btn = gr.Button("🔬 Analyze X-ray", variant="primary")
                    
                    with gr.Column():
                        xray_summary = gr.Markdown(label="Analysis")
                        xray_recommendations = gr.Markdown(label="Recommendations")
                        xray_details = gr.Markdown(label="Details")
                
                xray_btn.click(
                    fn=analyze_xray,
                    inputs=[xray_image, xray_complaint],
                    outputs=[xray_summary, xray_recommendations, xray_details],
                )
            
            # Tab 3: Audio Analysis (Novel Task)
            with gr.TabItem("🎙️ Cough Analysis"):
                gr.Markdown("""
                ### Novel Task: AI-Powered Cough Analysis for TB Screening
                
                Record or upload a cough sound for acoustic analysis.
                """)
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Cough Recording",
                            type="filepath",
                        )
                        audio_complaint = gr.Textbox(
                            label="Context",
                            value="Cough analysis for TB screening",
                        )
                        audio_btn = gr.Button("🔊 Analyze Cough", variant="primary")
                    
                    with gr.Column():
                        audio_summary = gr.Markdown(label="Analysis")
                        audio_recommendations = gr.Markdown(label="Recommendations")
                        audio_details = gr.Markdown(label="Details")
                
                audio_btn.click(
                    fn=analyze_audio,
                    inputs=[audio_input, audio_complaint],
                    outputs=[audio_summary, audio_recommendations, audio_details],
                )
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About CommunityMed AI
                
                CommunityMed AI is a multi-agent diagnostic assistant designed to support 
                Community Health Workers (CHWs) in resource-limited settings.
                
                ### 🎯 Competition Targets
                
                | Prize Track | Prize | Our Solution |
                |-------------|-------|--------------|
                | Main Track | $75,000 | Comprehensive TB screening platform |
                | Agentic Workflow | $10,000 | Multi-agent orchestration with specialist agents |
                | Novel Task | $10,000 | Cough-based TB screening using audio AI |
                | Edge AI | $5,000 | Quantized models for mobile deployment |
                
                ### 🏗️ Architecture
                
                ```
                Patient Case → Orchestrator → [Radiology Agent]
                                           → [Clinical Agent]
                                           → [Audio Agent]
                                           → [Triage Agent] → CHW Recommendations
                ```
                
                ### 🔬 Technology
                
                - **Models:** Google MedGemma (4B & 27B variants)
                - **Fine-tuning:** QLoRA with 4-bit quantization
                - **Framework:** HAI-DEF compliant
                - **Deployment:** FastAPI + Gradio
                
                ### 📊 Impact Metrics
                
                - Target: 10,000+ CHWs across Sub-Saharan Africa and South Asia
                - Goal: 50% reduction in TB diagnostic delays
                - Focus: WHO high-burden TB countries
                
                ---
                
                Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge)
                """)
        
        gr.Markdown("""
        ---
        ⚠️ **Disclaimer:** This is a demonstration tool for research purposes only. 
        Not intended for clinical diagnosis. Always consult healthcare professionals.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    if demo:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
        )
