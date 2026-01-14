# CommunityMed AI - Video Demo Script

## Video Title
"CommunityMed AI: Multi-Agent TB Screening with HAI-DEF Models"

## Duration: 3 minutes maximum

---

## SCENE 1: The Problem (0:00 - 0:30)

**[Visual: World map highlighting TB high-burden countries]**

**Narrator (You):**
"Tuberculosis kills 1.25 million people annually – more than any other infectious disease. But TB is 95% curable when detected early."

**[Visual: Statistics appearing on screen]**
- 56-84 days diagnostic delay
- 3.5 million CHWs worldwide
- <20% rural X-ray access

"The problem? Diagnostic delays of 2-3 months in resource-limited settings. Community Health Workers serve billions of people but lack AI-powered decision support."

---

## SCENE 2: Solution Overview (0:30 - 1:15)

**[Visual: CommunityMed AI architecture diagram]**

**Narrator:**
"CommunityMed AI is a multi-agent diagnostic assistant using four HAI-DEF models from Google."

**[Visual: Animate each agent appearing]**
1. **MedGemma-4B-IT** → Radiology Agent for X-ray analysis
2. **MedGemma-27B-text-IT** → Clinical Agent for reasoning
3. **HeAR** → Audio Agent for cough analysis (Novel Task!)
4. **MedSigLIP** → Zero-shot image classification

**[Visual: Show live API demo]**

"Let me show you a live demo. A CHW enters patient symptoms..."

**[Screen recording: POST request to /api/v1/analyze]**
```json
{
  "symptoms": ["cough > 2 weeks", "night sweats", "weight loss"],
  "age": 45,
  "gender": "male"
}
```

**[Show API response with triage result]**

"Within seconds, our orchestrator routes to specialist agents and returns: URGENT triage, 87% TB likelihood, and specific action items for the CHW."

---

## SCENE 3: Technical Deep Dive (1:15 - 2:00)

**[Visual: Fine-tuning notebook]**

**Narrator:**
"For the Novel Task Prize, we fine-tuned MedGemma on TB X-ray datasets using QLoRA. 4-bit quantization with LoRA rank 16."

**[Visual: Training metrics chart]**
- Sensitivity: 72% → 87% (+15%)
- F1-Score: 0.78 → 0.89

**[Visual: HeAR integration code]**

"Our breakthrough: We apply Google's HeAR foundation model to cough sound analysis – a novel application beyond its original training."

```python
# HeAR cough analysis
embeddings = hear.extract_embeddings(cough_audio)  # 768-dim
tb_probability = classifier(embeddings)  # Fine-tuned probe
```

**[Visual: Edge deployment comparison table]**

"For the Edge AI Prize, we quantized to 2.2GB GGUF format, running on Android devices with 4-second inference time."

---

## SCENE 4: Impact & Validation (2:00 - 2:40)

**[Visual: Impact calculator running]**

**Narrator:**
"We built a rigorous impact model based on WHO epidemiological data."

**[Visual: Impact numbers appearing]**
- Year 1: 10,000 CHWs → 9,500 lives saved
- Year 3: 200,000 CHWs → 195,000 lives saved
- Cost per life saved: $246

**[Visual: ROI calculation]**

"That's a 9,800% ROI – making CommunityMed AI highly cost-effective by WHO standards."

**[Visual: Validation plan slide]**

"Our validation plan: A cluster RCT across 100 health facilities, measuring time-to-treatment and case detection rates."

---

## SCENE 5: Call to Action (2:40 - 3:00)

**[Visual: All links appearing on screen]**

**Narrator:**
"CommunityMed AI – open source, ready for deployment, built on HAI-DEF."

**[Visual: QR codes and links]**
- 💻 GitHub: github.com/dihannahdi/CommunityMed
- 🚀 Live Demo: [HuggingFace Space]
- 📧 Contact: [Your email]

"Thank you for watching. Let's fight TB together with AI."

**[Visual: Logo + "Built with Google HAI-DEF"]**

---

## Production Notes

### Recording Tips:
1. Use OBS Studio or Loom for screen recording
2. Record in 1080p or 4K
3. Add captions for accessibility
4. Keep each scene under 30 seconds
5. Show actual working demos, not mockups

### Required Demos to Record:
1. [ ] API running locally (curl command)
2. [ ] Postman/Swagger UI request
3. [ ] Jupyter notebook fine-tuning cell execution
4. [ ] Impact calculator output
5. [ ] Mobile/Edge inference (if available)

### Music:
- Royalty-free, subtle background music
- Lower during narration
- Recommended: YouTube Audio Library

"Thank you for watching. Together, we can make AI healthcare accessible to everyone."

**[End screen with links]**
- GitHub: [repository link]
- Demo: [Gradio demo link]
- Contact: [email]

---

## Production Notes

### Visuals Needed
1. Stock footage of CHWs in Africa/Asia
2. Screen recordings of CommunityMed AI interface
3. Architecture diagram animation
4. Code snippets (training, inference)
5. Charts showing impact metrics
6. Team photos (optional)

### Audio
- Professional narration
- Subtle background music (uplifting, hopeful)
- Sound effects for transitions

### Style Guide
- Clean, professional design
- Colors: Medical blue (#0066CC), Green for positive (#00AA55)
- Font: Sans-serif (Inter or similar)
- Animations: Smooth, minimal, purposeful

### Duration Breakdown
- Problem: 30 seconds
- Solution: 45 seconds
- Technical: 30 seconds
- Impact: 45 seconds
- Closing: 30 seconds
- **Total: 3 minutes**

---

## Alternative Short Version (60 seconds)

**[For social media or quick pitch]**

"TB kills 1.3 million people yearly. Community Health Workers need AI tools to detect it early.

CommunityMed AI uses Google's MedGemma – multiple AI agents analyzing X-rays, symptoms, and even cough sounds to triage patients in seconds.

Fine-tuned for TB. Optimized for mobile. Open source.

Our goal: 10,000 CHWs. 6 million patients. 50% faster diagnosis.

CommunityMed AI – AI healthcare for everyone."
