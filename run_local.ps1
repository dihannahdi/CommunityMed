# CommunityMed AI - Local Simulation Script
# Test the system locally before VPS deployment

Write-Host "=========================================="
Write-Host "🏥 CommunityMed AI - Local Simulation"
Write-Host "=========================================="
Write-Host ""

# Check Python
Write-Host "📋 Checking requirements..."
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.10+"
    exit 1
}

# Create virtual environment if needed
if (-not (Test-Path "venv")) {
    Write-Host "`n🔧 Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "`n⚡ Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "`n📦 Installing dependencies..."
pip install -q -r requirements.txt

# Set environment variables
$env:USE_MOCK_AGENTS = "true"
$env:PYTHONPATH = $PWD

Write-Host "`n=========================================="
Write-Host "🚀 Starting CommunityMed AI API"
Write-Host "=========================================="
Write-Host ""
Write-Host "Mode: Mock Agents (Demo)"
Write-Host "Port: 8000"
Write-Host ""
Write-Host "Available endpoints:"
Write-Host "  🔍 Health: http://localhost:8000/health"
Write-Host "  📚 Docs: http://localhost:8000/docs"
Write-Host "  🎯 API: http://localhost:8000/api/v1/"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host "=========================================="
Write-Host ""

# Start API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
