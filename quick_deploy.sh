#!/bin/bash
# Quick deployment commands for VPS

echo "=== CommunityMed AI Deployment ==="
echo "System: $(uname -s)"
echo "User: $(whoami)"
echo "Directory: $(pwd)"

# Navigate to opt directory
cd /opt || exit 1

# Clone or update repository
if [ -d "communitymed" ]; then
  echo "Updating existing installation..."
  cd communitymed
  git pull origin main
else
  echo "Fresh installation..."
  git clone https://github.com/dihannahdi/CommunityMed.git communitymed
  cd communitymed
fi

echo "✅ Repository ready at: $(pwd)"
ls -la

# Install Python dependencies
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

echo "Installing dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed"

# Test the API in mock mode
echo "🧪 Testing API..."
export USE_MOCK_AGENTS=true
export PYTHONPATH=/opt/communitymed

# Start API in background for testing
nohup venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > /opt/communitymed/api.log 2>&1 &
API_PID=$!

echo "API started with PID: $API_PID"
sleep 3

# Test health endpoint
curl -s http://localhost:8000/health || echo "API not responding"

echo ""
echo "✅ Deployment complete!"
echo "API running on: http://$(curl -s ifconfig.me):8000"
echo "Logs: tail -f /opt/communitymed/api.log"
echo "Stop: kill $API_PID"
