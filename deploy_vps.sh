#!/bin/bash
# CommunityMed AI - VPS Deployment Script
# Run on VPS: bash <(curl -s https://raw.githubusercontent.com/dihannahdi/CommunityMed/main/deploy_vps.sh)

set -e

echo "=========================================="
echo "CommunityMed AI - VPS Deployment"
echo "=========================================="

# Configuration
DEPLOY_DIR="/opt/communitymed"
REPO_URL="https://github.com/dihannahdi/CommunityMed.git"
PYTHON_VERSION="3.11"

# Check system
echo "📋 System check..."
echo "OS: $(uname -a)"
echo "Python: $(python3 --version 2>/dev/null || echo 'Not installed')"
echo "Git: $(git --version 2>/dev/null || echo 'Not installed')"
echo "Docker: $(docker --version 2>/dev/null || echo 'Not installed')"
echo ""

# Update system
echo "🔄 Updating system packages..."
apt update -qq

# Install dependencies
echo "📦 Installing dependencies..."
apt install -y python3 python3-pip python3-venv git curl wget || true

# Create deployment directory
echo "📁 Creating deployment directory..."
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

# Clone or update repository
if [ -d ".git" ]; then
    echo "🔄 Updating existing repository..."
    git fetch origin
    git reset --hard origin/main
    git clean -fd
else
    echo "📥 Cloning repository..."
    rm -rf $DEPLOY_DIR/*
    git clone $REPO_URL .
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv

# Activate venv and install dependencies
echo "📚 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📂 Creating data directories..."
mkdir -p data checkpoints logs

# Create systemd service
echo "⚙️  Creating systemd service..."
cat > /etc/systemd/system/communitymed.service << 'EOF'
[Unit]
Description=CommunityMed AI API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/communitymed
Environment=PATH=/opt/communitymed/venv/bin:/usr/bin
Environment=USE_MOCK_AGENTS=true
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/communitymed/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
echo "🚀 Starting CommunityMed AI service..."
systemctl daemon-reload
systemctl enable communitymed
systemctl restart communitymed

# Wait for service to start
sleep 3

# Check service status
echo ""
echo "=========================================="
echo "📊 Service Status"
echo "=========================================="
systemctl status communitymed --no-pager || true

# Test API
echo ""
echo "🧪 Testing API..."
sleep 2
curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not responding yet"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "API: http://$(curl -s ifconfig.me):8000"
echo "Docs: http://$(curl -s ifconfig.me):8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs: journalctl -u communitymed -f"
echo "  Restart: systemctl restart communitymed"
echo "  Stop: systemctl stop communitymed"
echo "  Status: systemctl status communitymed"
echo ""
