# CommunityMed AI - Manual VPS Deployment Guide

## Prerequisites Check

Before deploying, ensure your VPS has:
- Ubuntu 20.04+ or Debian 11+
- Python 3.10+
- Git installed
- At least 4GB RAM
- Port 8000 open for API access

## Step 1: Connect to VPS

```bash
ssh root@72.62.122.166
```

## Step 2: Quick Deployment (Automated)

Run this single command:

```bash
curl -fsSL https://raw.githubusercontent.com/dihannahdi/CommunityMed/main/deploy_vps.sh | bash
```

This will:
- Clone the repository to `/opt/communitymed`
- Install Python dependencies
- Create a systemd service
- Start the API on port 8000

## Step 3: Manual Deployment (If automated fails)

```bash
# Update system
apt update && apt install -y python3 python3-pip python3-venv git

# Clone repository
cd /opt
git clone https://github.com/dihannahdi/CommunityMed.git communitymed
cd communitymed

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API (test mode)
export USE_MOCK_AGENTS=true
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Step 4: Access the API

Open in browser:
- **Health Check**: http://72.62.122.166:8000/health
- **API Docs**: http://72.62.122.166:8000/docs
- **Swagger UI**: http://72.62.122.166:8000/redoc

## Step 5: Test the API

```bash
# From VPS
curl http://localhost:8000/health

# From your computer
curl http://72.62.122.166:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agents_loaded": true,
  "mode": "mock"
}
```

## Troubleshooting

### SSH Connection Timeout
```bash
# Check if VPS is reachable
ping 72.62.122.166

# Try different SSH port
ssh -p 2222 root@72.62.122.166
```

### Firewall Issues
```bash
# On VPS, allow port 8000
ufw allow 8000/tcp
ufw status
```

### API Not Starting
```bash
# Check logs
journalctl -u communitymed -f

# Or if running manually
tail -f /opt/communitymed/api.log
```

## Production Deployment (systemd)

Create `/etc/systemd/system/communitymed.service`:

```ini
[Unit]
Description=CommunityMed AI API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/communitymed
Environment=PATH=/opt/communitymed/venv/bin
Environment=USE_MOCK_AGENTS=true
ExecStart=/opt/communitymed/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl daemon-reload
systemctl enable communitymed
systemctl start communitymed
systemctl status communitymed
```

## Useful Commands

```bash
# View logs
journalctl -u communitymed -f

# Restart service
systemctl restart communitymed

# Stop service
systemctl stop communitymed

# Update code
cd /opt/communitymed
git pull origin main
systemctl restart communitymed
```
