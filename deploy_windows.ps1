# CommunityMed AI - Windows to VPS Deployment Script
# Run: .\deploy_windows.ps1

$VPS_IP = "72.62.122.166"
$VPS_USER = "root"
$DEPLOY_DIR = "/opt/communitymed"

Write-Host "=========================================="
Write-Host "CommunityMed AI - VPS Deployment"
Write-Host "=========================================="

# Test SSH connection
Write-Host "`n🔍 Testing SSH connection..."
ssh -o ConnectTimeout=5 ${VPS_USER}@${VPS_IP} "echo 'SSH connection successful'" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ SSH connection failed. Please check:"
    Write-Host "  1. VPS is running and accessible"
    Write-Host "  2. SSH key is configured"
    Write-Host "  3. Firewall allows SSH (port 22)"
    Write-Host "`nTry manually: ssh ${VPS_USER}@${VPS_IP}"
    exit 1
}

Write-Host "✅ SSH connection successful"

# Deploy using the deployment script
Write-Host "`n🚀 Running deployment script on VPS..."
Write-Host "This will:"
Write-Host "  - Clone the repository"
Write-Host "  - Install dependencies"
Write-Host "  - Start the API service on port 8000"
Write-Host ""

$deployCommand = @"
curl -fsSL https://raw.githubusercontent.com/dihannahdi/CommunityMed/main/deploy_vps.sh | bash
"@

ssh ${VPS_USER}@${VPS_IP} $deployCommand

Write-Host "`n=========================================="
Write-Host "✅ Deployment initiated!"
Write-Host "=========================================="
Write-Host "`nAPI will be available at:"
Write-Host "  http://${VPS_IP}:8000"
Write-Host "  http://${VPS_IP}:8000/docs (Swagger UI)"
Write-Host ""
