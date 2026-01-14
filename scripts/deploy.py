#!/usr/bin/env python3
"""
Deployment Script for CommunityMed AI
Deploy API to VPS server
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Deploy CommunityMed AI to server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="72.62.122.166",
        help="Server host/IP",
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="root",
        help="SSH user",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=22,
        help="SSH port",
    )
    
    parser.add_argument(
        "--deploy-path",
        type=str,
        default="/opt/communitymed",
        help="Deployment path on server",
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API port",
    )
    
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Deploy using Docker",
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup environment, don't deploy code",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing",
    )
    
    return parser.parse_args()


def run_ssh(host: str, user: str, port: int, command: str, dry_run: bool = False):
    """Run command on remote server via SSH"""
    ssh_cmd = f'ssh -p {port} {user}@{host} "{command}"'
    
    if dry_run:
        print(f"[DRY RUN] {ssh_cmd}")
        return 0
    
    print(f"$ {command}")
    result = subprocess.run(ssh_cmd, shell=True)
    return result.returncode


def run_scp(host: str, user: str, port: int, local: str, remote: str, dry_run: bool = False):
    """Copy files to remote server via SCP"""
    scp_cmd = f"scp -P {port} -r {local} {user}@{host}:{remote}"
    
    if dry_run:
        print(f"[DRY RUN] {scp_cmd}")
        return 0
    
    print(f"$ scp {local} -> {remote}")
    result = subprocess.run(scp_cmd, shell=True)
    return result.returncode


def setup_server(args):
    """Setup server environment"""
    print("\n📦 Setting up server environment...")
    
    commands = [
        # Update system
        "apt update && apt upgrade -y",
        
        # Install Python and dependencies
        "apt install -y python3.11 python3.11-venv python3-pip git",
        
        # Install CUDA drivers (if GPU available)
        "apt install -y nvidia-driver-535 nvidia-cuda-toolkit || echo 'No NVIDIA GPU found'",
        
        # Create deployment directory
        f"mkdir -p {args.deploy_path}",
        
        # Create virtual environment
        f"python3.11 -m venv {args.deploy_path}/venv",
        
        # Install UV for faster package installation
        "pip install uv",
    ]
    
    for cmd in commands:
        result = run_ssh(args.host, args.user, args.port, cmd, args.dry_run)
        if result != 0:
            print(f"Warning: Command failed with code {result}")


def deploy_code(args):
    """Deploy code to server"""
    print("\n📁 Deploying code...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Files to deploy
    deploy_files = [
        "src",
        "config",
        "requirements.txt",
        "README.md",
    ]
    
    # Create tarball
    print("Creating deployment archive...")
    tar_cmd = f"tar -czf /tmp/communitymed.tar.gz -C {project_root} " + " ".join(deploy_files)
    
    if args.dry_run:
        print(f"[DRY RUN] {tar_cmd}")
    else:
        subprocess.run(tar_cmd, shell=True)
    
    # Copy to server
    run_scp(
        args.host, args.user, args.port,
        "/tmp/communitymed.tar.gz",
        "/tmp/communitymed.tar.gz",
        args.dry_run
    )
    
    # Extract on server
    run_ssh(
        args.host, args.user, args.port,
        f"tar -xzf /tmp/communitymed.tar.gz -C {args.deploy_path}",
        args.dry_run
    )


def install_dependencies(args):
    """Install Python dependencies on server"""
    print("\n📚 Installing dependencies...")
    
    commands = [
        # Activate venv and install requirements
        f"source {args.deploy_path}/venv/bin/activate && "
        f"pip install -r {args.deploy_path}/requirements.txt",
        
        # Install additional deployment dependencies
        f"source {args.deploy_path}/venv/bin/activate && "
        "pip install uvicorn gunicorn",
    ]
    
    for cmd in commands:
        run_ssh(args.host, args.user, args.port, cmd, args.dry_run)


def create_systemd_service(args):
    """Create systemd service for auto-start"""
    print("\n⚙️ Creating systemd service...")
    
    service_content = f'''[Unit]
Description=CommunityMed AI API
After=network.target

[Service]
Type=simple
User={args.user}
WorkingDirectory={args.deploy_path}
Environment=PATH={args.deploy_path}/venv/bin:/usr/bin
Environment=USE_MOCK_AGENTS=false
ExecStart={args.deploy_path}/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port {args.api_port}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
    
    if args.dry_run:
        print("[DRY RUN] Would create systemd service:")
        print(service_content)
        return
    
    # Write service file locally
    with open("/tmp/communitymed.service", "w") as f:
        f.write(service_content)
    
    # Copy to server
    run_scp(
        args.host, args.user, args.port,
        "/tmp/communitymed.service",
        "/etc/systemd/system/communitymed.service",
        args.dry_run
    )
    
    # Enable and start service
    run_ssh(args.host, args.user, args.port, "systemctl daemon-reload", args.dry_run)
    run_ssh(args.host, args.user, args.port, "systemctl enable communitymed", args.dry_run)
    run_ssh(args.host, args.user, args.port, "systemctl start communitymed", args.dry_run)


def deploy_docker(args):
    """Deploy using Docker"""
    print("\n🐳 Docker deployment...")
    
    # Create Dockerfile content
    dockerfile = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/

# Environment variables
ENV USE_MOCK_AGENTS=true
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    docker_compose = f'''version: '3.8'

services:
  api:
    build: .
    ports:
      - "{args.api_port}:8000"
    environment:
      - USE_MOCK_AGENTS=false
      - HF_TOKEN=${{HF_TOKEN}}
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped
    
  # Optional: Add GPU support
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: 1
  #           capabilities: [gpu]
'''
    
    if args.dry_run:
        print("[DRY RUN] Would create Docker files")
        print("\nDockerfile:")
        print(dockerfile)
        print("\ndocker-compose.yml:")
        print(docker_compose)
        return
    
    # Write files
    project_root = Path(__file__).parent.parent
    
    with open(project_root / "Dockerfile", "w") as f:
        f.write(dockerfile)
    
    with open(project_root / "docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("Created Dockerfile and docker-compose.yml")
    print(f"\nTo deploy with Docker:")
    print(f"  1. scp -r . {args.user}@{args.host}:{args.deploy_path}")
    print(f"  2. ssh {args.user}@{args.host}")
    print(f"  3. cd {args.deploy_path}")
    print(f"  4. docker-compose up -d")


def verify_deployment(args):
    """Verify deployment is working"""
    print("\n✅ Verifying deployment...")
    
    import time
    time.sleep(5)  # Wait for service to start
    
    # Check service status
    run_ssh(args.host, args.user, args.port, "systemctl status communitymed", args.dry_run)
    
    # Test API
    test_cmd = f"curl -s http://localhost:{args.api_port}/health"
    run_ssh(args.host, args.user, args.port, test_cmd, args.dry_run)
    
    print(f"\n🎉 Deployment complete!")
    print(f"API available at: http://{args.host}:{args.api_port}")
    print(f"Documentation at: http://{args.host}:{args.api_port}/docs")


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 60)
    print("CommunityMed AI - Deployment Script")
    print("=" * 60)
    print(f"Target: {args.user}@{args.host}")
    print(f"Deploy path: {args.deploy_path}")
    print(f"API port: {args.api_port}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Commands will not be executed")
    
    if args.use_docker:
        deploy_docker(args)
        return 0
    
    # Standard deployment
    if args.setup_only:
        setup_server(args)
    else:
        setup_server(args)
        deploy_code(args)
        install_dependencies(args)
        create_systemd_service(args)
        verify_deployment(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
