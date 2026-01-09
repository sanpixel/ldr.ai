#!/usr/bin/env python3
"""
Deploy source archive to Google Cloud Run using --no-build flag.

This script:
1. Authenticates with GCP
2. Deploys archive to Cloud Run with --no-build flag
3. Verifies service is running
4. Handles deployment errors
"""

import os
import sys
import subprocess
import time
import requests
from typing import Dict, List, Optional

# Configuration
PYTHON_BASE_IMAGE = "python311"
PORT = "8080"
STARTUP_COMMAND = "streamlit"
STARTUP_ARGS = ["main.py"]
REGION = "us-central1"
MAX_WAIT_TIME = 300  # 5 minutes
WAIT_INTERVAL = 10  # seconds

def run_command(cmd: List[str], capture_output: bool = True) -> tuple:
    """Run shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def authenticate_gcp(credentials_path: Optional[str] = None) -> bool:
    """Authenticate with GCP."""
    print("Authenticating with GCP...")
    
    if credentials_path and os.path.exists(credentials_path):
        cmd = ["gcloud", "auth", "activate-service-account", "--key-file", credentials_path]
        returncode, stdout, stderr = run_command(cmd)
        if returncode != 0:
            print(f"Error: {stderr}")
            return False
    
    # Verify authentication
    cmd = ["gcloud", "auth", "list"]
    returncode, stdout, stderr = run_command(cmd)
    if returncode != 0:
        print(f"Error verifying authentication: {stderr}")
        return False
    
    print("Authentication successful")
    return True

def deploy_to_cloudrun(
    service_name: str,
    gcs_path: str,
    project_id: str,
    region: str = REGION,
    env_vars: Optional[Dict[str, str]] = None,
    credentials_path: Optional[str] = None
) -> bool:
    """Deploy source archive to Cloud Run."""
    print(f"Deploying to Cloud Run...")
    print(f"Service: {service_name}")
    print(f"Archive: {gcs_path}")
    print(f"Region: {region}")
    
    # Authenticate
    if not authenticate_gcp(credentials_path):
        return False
    
    # Set project
    cmd = ["gcloud", "config", "set", "project", project_id]
    returncode, stdout, stderr = run_command(cmd)
    if returncode != 0:
        print(f"Error setting project: {stderr}")
        return False
    
    # Build deployment command
    cmd = [
        "gcloud",
        "beta",
        "run",
        "deploy",
        service_name,
        f"--source={gcs_path}",
        f"--region={region}",
        "--no-build",
        f"--base-image={PYTHON_BASE_IMAGE}",
        f"--command={STARTUP_COMMAND}",
    ]
    
    # Add startup arguments
    for arg in STARTUP_ARGS:
        cmd.append(f"--args={arg}")
    
    # Add environment variables
    if env_vars:
        env_var_list = [f"{k}={v}" for k, v in env_vars.items()]
        cmd.append(f"--set-env-vars={','.join(env_var_list)}")
    
    # Add labels
    cmd.extend([
        "--labels=managed-by=github-actions",
        "--allow-unauthenticated",
    ])
    
    print(f"Running: {' '.join(cmd)}")
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    if returncode != 0:
        print(f"Deployment failed: {stderr}")
        return False
    
    print("Deployment command completed")
    return True

def verify_service_running(
    service_name: str,
    region: str = REGION,
    project_id: Optional[str] = None,
    max_wait: int = MAX_WAIT_TIME
) -> bool:
    """Verify service is running and accessible."""
    print(f"Verifying service is running...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Get service details
            cmd = [
                "gcloud",
                "run",
                "services",
                "describe",
                service_name,
                f"--region={region}",
                "--format=value(status.url)",
            ]
            
            if project_id:
                cmd.extend(["--project", project_id])
            
            returncode, stdout, stderr = run_command(cmd)
            
            if returncode == 0 and stdout.strip():
                service_url = stdout.strip()
                print(f"Service URL: {service_url}")
                
                # Try to access service
                try:
                    response = requests.get(service_url, timeout=10)
                    if response.status_code < 500:
                        print(f"Service is running and responding (status: {response.status_code})")
                        return True
                except Exception as e:
                    print(f"Service not yet responding: {e}")
            
        except Exception as e:
            print(f"Error checking service: {e}")
        
        print(f"Waiting for service to be ready... ({int(time.time() - start_time)}s)")
        time.sleep(WAIT_INTERVAL)
    
    print(f"Service verification timed out after {max_wait} seconds")
    return False

def main():
    """Main execution."""
    print("=" * 60)
    print("Cloud Run Deployment (Source without Build)")
    print("=" * 60)
    
    # Get arguments
    if len(sys.argv) < 4:
        print("Usage: python deploy_to_cloudrun.py <service_name> <gcs_path> <project_id> [region] [credentials_path]")
        print("Example: python deploy_to_cloudrun.py ldr-ai gs://ldr-deploy-bucket/ldr-ai-source-20240109-143022.tar.gz my-project")
        return 1
    
    service_name = sys.argv[1]
    gcs_path = sys.argv[2]
    project_id = sys.argv[3]
    region = sys.argv[4] if len(sys.argv) > 4 else REGION
    credentials_path = sys.argv[5] if len(sys.argv) > 5 else None
    
    # Build environment variables
    env_vars = {
        "PORT": PORT,
        "ENVIRONMENT": "production",
    }
    
    # Add from environment if available
    for key in [
        "OPENAI_API_KEY",
        "GOOGLE_DRIVE_API_KEY",
        "GOOGLE_VISION_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "RESEND_API_KEY",
        "APP_URL",
    ]:
        if key in os.environ:
            env_vars[key] = os.environ[key]
    
    # Deploy
    if not deploy_to_cloudrun(
        service_name,
        gcs_path,
        project_id,
        region,
        env_vars,
        credentials_path
    ):
        print("=" * 60)
        print("Deployment failed!")
        print("=" * 60)
        return 1
    
    # Verify service
    if not verify_service_running(service_name, region, project_id):
        print("=" * 60)
        print("Service verification failed!")
        print("=" * 60)
        return 1
    
    print("=" * 60)
    print("Deployment successful!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
