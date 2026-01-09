#!/usr/bin/env python3
"""
Rollback Cloud Run deployment to a previous version.

This script:
1. Lists available revisions
2. Rolls back to a previous revision
3. Verifies rollback success
"""

import os
import sys
import subprocess
import time
from typing import List, Optional, Tuple

# Configuration
REGION = "us-central1"
MAX_WAIT_TIME = 300  # 5 minutes
WAIT_INTERVAL = 10  # seconds

def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
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

def list_revisions(service_name: str, region: str = REGION, project_id: Optional[str] = None) -> List[dict]:
    """List available revisions for a service."""
    print(f"Listing revisions for {service_name}...")
    
    cmd = [
        "gcloud",
        "run",
        "revisions",
        "list",
        f"--service={service_name}",
        f"--region={region}",
        "--format=json",
    ]
    
    if project_id:
        cmd.extend(["--project", project_id])
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"Error listing revisions: {stderr}")
        return []
    
    try:
        import json
        revisions = json.loads(stdout)
        return revisions
    except Exception as e:
        print(f"Error parsing revisions: {e}")
        return []

def get_revision_details(service_name: str, revision_name: str, region: str = REGION, project_id: Optional[str] = None) -> dict:
    """Get details for a specific revision."""
    cmd = [
        "gcloud",
        "run",
        "revisions",
        "describe",
        revision_name,
        f"--service={service_name}",
        f"--region={region}",
        "--format=json",
    ]
    
    if project_id:
        cmd.extend(["--project", project_id])
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"Error getting revision details: {stderr}")
        return {}
    
    try:
        import json
        return json.loads(stdout)
    except Exception as e:
        print(f"Error parsing revision details: {e}")
        return {}

def rollback_to_revision(service_name: str, revision_name: str, region: str = REGION, project_id: Optional[str] = None) -> bool:
    """Rollback service to a specific revision."""
    print(f"Rolling back {service_name} to {revision_name}...")
    
    cmd = [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        service_name,
        f"--to-revisions={revision_name}=100",
        f"--region={region}",
    ]
    
    if project_id:
        cmd.extend(["--project", project_id])
    
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    if returncode != 0:
        print(f"Rollback failed: {stderr}")
        return False
    
    print("Rollback command executed")
    return True

def verify_rollback(service_name: str, region: str = REGION, project_id: Optional[str] = None, max_wait: int = MAX_WAIT_TIME) -> bool:
    """Verify that rollback was successful."""
    print(f"Verifying rollback...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
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
                    import requests
                    response = requests.get(service_url, timeout=10)
                    if response.status_code < 500:
                        print(f"Service is running (status: {response.status_code})")
                        return True
                except Exception as e:
                    print(f"Service not yet responding: {e}")
        
        except Exception as e:
            print(f"Error verifying rollback: {e}")
        
        print(f"Waiting for service to be ready... ({int(time.time() - start_time)}s)")
        time.sleep(WAIT_INTERVAL)
    
    print(f"Rollback verification timed out after {max_wait} seconds")
    return False

def main():
    """Main execution."""
    print("=" * 60)
    print("Cloud Run Rollback Tool")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python rollback_deployment.py <service_name> [revision_name] [project_id]")
        print("Example: python rollback_deployment.py ldr-ai ldr-ai-00001-abc my-project")
        print("\nIf revision_name is not provided, will list available revisions")
        return 1
    
    service_name = sys.argv[1]
    revision_name = sys.argv[2] if len(sys.argv) > 2 else None
    project_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    # List revisions
    revisions = list_revisions(service_name, REGION, project_id)
    
    if not revisions:
        print("No revisions found")
        return 1
    
    print(f"\nAvailable revisions:")
    print("-" * 60)
    for i, rev in enumerate(revisions[:10]):  # Show last 10
        name = rev.get("metadata", {}).get("name", "unknown")
        created = rev.get("metadata", {}).get("creationTimestamp", "unknown")
        status = rev.get("status", {}).get("conditions", [{}])[0].get("status", "unknown")
        print(f"{i+1}. {name}")
        print(f"   Created: {created}")
        print(f"   Status: {status}")
    
    # If no revision specified, ask user
    if not revision_name:
        print("\nNo revision specified. Use the revision name from the list above.")
        print("Example: python rollback_deployment.py ldr-ai ldr-ai-00001-abc")
        return 1
    
    # Verify revision exists
    revision_details = get_revision_details(service_name, revision_name, REGION, project_id)
    if not revision_details:
        print(f"Revision not found: {revision_name}")
        return 1
    
    print(f"\nRolling back to: {revision_name}")
    
    # Perform rollback
    if not rollback_to_revision(service_name, revision_name, REGION, project_id):
        print("=" * 60)
        print("Rollback failed!")
        print("=" * 60)
        return 1
    
    # Verify rollback
    if not verify_rollback(service_name, REGION, project_id):
        print("=" * 60)
        print("Rollback verification failed!")
        print("=" * 60)
        return 1
    
    print("=" * 60)
    print("Rollback successful!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
