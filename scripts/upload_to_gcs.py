#!/usr/bin/env python3
"""
Upload source archive to Google Cloud Storage.

This script:
1. Authenticates with GCP
2. Uploads archive to designated bucket
3. Verifies upload completion
4. Handles retries on failure
"""

import os
import sys
import time
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def get_gcs_client(credentials_path=None):
    """Get authenticated GCS client."""
    try:
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            return storage.Client(credentials=credentials)
        else:
            # Use default credentials (from environment)
            return storage.Client()
    except Exception as e:
        print(f"Error authenticating with GCP: {e}")
        return None

def upload_archive(archive_path, bucket_name, gcs_path=None, credentials_path=None):
    """Upload archive to GCS with retry logic."""
    print(f"Uploading archive to GCS...")
    print(f"Archive: {archive_path}")
    print(f"Bucket: {bucket_name}")
    
    # Verify archive exists
    if not os.path.exists(archive_path):
        print(f"Error: Archive file not found: {archive_path}")
        return False
    
    # Get GCS client
    client = get_gcs_client(credentials_path)
    if not client:
        print("Failed to authenticate with GCP")
        return False
    
    # Get bucket
    try:
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            print(f"Error: Bucket does not exist: {bucket_name}")
            return False
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return False
    
    # Determine GCS path
    if not gcs_path:
        gcs_path = os.path.basename(archive_path)
    
    # Upload with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Upload attempt {attempt + 1}/{MAX_RETRIES}...")
            
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(archive_path)
            
            print(f"Upload successful: gs://{bucket_name}/{gcs_path}")
            
            # Verify upload
            if verify_upload(bucket, gcs_path, archive_path):
                return True
            else:
                print("Upload verification failed")
                return False
                
        except Exception as e:
            print(f"Upload failed (attempt {attempt + 1}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries exceeded")
                return False
    
    return False

def verify_upload(bucket, gcs_path, local_path):
    """Verify uploaded file matches local file."""
    try:
        blob = bucket.blob(gcs_path)
        
        # Check if blob exists
        if not blob.exists():
            print(f"Error: Uploaded file not found in GCS: {gcs_path}")
            return False
        
        # Compare sizes
        local_size = os.path.getsize(local_path)
        remote_size = blob.size
        
        if local_size != remote_size:
            print(f"Error: Size mismatch. Local: {local_size}, Remote: {remote_size}")
            return False
        
        print(f"Upload verified: {remote_size} bytes")
        return True
        
    except Exception as e:
        print(f"Error verifying upload: {e}")
        return False

def main():
    """Main execution."""
    print("=" * 60)
    print("Cloud Storage Archive Uploader")
    print("=" * 60)
    
    # Get arguments
    if len(sys.argv) < 3:
        print("Usage: python upload_to_gcs.py <archive_path> <bucket_name> [gcs_path] [credentials_path]")
        print("Example: python upload_to_gcs.py archives/ldr-ai-source-20240109-143022.tar.gz ldr-deploy-bucket")
        return 1
    
    archive_path = sys.argv[1]
    bucket_name = sys.argv[2]
    gcs_path = sys.argv[3] if len(sys.argv) > 3 else None
    credentials_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Upload archive
    if upload_archive(archive_path, bucket_name, gcs_path, credentials_path):
        print("=" * 60)
        print("Upload successful!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("Upload failed!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
