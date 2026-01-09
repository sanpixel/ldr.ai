#!/usr/bin/env python3
"""
Create a deployable source archive for Cloud Run deploy without build.

This script:
1. Installs all dependencies to a vendor directory
2. Creates a .tar.gz archive with source and dependencies
3. Excludes unnecessary files
4. Verifies archive size is under 250 MiB
"""

import os
import sys
import tarfile
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
VENDOR_DIR = "vendor"
ARCHIVE_DIR = "archives"
MAX_SIZE_MB = 250
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

# Files and directories to exclude
EXCLUDE_PATTERNS = {
    ".git",
    ".gitignore",
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    ".vscode",
    ".cursor",
    ".idea",
    ".env",
    ".env.example",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    ".DS_Store",
    "node_modules",
    ".streamlit/secrets.toml",
    "archives",
    "vendor",
    ".kiro",
    "extension",
    "*.pdf",
    "*.lsp",
    "*.bat",
    "*.gif",
    "*.mp4",
    "*.png",
    "*.json",
    "*.sql",
    "*.jsonl",
    "*.md",
    "Dockerfile",
    "legacy-cloudbuild.yaml",
    "cloudbuild.yaml",
    "dxfimport.lsp",
    "dumbrobot.txt",
    "bearings_prompt.txt",
    "classification_reasoning.json",
    "classification_table.sql",
    "classified_legal_descriptions.jsonl",
    "fixed_classified_legal_descriptions.jsonl",
    "gcs_rules_current.json",
    "ocr_legal_description.jsonl",
    "rules.json",
    "rule_versions_table.sql",
    "test_*.py",
    "export_training_data.py",
    "example_client.py",
    "upload_rules_cli.py",
    "upload_rules_to_gcs.py",
    "webhook_print_server.py",
    "webhook_print_server_README.md",
    "startwebprintserver.bat",
}

def should_exclude(path, root):
    """Check if a path should be excluded from the archive."""
    rel_path = os.path.relpath(path, root)
    
    # Check exact matches
    if rel_path in EXCLUDE_PATTERNS:
        return True
    
    # Check directory names
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return True
    
    # Check patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*."):
            ext = pattern[1:]
            if path.endswith(ext):
                return True
    
    return False

def install_dependencies():
    """Install dependencies to vendor directory."""
    print("Installing dependencies to vendor directory...")
    
    # Clean vendor directory if it exists
    if os.path.exists(VENDOR_DIR):
        shutil.rmtree(VENDOR_DIR)
    
    os.makedirs(VENDOR_DIR, exist_ok=True)
    
    # Install requirements to vendor directory
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
        "-t",
        VENDOR_DIR,
        "--no-cache-dir",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error installing dependencies: {result.stderr}")
        return False
    
    print(f"Dependencies installed to {VENDOR_DIR}")
    return True

def create_archive():
    """Create .tar.gz archive with source and dependencies."""
    print("Creating source archive...")
    
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # Generate archive filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_name = f"ldr-ai-source-{timestamp}.tar.gz"
    archive_path = os.path.join(ARCHIVE_DIR, archive_name)
    
    # Create tar archive
    with tarfile.open(archive_path, "w:gz") as tar:
        root_dir = os.getcwd()
        
        # Add all files except excluded ones
        for root, dirs, files in os.walk(root_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), root_dir)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if should_exclude(file_path, root_dir):
                    continue
                
                # Add file to archive with relative path
                arcname = os.path.relpath(file_path, root_dir)
                tar.add(file_path, arcname=arcname)
    
    return archive_path

def verify_archive(archive_path):
    """Verify archive contents and size."""
    print(f"Verifying archive: {archive_path}")
    
    # Check size
    size_bytes = os.path.getsize(archive_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"Archive size: {size_mb:.2f} MB")
    
    if size_bytes > MAX_SIZE_BYTES:
        print(f"Error: Archive size {size_mb:.2f} MB exceeds limit of {MAX_SIZE_MB} MB")
        return False
    
    # Verify archive can be read
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            print(f"Archive contains {len(members)} files")
            
            # Check for critical files
            critical_files = ["main.py", "requirements.txt", ".streamlit/config.toml"]
            archive_names = {m.name for m in members}
            
            for critical_file in critical_files:
                if not any(critical_file in name for name in archive_names):
                    print(f"Warning: Critical file {critical_file} not found in archive")
            
            # Check for vendor directory
            if not any("vendor" in name for name in archive_names):
                print("Warning: vendor directory not found in archive")
    except Exception as e:
        print(f"Error reading archive: {e}")
        return False
    
    print("Archive verification passed")
    return True

def main():
    """Main execution."""
    print("=" * 60)
    print("Cloud Run Source Archive Creator")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return 1
    
    # Create archive
    archive_path = create_archive()
    if not archive_path:
        print("Failed to create archive")
        return 1
    
    print(f"Archive created: {archive_path}")
    
    # Verify archive
    if not verify_archive(archive_path):
        print("Archive verification failed")
        return 1
    
    print("=" * 60)
    print("Archive creation successful!")
    print(f"Archive: {archive_path}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
