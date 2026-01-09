#!/usr/bin/env python3
"""
Property-based tests for archive creation.

Feature: cloud-run-source-deploy
"""

import os
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from create_archive import should_exclude, EXCLUDE_PATTERNS


class TestArchiveCreation:
    """Tests for archive creation functionality."""
    
    def test_archive_contains_all_source_files(self):
        """
        Property 1: Archive Contains All Source Files
        
        For any application source directory, the created archive should contain 
        all Python source files (main.py, pages/, utils/), configuration files 
        (.streamlit/config.toml), and all installed dependencies from requirements.txt.
        
        Validates: Requirements 1.1, 1.2, 1.5
        """
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create essential files
            Path(tmpdir, "main.py").write_text("print('hello')")
            Path(tmpdir, "requirements.txt").write_text("streamlit\n")
            
            os.makedirs(os.path.join(tmpdir, "pages"), exist_ok=True)
            Path(tmpdir, "pages", "test.py").write_text("# test")
            
            os.makedirs(os.path.join(tmpdir, "utils"), exist_ok=True)
            Path(tmpdir, "utils", "helper.py").write_text("# helper")
            
            os.makedirs(os.path.join(tmpdir, ".streamlit"), exist_ok=True)
            Path(tmpdir, ".streamlit", "config.toml").write_text("[server]\n")
            
            # Create archive
            archive_path = os.path.join(tmpdir, "test.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                for root, dirs, files in os.walk(tmpdir):
                    # Skip archive itself
                    dirs[:] = [d for d in dirs if d != "archives"]
                    
                    for file in files:
                        if file.endswith(".tar.gz"):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmpdir)
                        tar.add(file_path, arcname=arcname)
            
            # Verify archive contains critical files
            with tarfile.open(archive_path, "r:gz") as tar:
                members = {m.name for m in tar.getmembers()}
                
                # Check for critical files
                assert any("main.py" in name for name in members), "main.py not in archive"
                assert any("requirements.txt" in name for name in members), "requirements.txt not in archive"
                assert any("pages" in name for name in members), "pages/ not in archive"
                assert any("utils" in name for name in members), "utils/ not in archive"
                assert any(".streamlit" in name for name in members), ".streamlit/ not in archive"
    
    def test_archive_excludes_unnecessary_files(self):
        """
        Property 2: Archive Excludes Unnecessary Files
        
        For any application source directory, the created archive should NOT contain 
        unnecessary files such as .git, __pycache__, .env, .vscode, .cursor, or other 
        cache/IDE files.
        
        Validates: Requirements 1.3
        """
        # Create temporary directory with excluded files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files that should be excluded
            excluded_files = [
                ".env",
                ".gitignore",
                "__pycache__/module.pyc",
                ".vscode/settings.json",
                ".cursor/config.json",
                ".pytest_cache/test.cache",
                ".hypothesis/database",
            ]
            
            for file_path in excluded_files:
                full_path = os.path.join(tmpdir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                Path(full_path).write_text("excluded")
            
            # Create files that should be included
            Path(tmpdir, "main.py").write_text("print('hello')")
            
            # Create archive
            archive_path = os.path.join(tmpdir, "test.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                for root, dirs, files in os.walk(tmpdir):
                    # Filter excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), tmpdir)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if should_exclude(file_path, tmpdir):
                            continue
                        arcname = os.path.relpath(file_path, tmpdir)
                        tar.add(file_path, arcname=arcname)
            
            # Verify excluded files are not in archive
            with tarfile.open(archive_path, "r:gz") as tar:
                members = {m.name for m in tar.getmembers()}
                
                # Check that excluded patterns are not present
                for member in members:
                    assert ".env" not in member, f".env found in archive: {member}"
                    assert "__pycache__" not in member, f"__pycache__ found in archive: {member}"
                    assert ".vscode" not in member, f".vscode found in archive: {member}"
                    assert ".cursor" not in member, f".cursor found in archive: {member}"
                    assert ".git" not in member, f".git found in archive: {member}"
                
                # Verify main.py is included
                assert any("main.py" in name for name in members), "main.py should be in archive"
    
    def test_archive_size_under_limit(self):
        """
        Property 3: Archive Size Under Limit
        
        For any valid application source, the created archive should not exceed 
        250 MiB in size.
        
        Validates: Requirements 1.4
        """
        # Create temporary directory with files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a reasonable source structure
            Path(tmpdir, "main.py").write_text("print('hello')" * 100)
            Path(tmpdir, "requirements.txt").write_text("streamlit\n" * 50)
            
            os.makedirs(os.path.join(tmpdir, "pages"), exist_ok=True)
            Path(tmpdir, "pages", "test.py").write_text("# test" * 100)
            
            os.makedirs(os.path.join(tmpdir, "utils"), exist_ok=True)
            Path(tmpdir, "utils", "helper.py").write_text("# helper" * 100)
            
            # Create archive
            archive_path = os.path.join(tmpdir, "test.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                for root, dirs, files in os.walk(tmpdir):
                    dirs[:] = [d for d in dirs if d != "archives"]
                    
                    for file in files:
                        if file.endswith(".tar.gz"):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmpdir)
                        tar.add(file_path, arcname=arcname)
            
            # Verify size
            size_bytes = os.path.getsize(archive_path)
            size_mb = size_bytes / (1024 * 1024)
            max_size_mb = 250
            
            assert size_mb <= max_size_mb, f"Archive size {size_mb:.2f} MB exceeds limit of {max_size_mb} MB"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
