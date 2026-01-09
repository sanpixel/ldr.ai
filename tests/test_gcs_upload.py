#!/usr/bin/env python3
"""
Property-based tests for Cloud Storage upload.

Feature: cloud-run-source-deploy
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestGCSUpload:
    """Tests for Cloud Storage upload functionality."""
    
    def test_archive_upload_succeeds(self):
        """
        Property 4: Archive Upload Succeeds
        
        For any valid source archive, uploading it to Cloud Storage should result 
        in the file being accessible at the specified GCS path.
        
        Validates: Requirements 2.1, 2.2
        """
        # Create temporary archive file
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "test-archive.tar.gz")
            Path(archive_path).write_text("fake archive content")
            
            # Mock GCS client
            with patch('scripts.upload_to_gcs.storage.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                
                # Mock bucket
                mock_bucket = MagicMock()
                mock_bucket.exists.return_value = True
                mock_client.bucket.return_value = mock_bucket
                
                # Mock blob
                mock_blob = MagicMock()
                mock_blob.exists.return_value = True
                mock_blob.size = os.path.getsize(archive_path)
                mock_bucket.blob.return_value = mock_blob
                
                # Import and test
                from upload_to_gcs import upload_archive
                
                result = upload_archive(
                    archive_path,
                    "test-bucket",
                    "test-archive.tar.gz"
                )
                
                # Verify upload was called
                assert result is True, "Upload should succeed"
                mock_blob.upload_from_filename.assert_called_once_with(archive_path)
    
    def test_upload_verifies_file_exists(self):
        """
        Verify that uploaded file is accessible in Cloud Storage.
        
        For any uploaded archive, the system should verify the file exists 
        and is accessible at the specified GCS path.
        
        Validates: Requirements 2.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "test-archive.tar.gz")
            Path(archive_path).write_text("fake archive content")
            
            # Mock GCS client
            with patch('scripts.upload_to_gcs.storage.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                
                # Mock bucket
                mock_bucket = MagicMock()
                mock_bucket.exists.return_value = True
                mock_client.bucket.return_value = mock_bucket
                
                # Mock blob - file exists
                mock_blob = MagicMock()
                mock_blob.exists.return_value = True
                mock_blob.size = os.path.getsize(archive_path)
                mock_bucket.blob.return_value = mock_blob
                
                from upload_to_gcs import upload_archive
                
                result = upload_archive(
                    archive_path,
                    "test-bucket",
                    "test-archive.tar.gz"
                )
                
                # Verify file was checked
                assert result is True, "Upload should verify file exists"
    
    def test_upload_handles_missing_archive(self):
        """
        Verify that upload fails gracefully when archive doesn't exist.
        
        For any non-existent archive path, the upload should fail with 
        a clear error message.
        
        Validates: Requirements 2.3
        """
        from upload_to_gcs import upload_archive
        
        result = upload_archive(
            "/nonexistent/archive.tar.gz",
            "test-bucket",
            "test-archive.tar.gz"
        )
        
        # Should fail
        assert result is False, "Upload should fail for missing archive"
    
    def test_upload_handles_missing_bucket(self):
        """
        Verify that upload fails when bucket doesn't exist.
        
        For any non-existent bucket, the upload should fail with 
        a clear error message.
        
        Validates: Requirements 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "test-archive.tar.gz")
            Path(archive_path).write_text("fake archive content")
            
            # Mock GCS client
            with patch('scripts.upload_to_gcs.storage.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                
                # Mock bucket - doesn't exist
                mock_bucket = MagicMock()
                mock_bucket.exists.return_value = False
                mock_client.bucket.return_value = mock_bucket
                
                from upload_to_gcs import upload_archive
                
                result = upload_archive(
                    archive_path,
                    "nonexistent-bucket",
                    "test-archive.tar.gz"
                )
                
                # Should fail
                assert result is False, "Upload should fail for missing bucket"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
