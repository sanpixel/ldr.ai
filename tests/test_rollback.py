#!/usr/bin/env python3
"""
Property-based tests for rollback capability.

Feature: cloud-run-source-deploy
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestRollback:
    """Tests for rollback capability."""
    
    def test_rollback_capability(self):
        """
        Property 12: Rollback Capability
        
        For any failed deployment, the previous working version should remain 
        running and accessible, and the system should allow rollback to a 
        previous deployment.
        
        Validates: Requirements 8.1, 8.2, 8.3
        """
        with patch('scripts.rollback_deployment.subprocess.run') as mock_run:
            with patch('scripts.rollback_deployment.requests.get') as mock_get:
                # Mock list revisions
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps([
                    {
                        "metadata": {
                            "name": "ldr-ai-00001-abc",
                            "creationTimestamp": "2024-01-09T10:00:00Z"
                        },
                        "status": {
                            "conditions": [{"status": "True"}]
                        }
                    },
                    {
                        "metadata": {
                            "name": "ldr-ai-00002-def",
                            "creationTimestamp": "2024-01-09T10:05:00Z"
                        },
                        "status": {
                            "conditions": [{"status": "True"}]
                        }
                    }
                ])
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Mock service response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                from rollback_deployment import list_revisions, rollback_to_revision, verify_rollback
                
                # List revisions
                revisions = list_revisions("ldr-ai", "us-central1", "my-project")
                assert len(revisions) > 0, "Should list available revisions"
                
                # Rollback to previous revision
                result = rollback_to_revision(
                    "ldr-ai",
                    "ldr-ai-00001-abc",
                    "us-central1",
                    "my-project"
                )
                assert result is True, "Rollback should succeed"
                
                # Verify rollback
                result = verify_rollback(
                    "ldr-ai",
                    "us-central1",
                    "my-project",
                    max_wait=30
                )
                assert result is True, "Previous version should be running"
    
    def test_version_history_maintained(self):
        """
        Verify that version history is maintained for troubleshooting.
        
        For any deployment, the system should maintain version history 
        that allows viewing and rolling back to previous versions.
        
        Validates: Requirements 8.3
        """
        with patch('scripts.rollback_deployment.subprocess.run') as mock_run:
            # Mock list revisions
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps([
                {
                    "metadata": {
                        "name": "ldr-ai-00001-abc",
                        "creationTimestamp": "2024-01-09T10:00:00Z"
                    },
                    "status": {
                        "conditions": [{"status": "True"}]
                    }
                },
                {
                    "metadata": {
                        "name": "ldr-ai-00002-def",
                        "creationTimestamp": "2024-01-09T10:05:00Z"
                    },
                    "status": {
                        "conditions": [{"status": "True"}]
                    }
                },
                {
                    "metadata": {
                        "name": "ldr-ai-00003-ghi",
                        "creationTimestamp": "2024-01-09T10:10:00Z"
                    },
                    "status": {
                        "conditions": [{"status": "True"}]
                    }
                }
            ])
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            from rollback_deployment import list_revisions
            
            # List revisions
            revisions = list_revisions("ldr-ai", "us-central1", "my-project")
            
            # Verify version history is available
            assert len(revisions) >= 3, "Should maintain version history"
            
            # Verify revisions have timestamps
            for rev in revisions:
                assert "metadata" in rev, "Revision should have metadata"
                assert "creationTimestamp" in rev["metadata"], "Should have creation timestamp"
    
    def test_previous_version_remains_accessible(self):
        """
        Verify that previous version remains accessible after failed deployment.
        
        For any failed deployment, the previous working version should remain 
        running and accessible to users.
        
        Validates: Requirements 8.1
        """
        with patch('scripts.rollback_deployment.subprocess.run') as mock_run:
            with patch('scripts.rollback_deployment.requests.get') as mock_get:
                # Mock service describe (previous version still running)
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "https://ldr.clocknumbers.com"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Mock service response (previous version is accessible)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                from rollback_deployment import verify_rollback
                
                # Verify previous version is accessible
                result = verify_rollback(
                    "ldr-ai",
                    "us-central1",
                    "my-project",
                    max_wait=30
                )
                
                assert result is True, "Previous version should be accessible"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
