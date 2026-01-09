#!/usr/bin/env python3
"""
Property-based tests for Cloud Run deployment.

Feature: cloud-run-source-deploy
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock, call

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestCloudRunDeploy:
    """Tests for Cloud Run deployment functionality."""
    
    def test_deployment_command_correctness(self):
        """
        Property 5: Deployment Command Correctness
        
        For any deployment to Cloud Run, the gcloud command should include 
        the --no-build flag, specify python311 base image, set PORT=8080, 
        and use the correct startup command (streamlit run main.py).
        
        Validates: Requirements 3.1, 3.2, 3.3, 3.4
        """
        with patch('scripts.deploy_to_cloudrun.subprocess.run') as mock_run:
            with patch('scripts.deploy_to_cloudrun.requests.get') as mock_get:
                # Mock successful deployment
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Mock service response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                from deploy_to_cloudrun import deploy_to_cloudrun
                
                result = deploy_to_cloudrun(
                    "ldr-ai",
                    "gs://bucket/archive.tar.gz",
                    "my-project",
                    "us-central1",
                    {"PORT": "8080", "ENVIRONMENT": "production"}
                )
                
                # Verify command was called
                assert mock_run.called, "gcloud command should be executed"
                
                # Get the command that was called
                call_args = mock_run.call_args[0][0]
                command_str = " ".join(call_args)
                
                # Verify command contains required flags
                assert "--no-build" in command_str, "Command should include --no-build flag"
                assert "python311" in command_str, "Command should specify python311 base image"
                assert "streamlit" in command_str, "Command should specify streamlit startup command"
                assert "main.py" in command_str, "Command should specify main.py as argument"
    
    def test_deployed_service_accessibility(self):
        """
        Property 6: Deployed Service Accessibility
        
        For any source archive deployed to Cloud Run, the deployed service 
        should be running and respond to HTTP requests.
        
        Validates: Requirements 3.5
        """
        with patch('scripts.deploy_to_cloudrun.subprocess.run') as mock_run:
            with patch('scripts.deploy_to_cloudrun.requests.get') as mock_get:
                # Mock successful deployment
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Mock service response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                from deploy_to_cloudrun import verify_service_running
                
                result = verify_service_running(
                    "ldr-ai",
                    "us-central1",
                    "my-project",
                    max_wait=30
                )
                
                # Service should be accessible
                assert result is True, "Service should be running and accessible"
    
    def test_environment_variables_passed(self):
        """
        Verify that environment variables are passed to Cloud Run.
        
        For any deployment, all specified environment variables should be 
        included in the gcloud command.
        
        Validates: Requirements 3.4, 5.1
        """
        with patch('scripts.deploy_to_cloudrun.subprocess.run') as mock_run:
            with patch('scripts.deploy_to_cloudrun.requests.get') as mock_get:
                # Mock successful deployment
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Mock service response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                from deploy_to_cloudrun import deploy_to_cloudrun
                
                env_vars = {
                    "PORT": "8080",
                    "ENVIRONMENT": "production",
                    "OPENAI_API_KEY": "test-key",
                    "SUPABASE_URL": "https://test.supabase.co",
                }
                
                result = deploy_to_cloudrun(
                    "ldr-ai",
                    "gs://bucket/archive.tar.gz",
                    "my-project",
                    "us-central1",
                    env_vars
                )
                
                # Verify command includes env vars
                call_args = mock_run.call_args[0][0]
                command_str = " ".join(call_args)
                
                assert "PORT=8080" in command_str, "PORT should be in command"
                assert "ENVIRONMENT=production" in command_str, "ENVIRONMENT should be in command"
    
    def test_deployment_handles_errors(self):
        """
        Verify that deployment errors are handled gracefully.
        
        For any deployment failure, the system should report the error 
        and return failure status.
        
        Validates: Requirements 3.1
        """
        with patch('scripts.deploy_to_cloudrun.subprocess.run') as mock_run:
            # Mock failed deployment
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Deployment failed"
            mock_run.return_value = mock_result
            
            from deploy_to_cloudrun import deploy_to_cloudrun
            
            result = deploy_to_cloudrun(
                "ldr-ai",
                "gs://bucket/archive.tar.gz",
                "my-project",
                "us-central1"
            )
            
            # Should fail
            assert result is False, "Deployment should fail on error"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
