#!/usr/bin/env python3
"""
Property-based tests for GitHub Actions workflow automation.

Feature: cloud-run-source-deploy
"""

import os
import yaml
from pathlib import Path


class TestWorkflowAutomation:
    """Tests for GitHub Actions workflow automation."""
    
    def test_workflow_automation(self):
        """
        Property 10: Workflow Automation
        
        For any git push to main/dev/prod branches, the GitHub Actions workflow 
        should automatically trigger and execute all deployment steps 
        (archive creation, upload, deployment).
        
        Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Load workflow file
        workflow_path = ".github/workflows/deploy.yml"
        assert os.path.exists(workflow_path), f"Workflow file not found: {workflow_path}"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Verify workflow triggers on push
        assert 'on' in workflow, "Workflow should have 'on' trigger"
        assert 'push' in workflow['on'], "Workflow should trigger on push"
        
        branches = workflow['on']['push']['branches']
        assert 'main' in branches, "Workflow should trigger on main branch"
        assert 'dev' in branches, "Workflow should trigger on dev branch"
        assert 'prod' in branches, "Workflow should trigger on prod branch"
        
        # Verify workflow has deploy job
        assert 'jobs' in workflow, "Workflow should have jobs"
        assert 'deploy' in workflow['jobs'], "Workflow should have deploy job"
        
        deploy_job = workflow['jobs']['deploy']
        assert 'steps' in deploy_job, "Deploy job should have steps"
        
        # Verify required steps exist
        step_names = [step.get('name', '') for step in deploy_job['steps']]
        
        assert any('Checkout' in name for name in step_names), "Should have Checkout step"
        assert any('archive' in name.lower() for name in step_names), "Should have archive creation step"
        assert any('upload' in name.lower() for name in step_names), "Should have upload step"
        assert any('deploy' in name.lower() for name in step_names), "Should have deployment step"
        
        # Verify environment variables are set
        assert 'env' in deploy_job, "Deploy job should have environment variables"
        env = deploy_job['env']
        assert 'SERVICE' in env, "Should have SERVICE env var"
        assert 'REGION' in env, "Should have REGION env var"
        assert 'GCS_BUCKET' in env, "Should have GCS_BUCKET env var"
    
    def test_workflow_has_error_handling(self):
        """
        Verify that workflow has error handling and reporting.
        
        For any deployment failure, the workflow should report the error 
        and stop execution.
        
        Validates: Requirements 4.6
        """
        workflow_path = ".github/workflows/deploy.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        deploy_job = workflow['jobs']['deploy']
        steps = deploy_job['steps']
        
        # Verify there's a failure handling step
        step_names = [step.get('name', '') for step in steps]
        assert any('failed' in name.lower() or 'error' in name.lower() for name in step_names), \
            "Workflow should have error handling step"
    
    def test_workflow_preserves_environment_variables(self):
        """
        Verify that workflow preserves all environment variables.
        
        For any deployment, all required environment variables should be 
        passed to the deployment scripts.
        
        Validates: Requirements 5.1, 5.2
        """
        workflow_path = ".github/workflows/deploy.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        deploy_job = workflow['jobs']['deploy']
        
        # Find the deploy step
        deploy_step = None
        for step in deploy_job['steps']:
            if 'deploy' in step.get('name', '').lower() and 'cloud run' in step.get('name', '').lower():
                deploy_step = step
                break
        
        assert deploy_step is not None, "Should have Cloud Run deployment step"
        assert 'env' in deploy_step, "Deployment step should have environment variables"
        
        env = deploy_step['env']
        required_vars = [
            'OPENAI_API_KEY',
            'SUPABASE_URL',
            'SUPABASE_ANON_KEY',
            'APP_URL',
        ]
        
        for var in required_vars:
            assert var in env, f"Environment variable {var} should be passed to deployment"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
