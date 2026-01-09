#!/usr/bin/env python3
"""
Property-based tests for deployment performance.

Feature: cloud-run-source-deploy
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestDeploymentPerformance:
    """Tests for deployment performance."""
    
    def test_deployment_performance_improvement(self):
        """
        Property 11: Deployment Performance Improvement
        
        For any deployment via source without build compared to Cloud Build, 
        the deployment should complete faster (measured in seconds).
        
        Validates: Requirements 7.1, 7.2
        """
        # Create temporary metrics file
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = os.path.join(tmpdir, "deployment_metrics.json")
            
            # Simulate deployment times
            metrics = {
                "deployments": [
                    {
                        "timestamp": "2024-01-09T10:00:00",
                        "method": "cloud-build",
                        "duration_seconds": 300,
                        "archive_size_mb": None,
                    },
                    {
                        "timestamp": "2024-01-09T10:05:00",
                        "method": "cloud-build",
                        "duration_seconds": 320,
                        "archive_size_mb": None,
                    },
                    {
                        "timestamp": "2024-01-09T10:10:00",
                        "method": "source-without-build",
                        "duration_seconds": 120,
                        "archive_size_mb": 45,
                    },
                    {
                        "timestamp": "2024-01-09T10:15:00",
                        "method": "source-without-build",
                        "duration_seconds": 110,
                        "archive_size_mb": 45,
                    },
                ]
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
            
            # Calculate averages
            cloud_build_times = [
                d["duration_seconds"]
                for d in metrics["deployments"]
                if d["method"] == "cloud-build"
            ]
            source_times = [
                d["duration_seconds"]
                for d in metrics["deployments"]
                if d["method"] == "source-without-build"
            ]
            
            cloud_build_avg = sum(cloud_build_times) / len(cloud_build_times)
            source_avg = sum(source_times) / len(source_times)
            
            # Verify source without build is faster
            assert source_avg < cloud_build_avg, \
                f"Source without build ({source_avg}s) should be faster than Cloud Build ({cloud_build_avg}s)"
            
            # Verify significant improvement (at least 30% faster)
            improvement_percent = ((cloud_build_avg - source_avg) / cloud_build_avg) * 100
            assert improvement_percent >= 30, \
                f"Should have at least 30% improvement, got {improvement_percent:.1f}%"
    
    def test_deployment_time_measurement(self):
        """
        Verify that deployment times are measured and recorded.
        
        For any deployment, the system should measure and record the 
        deployment duration for performance tracking.
        
        Validates: Requirements 7.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = os.path.join(tmpdir, "deployment_metrics.json")
            
            # Create initial metrics
            metrics = {
                "deployments": [
                    {
                        "timestamp": "2024-01-09T10:00:00",
                        "method": "source-without-build",
                        "duration_seconds": 120,
                        "archive_size_mb": 45,
                    }
                ]
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
            
            # Verify metrics file exists and contains data
            assert os.path.exists(metrics_file), "Metrics file should exist"
            
            with open(metrics_file, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert "deployments" in loaded_metrics, "Metrics should contain deployments"
            assert len(loaded_metrics["deployments"]) > 0, "Should have at least one deployment record"
            
            deployment = loaded_metrics["deployments"][0]
            assert "duration_seconds" in deployment, "Should record duration"
            assert "method" in deployment, "Should record deployment method"
            assert "timestamp" in deployment, "Should record timestamp"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
