#!/usr/bin/env python3
"""
Measure and compare deployment times between Cloud Build and source without build.

This script:
1. Records deployment time for source without build
2. Compares with historical Cloud Build times
3. Documents time savings
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
METRICS_FILE = "deployment_metrics.json"

def record_deployment_time(deployment_method, duration_seconds, archive_size_mb=None):
    """Record deployment time to metrics file."""
    metrics = {}
    
    # Load existing metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    
    # Add new metric
    timestamp = datetime.now().isoformat()
    metric = {
        "timestamp": timestamp,
        "method": deployment_method,
        "duration_seconds": duration_seconds,
        "archive_size_mb": archive_size_mb,
    }
    
    if "deployments" not in metrics:
        metrics["deployments"] = []
    
    metrics["deployments"].append(metric)
    
    # Save metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Recorded deployment time: {duration_seconds}s ({deployment_method})")
    return metric

def calculate_average_time(deployment_method):
    """Calculate average deployment time for a method."""
    if not os.path.exists(METRICS_FILE):
        return None
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    
    times = [
        d["duration_seconds"]
        for d in metrics.get("deployments", [])
        if d["method"] == deployment_method
    ]
    
    if not times:
        return None
    
    return sum(times) / len(times)

def compare_deployment_methods():
    """Compare deployment times between methods."""
    print("=" * 60)
    print("Deployment Performance Comparison")
    print("=" * 60)
    
    if not os.path.exists(METRICS_FILE):
        print("No metrics available yet")
        return
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    
    # Group by method
    by_method = {}
    for deployment in metrics.get("deployments", []):
        method = deployment["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(deployment)
    
    # Calculate statistics
    print("\nDeployment Time Statistics:")
    print("-" * 60)
    
    for method, deployments in by_method.items():
        times = [d["duration_seconds"] for d in deployments]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n{method}:")
        print(f"  Count: {len(deployments)}")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Min: {min_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
    
    # Calculate time savings
    if "source-without-build" in by_method and "cloud-build" in by_method:
        source_avg = sum(d["duration_seconds"] for d in by_method["source-without-build"]) / len(by_method["source-without-build"])
        build_avg = sum(d["duration_seconds"] for d in by_method["cloud-build"]) / len(by_method["cloud-build"])
        
        savings = build_avg - source_avg
        savings_percent = (savings / build_avg) * 100
        
        print(f"\nTime Savings:")
        print(f"  Cloud Build Average: {build_avg:.2f}s")
        print(f"  Source Without Build Average: {source_avg:.2f}s")
        print(f"  Savings: {savings:.2f}s ({savings_percent:.1f}%)")
    
    print("=" * 60)

def main():
    """Main execution."""
    print("=" * 60)
    print("Deployment Performance Measurement")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("Usage: python measure_deployment_time.py <method> <duration_seconds> [archive_size_mb]")
        print("Example: python measure_deployment_time.py source-without-build 120 45")
        print("\nAvailable commands:")
        print("  record <method> <duration> [size] - Record deployment time")
        print("  compare - Compare deployment methods")
        print("  stats - Show statistics")
        return 1
    
    command = sys.argv[1]
    
    if command == "record":
        if len(sys.argv) < 4:
            print("Usage: python measure_deployment_time.py record <method> <duration_seconds> [archive_size_mb]")
            return 1
        
        method = sys.argv[2]
        duration = float(sys.argv[3])
        archive_size = float(sys.argv[4]) if len(sys.argv) > 4 else None
        
        record_deployment_time(method, duration, archive_size)
        print("=" * 60)
        return 0
    
    elif command == "compare":
        compare_deployment_methods()
        return 0
    
    elif command == "stats":
        compare_deployment_methods()
        return 0
    
    else:
        print(f"Unknown command: {command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
