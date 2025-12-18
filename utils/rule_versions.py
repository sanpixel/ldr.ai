"""
Database operations for rule version tracking
Handles rule version metadata and performance tracking
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from utils.auth import get_supabase_client
from datetime import datetime
import json


def save_rule_version(version_id: str, gcs_path: str, 
                     performance_metrics: Dict[str, Any] = None,
                     description: str = None,
                     parent_version_id: str = None) -> bool:
    """
    Save a new rule version to the database
    
    Args:
        version_id: Unique version identifier
        gcs_path: Path to rules in GCS
        performance_metrics: Optional performance data
        description: Optional description of changes
        parent_version_id: Optional parent version for tracking lineage
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare insert data
        insert_data = {
            'version_id': version_id,
            'gcs_path': gcs_path,
            'performance_metrics': performance_metrics or {},
            'description': description,
            'parent_version_id': parent_version_id,
            'created_by': st.session_state.get('user', {}).get('email', 'system')
        }
        
        # Insert the rule version
        result = supabase.table('rule_versions').insert(insert_data).execute()
        
        if result.data:
            return True
        else:
            st.error("Failed to save rule version")
            return False
            
    except Exception as e:
        st.error(f"Error saving rule version: {str(e)}")
        return False


def set_active_version(version_id: str) -> bool:
    """
    Set a specific version as the active version
    Deactivates all other versions
    
    Args:
        version_id: Version to make active
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # First, deactivate all versions
        supabase.table('rule_versions').update({'is_active': False}).neq('version_id', '').execute()
        
        # Then activate the target version
        result = supabase.table('rule_versions').update({
            'is_active': True,
            'deployed_at': datetime.now().isoformat()
        }).eq('version_id', version_id).execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        st.error(f"Error setting active version: {str(e)}")
        return False


def get_active_version() -> Optional[Dict[str, Any]]:
    """
    Get the currently active rule version
    
    Returns:
        Dict with version data or None if no active version
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('rule_versions')\
            .select('*')\
            .eq('is_active', True)\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception as e:
        st.error(f"Error getting active version: {str(e)}")
        return None


def list_rule_versions(limit: int = 50) -> List[Dict[str, Any]]:
    """
    List all rule versions ordered by creation time
    
    Args:
        limit: Maximum number of versions to return
    
    Returns:
        List of version records
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('rule_versions')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data or []
        
    except Exception as e:
        st.error(f"Error listing rule versions: {str(e)}")
        return []


def get_version_by_id(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific rule version by ID
    
    Args:
        version_id: Version identifier
    
    Returns:
        Dict with version data or None if not found
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('rule_versions')\
            .select('*')\
            .eq('version_id', version_id)\
            .execute()
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception as e:
        st.error(f"Error getting version {version_id}: {str(e)}")
        return None


def update_version_performance(version_id: str, 
                             total_cases: int,
                             passed_cases: int,
                             failed_cases: int,
                             performance_metrics: Dict[str, Any] = None) -> bool:
    """
    Update performance metrics for a rule version
    
    Args:
        version_id: Version to update
        total_cases: Total test cases
        passed_cases: Number of passed cases
        failed_cases: Number of failed cases
        performance_metrics: Additional performance data
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        accuracy = (passed_cases / total_cases * 100) if total_cases > 0 else 0
        
        update_data = {
            'total_test_cases': total_cases,
            'passed_test_cases': passed_cases,
            'failed_test_cases': failed_cases,
            'accuracy_percentage': round(accuracy, 2)
        }
        
        if performance_metrics:
            update_data['performance_metrics'] = performance_metrics
        
        result = supabase.table('rule_versions')\
            .update(update_data)\
            .eq('version_id', version_id)\
            .execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        st.error(f"Error updating version performance: {str(e)}")
        return False


def increment_rollback_count(version_id: str) -> bool:
    """
    Increment the rollback count for a version
    
    Args:
        version_id: Version that was rolled back to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Get current rollback count
        current = get_version_by_id(version_id)
        if not current:
            return False
        
        new_count = (current.get('rollback_count', 0) or 0) + 1
        
        result = supabase.table('rule_versions')\
            .update({'rollback_count': new_count})\
            .eq('version_id', version_id)\
            .execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        st.error(f"Error incrementing rollback count: {str(e)}")
        return False


def save_harness_result(version_id: str,
                       total_cases: int,
                       passed_cases: int,
                       failed_cases: int,
                       failure_details: Dict[str, Any] = None,
                       execution_time_ms: int = None,
                       triggered_by: str = 'automatic') -> bool:
    """
    Save harness test results for a rule version
    
    Args:
        version_id: Rule version that was tested
        total_cases: Total test cases
        passed_cases: Number of passed cases
        failed_cases: Number of failed cases
        failure_details: Details about failures
        execution_time_ms: Test execution time in milliseconds
        triggered_by: What triggered the test run
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        insert_data = {
            'rule_version_id': version_id,
            'total_cases': total_cases,
            'passed_cases': passed_cases,
            'failed_cases': failed_cases,
            'failure_details': failure_details or {},
            'execution_time_ms': execution_time_ms,
            'triggered_by': triggered_by
        }
        
        result = supabase.table('harness_results').insert(insert_data).execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        st.error(f"Error saving harness result: {str(e)}")
        return False


def get_version_harness_results(version_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get harness results for a specific version
    
    Args:
        version_id: Version to get results for
        limit: Maximum number of results
    
    Returns:
        List of harness result records
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('harness_results')\
            .select('*')\
            .eq('rule_version_id', version_id)\
            .order('run_at', desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data or []
        
    except Exception as e:
        st.error(f"Error getting harness results: {str(e)}")
        return []


def get_version_summary_stats() -> Dict[str, Any]:
    """
    Get summary statistics about rule versions
    
    Returns:
        Dict with summary statistics
    """
    try:
        supabase = get_supabase_client()
        
        # Get total versions
        versions_result = supabase.table('rule_versions').select('version_id', count='exact').execute()
        total_versions = versions_result.count or 0
        
        # Get active version
        active_version = get_active_version()
        
        # Get recent harness results
        recent_results = supabase.table('harness_results')\
            .select('accuracy_percentage')\
            .order('run_at', desc=True)\
            .limit(10)\
            .execute()
        
        avg_accuracy = 0
        if recent_results.data:
            accuracies = [r['accuracy_percentage'] for r in recent_results.data if r['accuracy_percentage']]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        return {
            'total_versions': total_versions,
            'active_version_id': active_version['version_id'] if active_version else None,
            'active_version_accuracy': active_version.get('accuracy_percentage') if active_version else None,
            'recent_avg_accuracy': round(avg_accuracy, 2),
            'active_version_created': active_version.get('created_at') if active_version else None
        }
        
    except Exception as e:
        st.error(f"Error getting version summary: {str(e)}")
        return {
            'total_versions': 0,
            'active_version_id': None,
            'active_version_accuracy': None,
            'recent_avg_accuracy': 0,
            'active_version_created': None
        }


def cleanup_old_versions(keep_count: int = 50) -> int:
    """
    Clean up old rule versions, keeping only the most recent ones
    Never deletes the active version
    
    Args:
        keep_count: Number of versions to keep
    
    Returns:
        int: Number of versions deleted
    """
    try:
        supabase = get_supabase_client()
        
        # Get versions to delete (excluding active version)
        all_versions = supabase.table('rule_versions')\
            .select('version_id, created_at, is_active')\
            .order('created_at', desc=True)\
            .execute()
        
        if not all_versions.data or len(all_versions.data) <= keep_count:
            return 0
        
        # Identify versions to delete
        versions_to_delete = []
        for i, version in enumerate(all_versions.data):
            if i >= keep_count and not version.get('is_active', False):
                versions_to_delete.append(version['version_id'])
        
        if not versions_to_delete:
            return 0
        
        # Delete old versions
        for version_id in versions_to_delete:
            # Delete harness results first (foreign key constraint)
            supabase.table('harness_results').delete().eq('rule_version_id', version_id).execute()
            
            # Delete version record
            supabase.table('rule_versions').delete().eq('version_id', version_id).execute()
        
        return len(versions_to_delete)
        
    except Exception as e:
        st.error(f"Error cleaning up old versions: {str(e)}")
        return 0