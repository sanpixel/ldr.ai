"""
GCS bucket setup and initialization for Legal Description Reader
Handles bucket creation, versioning, and permission configuration
"""

import os
import logging
from typing import Dict, Any, Optional
from google.cloud import storage
from google.cloud.exceptions import NotFound, Conflict
from utils.rules import create_default_rules_config
from utils.gcs_rules import GCSRulesManager

logger = logging.getLogger(__name__)


def setup_gcs_bucket(bucket_name: str = "ldr-ai", 
                    project_id: str = None,
                    location: str = "us-central1") -> Dict[str, Any]:
    """
    Set up GCS bucket with proper configuration for rules storage
    
    Args:
        bucket_name: Name of the bucket to create
        project_id: GCP project ID (uses default if None)
        location: GCS bucket location
        
    Returns:
        Dict with setup results and status
    """
    results = {
        'bucket_created': False,
        'versioning_enabled': False,
        'initial_rules_uploaded': False,
        'directory_structure_created': False,
        'errors': []
    }
    
    try:
        # Initialize client using same auth as gcs_storage.py
        from utils.gcs_storage import get_gcs_client
        client = get_gcs_client()
        
        if not client:
            results['errors'].append("Failed to get authenticated GCS client")
            return results
        
        # Check if bucket exists
        bucket = client.bucket(bucket_name)
        bucket_exists = False
        
        try:
            bucket.reload()
            bucket_exists = True
            logger.info(f"Bucket {bucket_name} already exists")
        except NotFound:
            logger.info(f"Bucket {bucket_name} does not exist, creating...")
        
        # Create bucket if it doesn't exist
        if not bucket_exists:
            try:
                bucket = client.create_bucket(bucket_name, location=location)
                results['bucket_created'] = True
                logger.info(f"Created bucket: {bucket_name}")
            except Conflict:
                # Bucket might have been created by another process
                bucket = client.bucket(bucket_name)
                bucket.reload()
                logger.info(f"Bucket {bucket_name} was created by another process")
        
        # Enable versioning
        if not bucket.versioning_enabled:
            bucket.versioning_enabled = True
            bucket.patch()
            results['versioning_enabled'] = True
            logger.info(f"Enabled versioning on bucket: {bucket_name}")
        else:
            results['versioning_enabled'] = True
            logger.info(f"Versioning already enabled on bucket: {bucket_name}")
        
        # Create directory structure
        directory_structure = [
            "rules/current.json",
            "rules/versions/",
            "gold-dataset/descriptions/",
            "gold-dataset/outputs/",
            "test-results/harness-runs/",
            "test-results/failure-analysis/"
        ]
        
        for path in directory_structure:
            if path.endswith('/'):
                # Create directory marker (empty blob)
                marker_blob = bucket.blob(path + '.gitkeep')
                if not marker_blob.exists():
                    marker_blob.upload_from_string('', content_type='text/plain')
            else:
                # Check if file exists
                blob = bucket.blob(path)
                if not blob.exists() and path == "rules/current.json":
                    # Upload initial rules
                    default_rules = create_default_rules_config()
                    blob.upload_from_string(
                        default_rules.to_json(),
                        content_type='application/json'
                    )
                    results['initial_rules_uploaded'] = True
                    logger.info("Uploaded initial rules to current.json")
        
        results['directory_structure_created'] = True
        logger.info("Created directory structure")
        
        # Test access
        test_blob = bucket.blob("test-access.txt")
        test_blob.upload_from_string("test", content_type='text/plain')
        test_blob.delete()
        
        logger.info(f"GCS bucket setup completed successfully: {bucket_name}")
        
    except Exception as e:
        error_msg = f"Failed to setup GCS bucket: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    return results


def verify_gcs_permissions(bucket_name: str = "ldr-ai") -> Dict[str, Any]:
    """
    Verify that the current service account has required permissions
    
    Args:
        bucket_name: Name of the bucket to test
        
    Returns:
        Dict with permission test results
    """
    results = {
        'can_read': False,
        'can_write': False,
        'can_delete': False,
        'can_list': False,
        'errors': []
    }
    
    try:
        from utils.gcs_storage import get_gcs_client
        client = get_gcs_client()
        
        if not client:
            return {
                'success': False,
                'error': 'Failed to get authenticated GCS client',
                'permissions': []
            }
        
        bucket = client.bucket(bucket_name)
        
        # Test list permission
        try:
            list(bucket.list_blobs(max_results=1))
            results['can_list'] = True
        except Exception as e:
            results['errors'].append(f"Cannot list objects: {e}")
        
        # Test write permission
        try:
            test_blob = bucket.blob("permission-test.txt")
            test_blob.upload_from_string("permission test", content_type='text/plain')
            results['can_write'] = True
        except Exception as e:
            results['errors'].append(f"Cannot write objects: {e}")
        
        # Test read permission
        try:
            if results['can_write']:
                content = test_blob.download_as_text()
                if content == "permission test":
                    results['can_read'] = True
        except Exception as e:
            results['errors'].append(f"Cannot read objects: {e}")
        
        # Test delete permission
        try:
            if results['can_write']:
                test_blob.delete()
                results['can_delete'] = True
        except Exception as e:
            results['errors'].append(f"Cannot delete objects: {e}")
        
    except Exception as e:
        results['errors'].append(f"General permission error: {e}")
    
    return results


def initialize_rules_system() -> Dict[str, Any]:
    """
    Initialize the complete rules system with GCS backend
    
    Returns:
        Dict with initialization results
    """
    results = {
        'gcs_setup': {},
        'permissions_verified': {},
        'rules_manager_created': False,
        'initial_upload_successful': False,
        'errors': []
    }
    
    try:
        # Setup GCS bucket
        bucket_name = os.getenv('GCS_RULES_BUCKET', 'ldr-ai')
        results['gcs_setup'] = setup_gcs_bucket(bucket_name)
        
        # Verify permissions
        results['permissions_verified'] = verify_gcs_permissions(bucket_name)
        
        # Create rules manager
        try:
            manager = GCSRulesManager(bucket_name)
            results['rules_manager_created'] = manager.is_available()
        except Exception as e:
            results['errors'].append(f"Failed to create rules manager: {e}")
        
        # Upload initial rules if needed
        if results['gcs_setup'].get('initial_rules_uploaded', False):
            results['initial_upload_successful'] = True
        
        logger.info("Rules system initialization completed")
        
    except Exception as e:
        error_msg = f"Failed to initialize rules system: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    return results


def get_required_permissions() -> List[str]:
    """
    Get list of required GCS permissions for the rules system
    
    Returns:
        List of required IAM permissions
    """
    return [
        "storage.objects.get",      # Read objects
        "storage.objects.create",   # Create objects
        "storage.objects.delete",   # Delete objects
        "storage.objects.list",     # List objects
        "storage.buckets.get"       # Get bucket metadata
    ]


def check_service_account_setup() -> Dict[str, Any]:
    """
    Check if service account is properly configured
    
    Returns:
        Dict with service account status
    """
    results = {
        'credentials_found': False,
        'credentials_type': None,
        'project_id': None,
        'service_account_email': None,
        'errors': []
    }
    
    try:
        # Check for credentials
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            results['credentials_found'] = True
            results['credentials_type'] = 'service_account_file'
        elif os.getenv('GOOGLE_CLOUD_PROJECT'):
            results['credentials_found'] = True
            results['credentials_type'] = 'default_credentials'
        
        # Try to get project info
        try:
            from utils.gcs_storage import get_gcs_client
            client = get_gcs_client()
            if client:
                results['project_id'] = client.project
            
            # Try to get service account info from credentials
            if hasattr(client._credentials, 'service_account_email'):
                results['service_account_email'] = client._credentials.service_account_email
                
        except Exception as e:
            results['errors'].append(f"Cannot access GCP client: {e}")
        
    except Exception as e:
        results['errors'].append(f"Service account check failed: {e}")
    
    return results


if __name__ == "__main__":
    # Run initialization when script is executed directly
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing GCS rules system...")
    results = initialize_rules_system()
    
    print("\nSetup Results:")
    print(f"GCS Setup: {results['gcs_setup']}")
    print(f"Permissions: {results['permissions_verified']}")
    print(f"Rules Manager: {results['rules_manager_created']}")
    
    if results['errors']:
        print(f"\nErrors: {results['errors']}")
        sys.exit(1)
    else:
        print("\nInitialization completed successfully!")
        sys.exit(0)