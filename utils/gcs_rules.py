"""
Google Cloud Storage rules management for Legal Description Reader
Handles versioned storage and retrieval of regex rule configurations
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from utils.rules import RulesConfig
import streamlit as st

# Handle optional Google Cloud Storage dependency
try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, GoogleCloudError
    GCS_AVAILABLE = True
except ImportError:
    # Mock classes for when GCS is not available
    class storage:
        class Client:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Google Cloud Storage not available")
        class Bucket:
            pass
    
    class NotFound(Exception):
        pass
    
    class GoogleCloudError(Exception):
        pass
    
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class GCSRulesManager:
    """
    Manages versioned regex rules in Google Cloud Storage
    Provides versioning, rollback, and atomic deployment capabilities
    """
    
    def __init__(self, bucket_name: str = "ldr-rules-bucket", project_id: str = None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.client = None
        self.bucket = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize GCS client and bucket"""
        try:
            if not GCS_AVAILABLE:
                logger.warning("Google Cloud Storage library not available")
                return
            
            if self.project_id:
                self.client = storage.Client(project=self.project_id)
            else:
                self.client = storage.Client()
            
            # Get or create bucket
            self.bucket = self._get_or_create_bucket()
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            if st.session_state.get('debug_enabled', False):
                st.error(f"GCS initialization failed: {e}")
    
    def _get_or_create_bucket(self) -> storage.Bucket:
        """Get existing bucket or create new one with versioning enabled"""
        try:
            # Try to get existing bucket
            bucket = self.client.bucket(self.bucket_name)
            
            # Check if bucket exists by trying to get its metadata
            try:
                bucket.reload()
                logger.info(f"Using existing bucket: {self.bucket_name}")
            except NotFound:
                # Bucket doesn't exist, create it
                logger.info(f"Creating new bucket: {self.bucket_name}")
                bucket = self.client.create_bucket(self.bucket_name)
            
            # Enable versioning if not already enabled
            if not bucket.versioning_enabled:
                bucket.versioning_enabled = True
                bucket.patch()
                logger.info(f"Enabled versioning on bucket: {self.bucket_name}")
            
            return bucket
            
        except Exception as e:
            logger.error(f"Failed to get/create bucket {self.bucket_name}: {e}")
            raise
    
    def _generate_version_id(self) -> str:
        """Generate next version ID in format: rules_v000001.json"""
        try:
            # List existing versions to find the highest number
            prefix = "rules/versions/rules_v"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            max_version = 0
            for blob in blobs:
                # Extract version number from filename
                filename = blob.name.split('/')[-1]  # Get just the filename
                if filename.startswith('rules_v') and filename.endswith('.json'):
                    try:
                        version_str = filename[7:13]  # Extract 6-digit number
                        version_num = int(version_str)
                        max_version = max(max_version, version_num)
                    except (ValueError, IndexError):
                        continue
            
            # Return next version
            next_version = max_version + 1
            return f"rules_v{next_version:06d}.json"
            
        except Exception as e:
            logger.error(f"Failed to generate version ID: {e}")
            # Fallback to timestamp-based version
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"rules_v{timestamp}.json"
    
    def save_rules_version(self, rules: RulesConfig) -> str:
        """
        Save rules as a new version and return version ID
        
        Args:
            rules: RulesConfig to save
            
        Returns:
            str: Version ID of saved rules
        """
        try:
            if not self.bucket:
                raise RuntimeError("GCS bucket not initialized")
            
            # Generate version ID
            version_filename = self._generate_version_id()
            version_path = f"rules/versions/{version_filename}"
            
            # Add version metadata
            rules_dict = rules.to_dict()
            rules_dict['metadata'] = rules_dict.get('metadata', {})
            rules_dict['metadata']['version_id'] = version_filename
            rules_dict['metadata']['created_at'] = datetime.now().isoformat()
            rules_dict['metadata']['gcs_path'] = version_path
            
            # Upload to versions directory
            version_blob = self.bucket.blob(version_path)
            version_blob.upload_from_string(
                json.dumps(rules_dict, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"Saved rules version: {version_path}")
            
            # Copy to current.json (atomic deployment)
            current_path = "rules/current.json"
            current_blob = self.bucket.blob(current_path)
            current_blob.upload_from_string(
                json.dumps(rules_dict, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"Deployed rules to current: {current_path}")
            
            return version_filename
            
        except Exception as e:
            logger.error(f"Failed to save rules version: {e}")
            raise
    
    def load_current_rules(self) -> Optional[RulesConfig]:
        """
        Load current active rules from GCS
        
        Returns:
            RulesConfig or None if not found
        """
        try:
            if not self.bucket:
                logger.warning("GCS bucket not initialized, cannot load rules")
                return None
            
            current_blob = self.bucket.blob("rules/current.json")
            
            if not current_blob.exists():
                logger.warning("No current rules found in GCS")
                return None
            
            rules_json = current_blob.download_as_text()
            rules_config = RulesConfig.from_json(rules_json)
            
            logger.info("Loaded current rules from GCS")
            return rules_config
            
        except Exception as e:
            logger.error(f"Failed to load current rules: {e}")
            return None
    
    def load_rules_version(self, version_id: str) -> Optional[RulesConfig]:
        """
        Load specific version of rules
        
        Args:
            version_id: Version ID (e.g., "rules_v000001.json")
            
        Returns:
            RulesConfig or None if not found
        """
        try:
            if not self.bucket:
                return None
            
            version_path = f"rules/versions/{version_id}"
            version_blob = self.bucket.blob(version_path)
            
            if not version_blob.exists():
                logger.warning(f"Rules version not found: {version_id}")
                return None
            
            rules_json = version_blob.download_as_text()
            rules_config = RulesConfig.from_json(rules_json)
            
            logger.info(f"Loaded rules version: {version_id}")
            return rules_config
            
        except Exception as e:
            logger.error(f"Failed to load rules version {version_id}: {e}")
            return None
    
    def list_versions(self) -> List[str]:
        """
        List all available rule versions
        
        Returns:
            List of version IDs sorted by creation time (newest first)
        """
        try:
            if not self.bucket:
                return []
            
            prefix = "rules/versions/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            versions = []
            for blob in blobs:
                filename = blob.name.split('/')[-1]
                if filename.endswith('.json'):
                    versions.append(filename)
            
            # Sort by version number (newest first)
            versions.sort(reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback current rules to a specific version
        
        Args:
            version_id: Version ID to rollback to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the target version
            target_rules = self.load_rules_version(version_id)
            if not target_rules:
                logger.error(f"Cannot rollback: version {version_id} not found")
                return False
            
            # Update metadata to indicate this is a rollback
            target_dict = target_rules.to_dict()
            target_dict['metadata'] = target_dict.get('metadata', {})
            target_dict['metadata']['rollback_from'] = version_id
            target_dict['metadata']['rollback_at'] = datetime.now().isoformat()
            
            # Deploy as current
            current_blob = self.bucket.blob("rules/current.json")
            current_blob.upload_from_string(
                json.dumps(target_dict, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"Rolled back to version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def get_version_metadata(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific version
        
        Args:
            version_id: Version ID
            
        Returns:
            Dict with version metadata or None if not found
        """
        try:
            rules = self.load_rules_version(version_id)
            if rules and rules.metadata:
                return rules.metadata
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metadata for version {version_id}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if GCS is available and accessible"""
        try:
            if not GCS_AVAILABLE:
                return False
            
            if not self.bucket:
                return False
            
            # Try a simple operation to test connectivity
            list(self.bucket.list_blobs(max_results=1))
            return True
            
        except Exception:
            return False
    
    def get_bucket_info(self) -> Dict[str, Any]:
        """Get information about the GCS bucket"""
        try:
            if not self.bucket:
                return {'error': 'Bucket not initialized'}
            
            self.bucket.reload()
            return {
                'name': self.bucket.name,
                'location': self.bucket.location,
                'versioning_enabled': self.bucket.versioning_enabled,
                'created': self.bucket.time_created.isoformat() if self.bucket.time_created else None,
                'storage_class': self.bucket.storage_class
            }
            
        except Exception as e:
            return {'error': str(e)}


def create_gcs_rules_manager() -> GCSRulesManager:
    """
    Factory function to create GCS rules manager with appropriate configuration
    Uses environment variables for configuration
    """
    bucket_name = os.getenv('GCS_RULES_BUCKET', 'ldr-rules-bucket')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    return GCSRulesManager(bucket_name=bucket_name, project_id=project_id)


def upload_initial_rules_to_gcs(rules_config: RulesConfig) -> bool:
    """
    Upload initial rules configuration to GCS
    Used for bootstrapping the system
    """
    try:
        manager = create_gcs_rules_manager()
        version_id = manager.save_rules_version(rules_config)
        
        logger.info(f"Uploaded initial rules as version: {version_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload initial rules: {e}")
        return False