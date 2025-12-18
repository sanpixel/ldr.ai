"""
Cached rule loader for Legal Description Reader
Provides resilient, performant access to regex rules with fallback mechanisms
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from utils.rules import RulesConfig, RulesProcessor, create_default_rules_config
from utils.gcs_rules import GCSRulesManager
import streamlit as st

logger = logging.getLogger(__name__)


class CachedRulesLoader:
    """
    Manages cached loading of regex rules with automatic refresh and fallback
    
    Features:
    - 300-second refresh cycle from GCS
    - Regex compilation validation
    - Fallback to last-known-good rules
    - Graceful degradation when GCS unavailable
    - Thread-safe operations
    """
    
    def __init__(self, refresh_interval_seconds: int = 300):
        self.refresh_interval = refresh_interval_seconds
        self.gcs_manager = None
        self.current_rules = None
        self.last_good_rules = None
        self.last_refresh_time = None
        self.last_refresh_success = False
        self.refresh_lock = threading.Lock()
        self.initialization_attempted = False
        
        # Error tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.last_error = None
        
        # Performance metrics
        self.load_count = 0
        self.cache_hit_count = 0
        self.refresh_count = 0
        self.fallback_count = 0
        
        # Initialize with default rules as ultimate fallback
        self.default_rules = create_default_rules_config()
        
        # Start with default rules
        self.current_rules = self.default_rules
        self.last_good_rules = self.default_rules
    
    def _initialize_gcs_manager(self) -> bool:
        """
        Initialize GCS manager with error handling
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.initialization_attempted:
            return self.gcs_manager is not None
        
        self.initialization_attempted = True
        
        try:
            self.gcs_manager = GCSRulesManager()
            
            # Test connectivity
            if self.gcs_manager.is_available():
                logger.info("GCS rules manager initialized successfully")
                return True
            else:
                logger.warning("GCS rules manager not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize GCS manager: {e}")
            self.last_error = str(e)
            return False
    
    def _should_refresh(self) -> bool:
        """
        Check if rules should be refreshed based on time interval
        
        Returns:
            bool: True if refresh is needed
        """
        if self.last_refresh_time is None:
            return True
        
        time_since_refresh = datetime.now() - self.last_refresh_time
        return time_since_refresh.total_seconds() >= self.refresh_interval
    
    def _validate_rules(self, rules: RulesConfig) -> bool:
        """
        Validate that rules can be compiled successfully
        
        Args:
            rules: RulesConfig to validate
            
        Returns:
            bool: True if all rules compile successfully
        """
        try:
            validation_errors = rules.validate_all_rules()
            if validation_errors:
                logger.error(f"Rules validation failed: {validation_errors}")
                return False
            
            # Try to create a processor (this will compile all rules)
            processor = RulesProcessor(rules)
            return True
            
        except Exception as e:
            logger.error(f"Rules validation exception: {e}")
            return False
    
    def _refresh_from_gcs(self) -> bool:
        """
        Attempt to refresh rules from GCS
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._initialize_gcs_manager():
                return False
            
            # Load current rules from GCS
            new_rules = self.gcs_manager.load_current_rules()
            
            if new_rules is None:
                logger.warning("No current rules found in GCS")
                return False
            
            # Validate new rules
            if not self._validate_rules(new_rules):
                logger.error("New rules failed validation, keeping current rules")
                return False
            
            # Update current rules
            self.current_rules = new_rules
            self.last_good_rules = new_rules
            self.consecutive_failures = 0
            self.last_error = None
            
            logger.info(f"Successfully refreshed rules from GCS (version: {new_rules.version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh rules from GCS: {e}")
            self.last_error = str(e)
            self.consecutive_failures += 1
            return False
    
    def _handle_refresh_failure(self):
        """Handle refresh failure with appropriate fallback strategy"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"Too many consecutive failures ({self.consecutive_failures}), "
                         f"falling back to last known good rules")
            
            if self.last_good_rules:
                self.current_rules = self.last_good_rules
                self.fallback_count += 1
            else:
                logger.warning("No last good rules available, using default rules")
                self.current_rules = self.default_rules
                self.fallback_count += 1
    
    def get_rules(self, force_refresh: bool = False) -> RulesConfig:
        """
        Get current rules with automatic refresh logic
        
        Args:
            force_refresh: Force refresh even if not due
            
        Returns:
            RulesConfig: Current rules (never None)
        """
        self.load_count += 1
        
        # Check if refresh is needed
        needs_refresh = force_refresh or self._should_refresh()
        
        if needs_refresh:
            with self.refresh_lock:
                # Double-check after acquiring lock
                if force_refresh or self._should_refresh():
                    self.refresh_count += 1
                    self.last_refresh_time = datetime.now()
                    
                    success = self._refresh_from_gcs()
                    self.last_refresh_success = success
                    
                    if not success:
                        self._handle_refresh_failure()
                else:
                    # Another thread already refreshed
                    self.cache_hit_count += 1
        else:
            self.cache_hit_count += 1
        
        # Always return valid rules
        return self.current_rules or self.default_rules
    
    def get_processor(self, force_refresh: bool = False) -> RulesProcessor:
        """
        Get a RulesProcessor with current rules
        
        Args:
            force_refresh: Force refresh rules before creating processor
            
        Returns:
            RulesProcessor: Processor with current rules
        """
        rules = self.get_rules(force_refresh)
        return RulesProcessor(rules)
    
    def force_refresh(self) -> bool:
        """
        Force an immediate refresh of rules
        
        Returns:
            bool: True if refresh was successful
        """
        return self.get_rules(force_refresh=True) is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the cached loader
        
        Returns:
            Dict with status information
        """
        return {
            'current_version': self.current_rules.version if self.current_rules else None,
            'last_refresh_time': self.last_refresh_time.isoformat() if self.last_refresh_time else None,
            'last_refresh_success': self.last_refresh_success,
            'consecutive_failures': self.consecutive_failures,
            'last_error': self.last_error,
            'gcs_available': self.gcs_manager.is_available() if self.gcs_manager else False,
            'refresh_interval_seconds': self.refresh_interval,
            'performance_metrics': {
                'load_count': self.load_count,
                'cache_hit_count': self.cache_hit_count,
                'refresh_count': self.refresh_count,
                'fallback_count': self.fallback_count,
                'cache_hit_rate': (self.cache_hit_count / self.load_count * 100) if self.load_count > 0 else 0
            }
        }
    
    def is_healthy(self) -> bool:
        """
        Check if the loader is in a healthy state
        
        Returns:
            bool: True if healthy, False if degraded
        """
        # Consider healthy if:
        # 1. We have current rules
        # 2. Not too many consecutive failures
        # 3. Recent refresh was successful OR we have fallback rules
        
        has_rules = self.current_rules is not None
        not_too_many_failures = self.consecutive_failures < self.max_consecutive_failures
        has_recent_success = (self.last_refresh_success or 
                            self.last_good_rules is not None or 
                            self.current_rules == self.default_rules)
        
        return has_rules and not_too_many_failures and has_recent_success
    
    def reset_error_state(self):
        """Reset error tracking state"""
        self.consecutive_failures = 0
        self.last_error = None
    
    def set_refresh_interval(self, seconds: int):
        """
        Update refresh interval
        
        Args:
            seconds: New refresh interval in seconds
        """
        if seconds > 0:
            self.refresh_interval = seconds
            logger.info(f"Updated refresh interval to {seconds} seconds")


# Global instance for application use
_global_loader = None
_loader_lock = threading.Lock()


def get_cached_rules_loader() -> CachedRulesLoader:
    """
    Get the global cached rules loader instance (singleton pattern)
    
    Returns:
        CachedRulesLoader: Global loader instance
    """
    global _global_loader
    
    if _global_loader is None:
        with _loader_lock:
            if _global_loader is None:
                _global_loader = CachedRulesLoader()
    
    return _global_loader


def get_current_rules() -> RulesConfig:
    """
    Convenience function to get current rules
    
    Returns:
        RulesConfig: Current rules
    """
    loader = get_cached_rules_loader()
    return loader.get_rules()


def get_current_processor() -> RulesProcessor:
    """
    Convenience function to get current rules processor
    
    Returns:
        RulesProcessor: Processor with current rules
    """
    loader = get_cached_rules_loader()
    return loader.get_processor()


def force_rules_refresh() -> bool:
    """
    Convenience function to force refresh of rules
    
    Returns:
        bool: True if refresh was successful
    """
    loader = get_cached_rules_loader()
    return loader.force_refresh()


def get_loader_status() -> Dict[str, Any]:
    """
    Convenience function to get loader status
    
    Returns:
        Dict with loader status
    """
    loader = get_cached_rules_loader()
    return loader.get_status()


class RulesLoaderHealthCheck:
    """
    Health check utility for monitoring rules loader
    """
    
    @staticmethod
    def check_health() -> Dict[str, Any]:
        """
        Perform comprehensive health check
        
        Returns:
            Dict with health check results
        """
        loader = get_cached_rules_loader()
        status = loader.get_status()
        
        health_result = {
            'healthy': loader.is_healthy(),
            'status': status,
            'issues': [],
            'recommendations': []
        }
        
        # Check for issues
        if status['consecutive_failures'] > 0:
            health_result['issues'].append(f"Consecutive failures: {status['consecutive_failures']}")
        
        if not status['gcs_available']:
            health_result['issues'].append("GCS not available")
            health_result['recommendations'].append("Check GCS connectivity and credentials")
        
        if status['last_error']:
            health_result['issues'].append(f"Last error: {status['last_error']}")
        
        if status['performance_metrics']['fallback_count'] > 0:
            health_result['issues'].append(f"Fallback used {status['performance_metrics']['fallback_count']} times")
        
        # Performance recommendations
        cache_hit_rate = status['performance_metrics']['cache_hit_rate']
        if cache_hit_rate < 80:
            health_result['recommendations'].append(f"Low cache hit rate: {cache_hit_rate:.1f}%")
        
        return health_result
    
    @staticmethod
    def display_health_status():
        """Display health status in Streamlit (if available)"""
        try:
            health = RulesLoaderHealthCheck.check_health()
            
            if health['healthy']:
                st.success("✅ Rules loader is healthy")
            else:
                st.warning("⚠️ Rules loader has issues")
            
            if st.session_state.get('debug_enabled', False):
                with st.expander("🔍 Rules Loader Status"):
                    st.json(health)
                    
        except Exception as e:
            logger.error(f"Failed to display health status: {e}")


if __name__ == "__main__":
    # Test the cached loader
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing cached rules loader...")
    
    loader = CachedRulesLoader()
    
    # Test basic functionality
    rules = loader.get_rules()
    print(f"Got rules version: {rules.version}")
    
    processor = loader.get_processor()
    print(f"Created processor with {len(processor.rules_config.rules)} rules")
    
    # Test health check
    health = RulesLoaderHealthCheck.check_health()
    print(f"Health check: {health}")
    
    print("Cached rules loader test completed successfully!")