"""
Regression Testing System for Legal Description Reader
Automatically detects regressions and manages rollbacks after rule changes
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from utils.testing_harness import TestingHarness
from utils.gcs_rules import GCSRulesManager
from utils.gold_dataset import GoldDatasetManager
from utils.rules import RulesConfig
from utils.fingerprinting import FingerprintEngine


@dataclass
class RegressionTestResult:
    """Result of a regression test"""
    version_id: str
    passed: bool
    accuracy_change: float
    new_failures: List[str]
    fixed_failures: List[str]
    overall_accuracy: float
    baseline_accuracy: float
    test_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'version_id': self.version_id,
            'passed': self.passed,
            'accuracy_change': self.accuracy_change,
            'new_failures': self.new_failures,
            'fixed_failures': self.fixed_failures,
            'overall_accuracy': self.overall_accuracy,
            'baseline_accuracy': self.baseline_accuracy,
            'test_timestamp': self.test_timestamp.isoformat()
        }


class RegressionTestingEngine:
    """
    Engine for detecting regressions and managing automatic rollbacks
    """
    
    def __init__(self, regression_threshold: float = -0.01):
        """
        Initialize regression testing engine
        
        Args:
            regression_threshold: Accuracy decrease threshold for regression detection (default: -1%)
        """
        self.harness = TestingHarness()
        self.gcs_manager = GCSRulesManager()
        self.gold_manager = GoldDatasetManager()
        self.fingerprint_engine = FingerprintEngine()
        self.regression_threshold = regression_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def test_version_against_baseline(self, new_version_id: str, 
                                    baseline_version_id: str) -> RegressionTestResult:
        """
        Test a new version against a baseline version for regressions
        
        Args:
            new_version_id: Version ID of the new rules to test
            baseline_version_id: Version ID of the baseline rules
            
        Returns:
            RegressionTestResult with detailed comparison
        """
        self.logger.info(f"Testing version {new_version_id} against baseline {baseline_version_id}")
        
        try:
            # Load both versions
            new_rules = self.gcs_manager.load_rules_version(new_version_id)
            baseline_rules = self.gcs_manager.load_rules_version(baseline_version_id)
            
            if not new_rules or not baseline_rules:
                raise ValueError("Failed to load one or both rule versions")
            
            # Load gold dataset
            gold_entries = self.gold_manager.get_all_entries()
            if not gold_entries:
                raise ValueError("No gold dataset entries available")
            
            # Run harness on both versions
            baseline_results = self.harness.run_complete_harness(gold_entries, baseline_rules)
            new_results = self.harness.run_complete_harness(gold_entries, new_rules)
            
            # Extract accuracy metrics
            baseline_accuracy = baseline_results.get('accuracy_metrics', {}).get('overall_accuracy', 0.0)
            new_accuracy = new_results.get('accuracy_metrics', {}).get('overall_accuracy', 0.0)
            accuracy_change = new_accuracy - baseline_accuracy
            
            # Compare failure patterns
            baseline_failures = set(self._extract_failure_fingerprints(baseline_results))
            new_failures = set(self._extract_failure_fingerprints(new_results))
            
            # Identify new failures and fixed failures
            newly_failing = new_failures - baseline_failures
            newly_fixed = baseline_failures - new_failures
            
            # Determine if regression occurred
            is_regression = accuracy_change < self.regression_threshold
            
            self.logger.info(f"Accuracy change: {accuracy_change:+.2%}, Regression: {is_regression}")
            
            return RegressionTestResult(
                version_id=new_version_id,
                passed=not is_regression,
                accuracy_change=accuracy_change,
                new_failures=list(newly_failing),
                fixed_failures=list(newly_fixed),
                overall_accuracy=new_accuracy,
                baseline_accuracy=baseline_accuracy,
                test_timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Regression test failed: {str(e)}")
            # Return a failed test result
            return RegressionTestResult(
                version_id=new_version_id,
                passed=False,
                accuracy_change=-1.0,  # Indicate severe failure
                new_failures=[f"Test execution error: {str(e)}"],
                fixed_failures=[],
                overall_accuracy=0.0,
                baseline_accuracy=0.0,
                test_timestamp=datetime.now()
            )
    
    def test_current_version(self) -> RegressionTestResult:
        """
        Test the current version against the previous version
        
        Returns:
            RegressionTestResult for current vs previous version
        """
        try:
            versions = self.gcs_manager.list_versions()
            if len(versions) < 2:
                raise ValueError("Need at least 2 versions for regression testing")
            
            current_version = versions[0]  # Most recent
            previous_version = versions[1]  # Second most recent
            
            return self.test_version_against_baseline(current_version, previous_version)
            
        except Exception as e:
            self.logger.error(f"Current version regression test failed: {str(e)}")
            return RegressionTestResult(
                version_id="unknown",
                passed=False,
                accuracy_change=-1.0,
                new_failures=[f"Test setup error: {str(e)}"],
                fixed_failures=[],
                overall_accuracy=0.0,
                baseline_accuracy=0.0,
                test_timestamp=datetime.now()
            )
    
    def automatic_rollback_if_regression(self, new_version_id: str, 
                                       baseline_version_id: str) -> Dict[str, Any]:
        """
        Test for regression and automatically rollback if detected
        
        Args:
            new_version_id: Version to test
            baseline_version_id: Baseline version to compare against
            
        Returns:
            Dict with test results and rollback status
        """
        # Run regression test
        test_result = self.test_version_against_baseline(new_version_id, baseline_version_id)
        
        result = {
            'regression_test': test_result.to_dict(),
            'rollback_performed': False,
            'rollback_success': False
        }
        
        # If regression detected, perform rollback
        if not test_result.passed:
            self.logger.warning(f"Regression detected in version {new_version_id}, rolling back")
            
            try:
                rollback_success = self.gcs_manager.rollback_to_version(baseline_version_id)
                result['rollback_performed'] = True
                result['rollback_success'] = rollback_success
                
                if rollback_success:
                    self.logger.info(f"Successfully rolled back to version {baseline_version_id}")
                else:
                    self.logger.error(f"Failed to rollback to version {baseline_version_id}")
                    
            except Exception as e:
                self.logger.error(f"Rollback failed: {str(e)}")
                result['rollback_error'] = str(e)
        
        return result
    
    def validate_improvement(self, new_version_id: str, 
                           baseline_version_id: str,
                           minimum_improvement: float = 0.0) -> Dict[str, Any]:
        """
        Validate that a new version shows improvement over baseline
        
        Args:
            new_version_id: Version to validate
            baseline_version_id: Baseline version
            minimum_improvement: Minimum required improvement (default: 0%)
            
        Returns:
            Dict with validation results
        """
        test_result = self.test_version_against_baseline(new_version_id, baseline_version_id)
        
        improvement_achieved = test_result.accuracy_change >= minimum_improvement
        no_regression = test_result.passed
        
        return {
            'improvement_validated': improvement_achieved and no_regression,
            'accuracy_improvement': test_result.accuracy_change,
            'minimum_required': minimum_improvement,
            'regression_detected': not no_regression,
            'new_failures_count': len(test_result.new_failures),
            'fixed_failures_count': len(test_result.fixed_failures),
            'test_details': test_result.to_dict()
        }
    
    def run_comprehensive_regression_suite(self, target_version_id: str) -> Dict[str, Any]:
        """
        Run comprehensive regression testing against multiple baseline versions
        
        Args:
            target_version_id: Version to test comprehensively
            
        Returns:
            Dict with comprehensive test results
        """
        try:
            versions = self.gcs_manager.list_versions()
            if not versions:
                return {'error': 'No versions available for testing'}
            
            # Test against last 3 versions (if available)
            baseline_versions = [v for v in versions if v != target_version_id][:3]
            
            test_results = []
            overall_passed = True
            
            for baseline_version in baseline_versions:
                self.logger.info(f"Testing {target_version_id} against {baseline_version}")
                
                test_result = self.test_version_against_baseline(target_version_id, baseline_version)
                test_results.append({
                    'baseline_version': baseline_version,
                    'result': test_result.to_dict()
                })
                
                if not test_result.passed:
                    overall_passed = False
            
            # Calculate summary metrics
            accuracy_changes = [r['result']['accuracy_change'] for r in test_results]
            avg_accuracy_change = sum(accuracy_changes) / len(accuracy_changes) if accuracy_changes else 0.0
            
            return {
                'target_version': target_version_id,
                'overall_passed': overall_passed,
                'average_accuracy_change': avg_accuracy_change,
                'baselines_tested': len(baseline_versions),
                'individual_results': test_results,
                'summary': {
                    'best_improvement': max(accuracy_changes) if accuracy_changes else 0.0,
                    'worst_regression': min(accuracy_changes) if accuracy_changes else 0.0,
                    'consistent_improvement': all(change >= 0 for change in accuracy_changes)
                }
            }
            
        except Exception as e:
            return {'error': f'Comprehensive testing failed: {str(e)}'}
    
    def _extract_failure_fingerprints(self, harness_results: Dict[str, Any]) -> List[str]:
        """
        Extract failure fingerprints from harness results
        
        Args:
            harness_results: Results from testing harness
            
        Returns:
            List of failure fingerprints
        """
        failure_fingerprints = []
        
        failed_cases = harness_results.get('failed_test_cases', [])
        for case in failed_cases:
            # Extract missing fingerprints (what should have been found but wasn't)
            comparison = case.get('comparison', {})
            missing_fps = comparison.get('missing_fingerprints', [])
            failure_fingerprints.extend(missing_fps)
            
            # Also include bucket mismatches as a type of failure
            if not comparison.get('bucket_match', True):
                expected_bucket = case.get('expected_bucket', 'unknown')
                actual_bucket = case.get('actual_bucket', 'unknown')
                bucket_failure = f"bucket_mismatch:{actual_bucket}->{expected_bucket}"
                failure_fingerprints.append(bucket_failure)
        
        return failure_fingerprints
    
    def get_regression_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of regression tests
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of historical regression test results
        """
        # This would typically load from a database or log file
        # For now, return empty list as placeholder
        return []
    
    def set_regression_threshold(self, threshold: float):
        """
        Update the regression threshold
        
        Args:
            threshold: New threshold for regression detection (negative value)
        """
        if threshold > 0:
            raise ValueError("Regression threshold should be negative (e.g., -0.01 for -1%)")
        
        self.regression_threshold = threshold
        self.logger.info(f"Regression threshold updated to {threshold:.2%}")


class RegressionMonitor:
    """
    Monitor for continuous regression detection
    """
    
    def __init__(self):
        self.testing_engine = RegressionTestingEngine()
        self.logger = logging.getLogger(__name__)
    
    def monitor_new_deployments(self) -> Dict[str, Any]:
        """
        Monitor for new deployments and automatically test for regressions
        
        Returns:
            Dict with monitoring results
        """
        try:
            # Check if there's a new version to test
            current_test = self.testing_engine.test_current_version()
            
            if not current_test.passed:
                # Regression detected - trigger automatic rollback
                versions = self.testing_engine.gcs_manager.list_versions()
                if len(versions) >= 2:
                    rollback_result = self.testing_engine.automatic_rollback_if_regression(
                        versions[0], versions[1]
                    )
                    return {
                        'regression_detected': True,
                        'automatic_action': 'rollback',
                        'rollback_result': rollback_result
                    }
            
            return {
                'regression_detected': False,
                'current_accuracy': current_test.overall_accuracy,
                'accuracy_change': current_test.accuracy_change,
                'test_timestamp': current_test.test_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Regression monitoring failed: {str(e)}")
            return {
                'error': f'Monitoring failed: {str(e)}',
                'regression_detected': None
            }


def create_regression_engine(threshold: float = -0.01) -> RegressionTestingEngine:
    """
    Factory function to create a regression testing engine
    
    Args:
        threshold: Regression detection threshold
        
    Returns:
        RegressionTestingEngine instance
    """
    return RegressionTestingEngine(threshold)


def test_for_regression(new_version: str, baseline_version: str) -> Dict[str, Any]:
    """
    Convenience function to test for regression between two versions
    
    Args:
        new_version: New version ID to test
        baseline_version: Baseline version ID
        
    Returns:
        Dict with regression test results
    """
    engine = create_regression_engine()
    result = engine.test_version_against_baseline(new_version, baseline_version)
    return result.to_dict()


def monitor_and_rollback_if_needed() -> Dict[str, Any]:
    """
    Convenience function to monitor current deployment and rollback if regression detected
    
    Returns:
        Dict with monitoring and rollback results
    """
    monitor = RegressionMonitor()
    return monitor.monitor_new_deployments()