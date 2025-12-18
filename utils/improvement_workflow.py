"""
Self-Improvement Workflow Orchestrator for Legal Description Reader
Coordinates the complete improvement cycle from failure analysis to patch deployment
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.testing_harness import TestingHarness
from utils.failure_clustering import FailureClusteringEngine
from utils.patch_generator import GPTPatchGenerator
from utils.patch_application import PatchApplicationEngine, PatchApplicationResult
from utils.gcs_rules import GCSRulesManager
from utils.cached_rules_loader import CachedRulesLoader
from utils.gold_dataset import GoldDatasetManager
from utils.regression_testing import RegressionTestingEngine


class ImprovementWorkflowOrchestrator:
    """
    Orchestrates the complete self-improvement workflow
    """
    
    def __init__(self):
        self.harness = TestingHarness()
        self.clustering_engine = FailureClusteringEngine()
        self.patch_generator = GPTPatchGenerator()
        self.patch_engine = PatchApplicationEngine()
        self.gcs_manager = GCSRulesManager()
        self.rules_loader = CachedRulesLoader()
        self.gold_manager = GoldDatasetManager()
        self.regression_engine = RegressionTestingEngine()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_complete_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run a complete improvement cycle from testing to deployment
        
        Returns:
            Dict with cycle results and metrics
        """
        cycle_start = datetime.now()
        self.logger.info("Starting complete improvement cycle")
        
        try:
            # Step 1: Load current rules and gold dataset
            current_rules = self.rules_loader.get_current_rules()
            if not current_rules:
                return self._create_error_result("Failed to load current rules")
            
            gold_entries = self.gold_manager.get_all_entries()
            if not gold_entries:
                return self._create_error_result("No gold dataset entries available")
            
            self.logger.info(f"Loaded {len(current_rules.rules)} rules and {len(gold_entries)} gold entries")
            
            # Step 2: Run testing harness
            harness_results = self.harness.run_complete_harness(gold_entries, current_rules)
            if not harness_results.get('success', False):
                return self._create_error_result("Harness execution failed", harness_results)
            
            accuracy = harness_results.get('accuracy_metrics', {}).get('overall_accuracy', 0.0)
            self.logger.info(f"Current accuracy: {accuracy:.2%}")
            
            # Step 3: Analyze failures and cluster
            failures = harness_results.get('failed_test_cases', [])
            if not failures:
                self.logger.info("No failures found - system is performing well")
                return self._create_success_result("No improvements needed", {
                    'current_accuracy': accuracy,
                    'failures_count': 0,
                    'patches_applied': 0
                })
            
            clustering_results = self.clustering_engine.cluster_failures(failures)
            top_clusters = clustering_results.get('priority_clusters', [])[:3]  # Top 3
            
            if not top_clusters:
                return self._create_error_result("No actionable failure clusters found")
            
            self.logger.info(f"Found {len(top_clusters)} priority failure clusters")
            
            # Step 4: Generate patches
            patch_results = self.patch_generator.generate_patches(clustering_results, current_rules)
            if not patch_results:
                return self._create_error_result("No patches generated")
            
            self.logger.info(f"Generated {len(patch_results)} patch sets")
            
            # Step 5: Apply patches with validation
            application_results = []
            successful_patches = 0
            
            for i, patch_set in enumerate(patch_results):
                self.logger.info(f"Applying patch set {i+1}/{len(patch_results)}")
                
                result = self.patch_engine.apply_patch_set(patch_set, current_rules)
                application_results.append(result)
                
                if result.success:
                    successful_patches += 1
                    current_rules = result.new_rules  # Use updated rules for next patch
                    self.logger.info(f"Patch set {i+1} applied successfully: {result.new_version_id}")
                else:
                    self.logger.warning(f"Patch set {i+1} failed: {result.validation_errors}")
            
            if successful_patches == 0:
                return self._create_error_result("No patches could be applied successfully", {
                    'patch_errors': [r.validation_errors for r in application_results]
                })
            
            # Step 6: Deploy best version and test
            best_result = None
            for result in application_results:
                if result.success:
                    best_result = result
                    break
            
            if best_result:
                # Deploy the new version
                deployment_result = self.patch_engine.deploy_new_version(best_result.new_version_id)
                if not deployment_result.get('success', False):
                    return self._create_error_result("Failed to deploy new version", deployment_result)
                
                # Test for regression using the regression engine
                previous_version_id = self._get_previous_version_id()
                if previous_version_id:
                    regression_result = self.regression_engine.automatic_rollback_if_regression(
                        best_result.new_version_id, previous_version_id
                    )
                    
                    if regression_result.get('rollback_performed', False):
                        return self._create_error_result("Regression detected, automatic rollback performed", {
                            'regression_test': regression_result['regression_test'],
                            'rollback_success': regression_result.get('rollback_success', False)
                        })
                    
                    # Get the actual accuracy metrics from regression test
                    regression_test = regression_result.get('regression_test', {})
                    new_accuracy = regression_test.get('overall_accuracy', 0.0)
                    improvement = regression_test.get('accuracy_change', 0.0)
                else:
                    # Fallback to manual testing if no previous version
                    new_harness_results = self.harness.run_complete_harness(gold_entries, best_result.new_rules)
                    new_accuracy = new_harness_results.get('accuracy_metrics', {}).get('overall_accuracy', 0.0)
                    improvement = new_accuracy - accuracy
                
                self.logger.info(f"New accuracy: {new_accuracy:.2%} (improvement: {improvement:+.2%})")
                
                cycle_end = datetime.now()
                cycle_duration = (cycle_end - cycle_start).total_seconds()
                
                return self._create_success_result("Improvement cycle completed successfully", {
                    'previous_accuracy': accuracy,
                    'new_accuracy': new_accuracy,
                    'improvement': improvement,
                    'patches_applied': successful_patches,
                    'new_version_id': best_result.new_version_id,
                    'cycle_duration_seconds': cycle_duration,
                    'failures_addressed': len(failures)
                })
            
            return self._create_error_result("No successful patch deployment")
            
        except Exception as e:
            self.logger.error(f"Improvement cycle failed: {str(e)}")
            return self._create_error_result(f"Unexpected error: {str(e)}")
    
    def run_testing_only(self) -> Dict[str, Any]:
        """
        Run only the testing harness without improvements
        
        Returns:
            Dict with testing results
        """
        try:
            current_rules = self.rules_loader.get_current_rules()
            gold_entries = self.gold_manager.get_all_entries()
            
            if not current_rules or not gold_entries:
                return self._create_error_result("Missing rules or gold dataset")
            
            harness_results = self.harness.run_complete_harness(gold_entries, current_rules)
            
            return {
                'success': True,
                'type': 'testing_only',
                'results': harness_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._create_error_result(f"Testing failed: {str(e)}")
    
    def run_patch_generation_only(self, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run only patch generation from provided failure analysis
        
        Args:
            failure_analysis: Pre-computed failure analysis data
            
        Returns:
            Dict with patch generation results
        """
        try:
            current_rules = self.rules_loader.get_current_rules()
            if not current_rules:
                return self._create_error_result("Failed to load current rules")
            
            patch_results = self.patch_generator.generate_patches(failure_analysis, current_rules)
            
            return {
                'success': True,
                'type': 'patch_generation_only',
                'patch_sets': [ps.to_dict() for ps in patch_results],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._create_error_result(f"Patch generation failed: {str(e)}")
    
    def validate_patches_only(self, patches_json: str) -> Dict[str, Any]:
        """
        Validate patches without applying them
        
        Args:
            patches_json: JSON string with patch data
            
        Returns:
            Dict with validation results
        """
        try:
            from utils.patch_application import apply_patches_from_json
            
            current_rules = self.rules_loader.get_current_rules()
            if not current_rules:
                return self._create_error_result("Failed to load current rules")
            
            # This will validate but not actually deploy
            results = apply_patches_from_json(patches_json, current_rules)
            
            validation_summary = {
                'total_patch_sets': len(results),
                'successful_validations': sum(1 for r in results if r.success),
                'failed_validations': sum(1 for r in results if not r.success),
                'validation_errors': [r.validation_errors for r in results if not r.success]
            }
            
            return {
                'success': True,
                'type': 'validation_only',
                'validation_summary': validation_summary,
                'detailed_results': [
                    {
                        'success': r.success,
                        'errors': r.validation_errors,
                        'operations': r.applied_operations
                    }
                    for r in results
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._create_error_result(f"Validation failed: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics
        
        Returns:
            Dict with system status information
        """
        try:
            # Get current rules info
            current_rules = self.rules_loader.get_current_rules()
            rules_count = len(current_rules.rules) if current_rules else 0
            
            # Get gold dataset info
            gold_entries = self.gold_manager.get_all_entries()
            gold_count = len(gold_entries) if gold_entries else 0
            
            # Get version info
            versions = self.gcs_manager.list_versions()
            
            # Get cache status
            cache_stats = self.rules_loader.get_cache_stats()
            
            return {
                'success': True,
                'status': {
                    'rules_count': rules_count,
                    'gold_dataset_size': gold_count,
                    'available_versions': len(versions) if versions else 0,
                    'latest_version': versions[0] if versions else None,
                    'cache_status': cache_stats,
                    'last_check': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Status check failed: {str(e)}")
    
    def _create_success_result(self, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized success result"""
        result = {
            'success': True,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            result.update(data)
        return result
    
    def _create_error_result(self, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized error result"""
        result = {
            'success': False,
            'error': message,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            result['details'] = data
        return result
    
    def _get_previous_version_id(self) -> Optional[str]:
        """Get the previous version ID for rollback"""
        try:
            versions = self.gcs_manager.list_versions()
            if len(versions) >= 2:
                return versions[1]  # Second most recent
            return None
        except:
            return None


def create_improvement_orchestrator() -> ImprovementWorkflowOrchestrator:
    """
    Factory function to create an improvement workflow orchestrator
    
    Returns:
        ImprovementWorkflowOrchestrator instance
    """
    return ImprovementWorkflowOrchestrator()


def run_automated_improvement() -> Dict[str, Any]:
    """
    Convenience function to run a complete automated improvement cycle
    
    Returns:
        Dict with improvement results
    """
    orchestrator = create_improvement_orchestrator()
    return orchestrator.run_complete_improvement_cycle()


def run_system_health_check() -> Dict[str, Any]:
    """
    Convenience function to check system health and status
    
    Returns:
        Dict with system status
    """
    orchestrator = create_improvement_orchestrator()
    return orchestrator.get_system_status()