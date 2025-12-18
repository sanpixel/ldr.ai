"""
Testing harness for Legal Description Reader
Automated comparison system using fingerprints for continuous accuracy monitoring
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from utils.gold_dataset import load_all_gold_entries, GoldDatasetManager
from utils.fingerprinting import FingerprintEngine, compare_fingerprint_sets
from utils.schema import SchemaOutput
from utils.cached_rules_loader import get_current_processor
from utils.rule_versions import save_harness_result, get_active_version
import streamlit as st
import logging

logger = logging.getLogger(__name__)


class TestingHarness:
    """
    Automated testing harness for comparing extractor output against gold dataset
    """
    
    def __init__(self):
        self.fingerprint_engine = FingerprintEngine()
        self.gold_manager = GoldDatasetManager()
        self.results_cache = {}
        
    def run_full_harness(self, rule_version_id: str = None) -> Dict[str, Any]:
        """
        Run complete harness test against all gold dataset entries
        
        Args:
            rule_version_id: Optional rule version ID for tracking
            
        Returns:
            Dict with comprehensive test results
        """
        start_time = time.time()
        
        # Load gold dataset
        gold_entries = load_all_gold_entries()
        
        if not gold_entries:
            return {
                'error': 'No gold dataset entries found',
                'total_cases': 0,
                'passed_cases': 0,
                'failed_cases': 0,
                'execution_time_ms': 0
            }
        
        # Get current processor
        processor = get_current_processor()
        
        # Get rule version info
        if not rule_version_id:
            active_version = get_active_version()
            rule_version_id = active_version.get('version_id') if active_version else 'unknown'
        
        # Run tests
        results = {
            'rule_version_id': rule_version_id,
            'total_cases': len(gold_entries),
            'passed_cases': 0,
            'failed_cases': 0,
            'test_results': [],
            'failure_summary': {},
            'accuracy_metrics': {},
            'execution_time_ms': 0,
            'run_at': datetime.now().isoformat()
        }
        
        failures = []
        
        for i, gold_entry in enumerate(gold_entries):
            try:
                # Run single test case
                test_result = self.run_single_test_case(
                    gold_entry, processor, case_index=i
                )
                
                results['test_results'].append(test_result)
                
                if test_result['passed']:
                    results['passed_cases'] += 1
                else:
                    results['failed_cases'] += 1
                    failures.append(test_result)
                
            except Exception as e:
                logger.error(f"Error running test case {i}: {e}")
                
                error_result = {
                    'case_index': i,
                    'entry_id': gold_entry.get('id', 'unknown'),
                    'passed': False,
                    'error': str(e),
                    'execution_error': True
                }
                
                results['test_results'].append(error_result)
                results['failed_cases'] += 1
                failures.append(error_result)
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        results['execution_time_ms'] = execution_time_ms
        
        # Generate failure analysis
        results['failure_details'] = self.analyze_failures(failures)
        
        # Calculate accuracy metrics
        results['accuracy_metrics'] = self.calculate_accuracy_metrics(results['test_results'])
        
        # Save results to database
        try:
            save_harness_result(
                version_id=rule_version_id,
                total_cases=results['total_cases'],
                passed_cases=results['passed_cases'],
                failed_cases=results['failed_cases'],
                failure_details=results['failure_details'],
                execution_time_ms=execution_time_ms,
                triggered_by='harness_run'
            )
        except Exception as e:
            logger.error(f"Failed to save harness results: {e}")
        
        return results
    
    def run_single_test_case(self, gold_entry: Dict[str, Any], 
                           processor, case_index: int) -> Dict[str, Any]:
        """
        Run a single test case against gold dataset entry
        
        Args:
            gold_entry: Gold dataset entry
            processor: Rules processor to test
            case_index: Index of test case
            
        Returns:
            Dict with test case results
        """
        entry_id = gold_entry.get('id', 'unknown')
        text = gold_entry.get('text', '')
        expected_output_dict = gold_entry.get('gold_output', {})
        
        try:
            # Parse expected output
            expected_output = SchemaOutput.from_dict(expected_output_dict)
            
            # Run current extractor
            extraction_results = processor.extract_all_patterns(text)
            
            # Convert extraction results to schema output
            # This is a simplified conversion - in practice would need more sophisticated mapping
            actual_lines = []
            for result_type, matches in extraction_results.items():
                for match in matches:
                    # Convert match to LineData - simplified for now
                    pass
            
            # For now, create a basic actual output for testing
            # In practice, this would use the full extraction pipeline
            actual_output = SchemaOutput(bucket='no_bearings', lines=[])
            
            # Compare using fingerprints
            comparison = self.fingerprint_engine.compare_outputs(actual_output, expected_output)
            
            # Determine if test passed
            passed = (comparison['missing_count'] == 0 and 
                     comparison['extra_count'] == 0 and
                     comparison['bucket_match'])
            
            return {
                'case_index': case_index,
                'entry_id': entry_id,
                'passed': passed,
                'comparison': comparison,
                'expected_bucket': expected_output.bucket,
                'actual_bucket': actual_output.bucket,
                'expected_line_count': len(expected_output.lines),
                'actual_line_count': len(actual_output.lines),
                'source_file': gold_entry.get('source_file'),
                'text_length': len(text)
            }
            
        except Exception as e:
            return {
                'case_index': case_index,
                'entry_id': entry_id,
                'passed': False,
                'error': str(e),
                'execution_error': True
            }
    
    def analyze_failures(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze failure patterns for patch generation
        
        Args:
            failures: List of failed test cases
            
        Returns:
            Dict with failure analysis
        """
        analysis = {
            'total_failures': len(failures),
            'failure_types': {},
            'bucket_mismatches': {},
            'missing_patterns': {},
            'extra_patterns': {},
            'execution_errors': 0,
            'common_failure_patterns': []
        }
        
        for failure in failures:
            # Count execution errors
            if failure.get('execution_error'):
                analysis['execution_errors'] += 1
                continue
            
            comparison = failure.get('comparison', {})
            
            # Analyze bucket mismatches
            if not comparison.get('bucket_match', True):
                expected_bucket = failure.get('expected_bucket', 'unknown')
                actual_bucket = failure.get('actual_bucket', 'unknown')
                mismatch_key = f"{actual_bucket}->{expected_bucket}"
                
                analysis['bucket_mismatches'][mismatch_key] = \
                    analysis['bucket_mismatches'].get(mismatch_key, 0) + 1
            
            # Analyze missing patterns
            missing_fps = comparison.get('missing_fingerprints', [])
            for fp in missing_fps:
                analysis['missing_patterns'][fp] = \
                    analysis['missing_patterns'].get(fp, 0) + 1
            
            # Analyze extra patterns
            extra_fps = comparison.get('extra_fingerprints', [])
            for fp in extra_fps:
                analysis['extra_patterns'][fp] = \
                    analysis['extra_patterns'].get(fp, 0) + 1
        
        # Identify common failure patterns
        analysis['common_failure_patterns'] = self.identify_common_patterns(analysis)
        
        return analysis
    
    def identify_common_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify the most common failure patterns for prioritization
        
        Args:
            analysis: Failure analysis data
            
        Returns:
            List of common patterns sorted by frequency
        """
        patterns = []
        
        # Most common missing patterns
        missing_patterns = analysis.get('missing_patterns', {})
        for pattern, count in sorted(missing_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            patterns.append({
                'type': 'missing_pattern',
                'pattern': pattern,
                'frequency': count,
                'description': f"Missing fingerprint: {pattern}"
            })
        
        # Most common bucket mismatches
        bucket_mismatches = analysis.get('bucket_mismatches', {})
        for mismatch, count in sorted(bucket_mismatches.items(), key=lambda x: x[1], reverse=True)[:3]:
            patterns.append({
                'type': 'bucket_mismatch',
                'pattern': mismatch,
                'frequency': count,
                'description': f"Bucket classification error: {mismatch}"
            })
        
        # Most common extra patterns
        extra_patterns = analysis.get('extra_patterns', {})
        for pattern, count in sorted(extra_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
            patterns.append({
                'type': 'extra_pattern',
                'pattern': pattern,
                'frequency': count,
                'description': f"Unexpected fingerprint: {pattern}"
            })
        
        return sorted(patterns, key=lambda x: x['frequency'], reverse=True)
    
    def calculate_accuracy_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            test_results: List of test case results
            
        Returns:
            Dict with accuracy metrics
        """
        if not test_results:
            return {}
        
        total_cases = len(test_results)
        passed_cases = sum(1 for result in test_results if result.get('passed', False))
        
        # Basic accuracy
        accuracy = passed_cases / total_cases if total_cases > 0 else 0.0
        
        # Bucket classification accuracy
        bucket_correct = sum(1 for result in test_results 
                           if result.get('comparison', {}).get('bucket_match', False))
        bucket_accuracy = bucket_correct / total_cases if total_cases > 0 else 0.0
        
        # Line extraction metrics
        total_precision = 0.0
        total_recall = 0.0
        valid_comparisons = 0
        
        for result in test_results:
            comparison = result.get('comparison', {})
            if 'precision' in comparison and 'recall' in comparison:
                total_precision += comparison['precision']
                total_recall += comparison['recall']
                valid_comparisons += 1
        
        avg_precision = total_precision / valid_comparisons if valid_comparisons > 0 else 0.0
        avg_recall = total_recall / valid_comparisons if valid_comparisons > 0 else 0.0
        f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) \
                  if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'overall_accuracy': round(accuracy, 4),
            'bucket_accuracy': round(bucket_accuracy, 4),
            'average_precision': round(avg_precision, 4),
            'average_recall': round(avg_recall, 4),
            'f1_score': round(f1_score, 4),
            'pass_rate': round(accuracy, 4)  # Same as overall accuracy
        }
    
    def generate_failures_json(self, harness_results: Dict[str, Any]) -> str:
        """
        Generate failures.json output for patch generation
        
        Args:
            harness_results: Results from run_full_harness
            
        Returns:
            JSON string with failure data for patch generation
        """
        failures_data = {
            'harness_run_info': {
                'rule_version_id': harness_results.get('rule_version_id'),
                'run_at': harness_results.get('run_at'),
                'total_cases': harness_results.get('total_cases', 0),
                'failed_cases': harness_results.get('failed_cases', 0),
                'execution_time_ms': harness_results.get('execution_time_ms', 0)
            },
            'failure_analysis': harness_results.get('failure_details', {}),
            'accuracy_metrics': harness_results.get('accuracy_metrics', {}),
            'failed_test_cases': [],
            'patch_generation_data': {
                'priority_patterns': [],
                'suggested_fixes': []
            }
        }
        
        # Extract failed test cases with context
        for result in harness_results.get('test_results', []):
            if not result.get('passed', True):
                failure_case = {
                    'entry_id': result.get('entry_id'),
                    'case_index': result.get('case_index'),
                    'comparison': result.get('comparison', {}),
                    'expected_bucket': result.get('expected_bucket'),
                    'actual_bucket': result.get('actual_bucket'),
                    'source_file': result.get('source_file'),
                    'error': result.get('error')
                }
                failures_data['failed_test_cases'].append(failure_case)
        
        # Add priority patterns for patch generation
        common_patterns = harness_results.get('failure_details', {}).get('common_failure_patterns', [])
        failures_data['patch_generation_data']['priority_patterns'] = common_patterns[:3]  # Top 3
        
        return json.dumps(failures_data, indent=2)
    
    def run_regression_check(self, previous_results: Dict[str, Any], 
                           current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current results against previous results to detect regressions
        
        Args:
            previous_results: Previous harness results
            current_results: Current harness results
            
        Returns:
            Dict with regression analysis
        """
        regression_check = {
            'has_regression': False,
            'improvement_detected': False,
            'metrics_comparison': {},
            'new_failures': [],
            'fixed_cases': [],
            'recommendation': 'keep'  # 'keep', 'rollback', 'investigate'
        }
        
        # Compare accuracy metrics
        prev_metrics = previous_results.get('accuracy_metrics', {})
        curr_metrics = current_results.get('accuracy_metrics', {})
        
        for metric_name in ['overall_accuracy', 'bucket_accuracy', 'f1_score']:
            prev_value = prev_metrics.get(metric_name, 0.0)
            curr_value = curr_metrics.get(metric_name, 0.0)
            
            regression_check['metrics_comparison'][metric_name] = {
                'previous': prev_value,
                'current': curr_value,
                'change': curr_value - prev_value,
                'improved': curr_value > prev_value
            }
            
            # Detect significant regression (>5% decrease)
            if curr_value < prev_value - 0.05:
                regression_check['has_regression'] = True
        
        # Compare failure counts
        prev_failed = previous_results.get('failed_cases', 0)
        curr_failed = current_results.get('failed_cases', 0)
        
        if curr_failed < prev_failed:
            regression_check['improvement_detected'] = True
        elif curr_failed > prev_failed:
            regression_check['has_regression'] = True
        
        # Determine recommendation
        if regression_check['has_regression']:
            regression_check['recommendation'] = 'rollback'
        elif regression_check['improvement_detected']:
            regression_check['recommendation'] = 'keep'
        else:
            regression_check['recommendation'] = 'investigate'
        
        return regression_check


def run_harness_for_version(rule_version_id: str = None) -> Dict[str, Any]:
    """
    Convenience function to run harness for a specific rule version
    
    Args:
        rule_version_id: Rule version to test (uses active if None)
        
    Returns:
        Dict with harness results
    """
    harness = TestingHarness()
    return harness.run_full_harness(rule_version_id)


def compare_harness_results(version_id_1: str, version_id_2: str) -> Dict[str, Any]:
    """
    Compare harness results between two rule versions
    
    Args:
        version_id_1: First version ID
        version_id_2: Second version ID
        
    Returns:
        Dict with comparison results
    """
    # This would load results from database and compare
    # For now, return placeholder
    return {
        'version_1': version_id_1,
        'version_2': version_id_2,
        'comparison': 'Not implemented yet'
    }


class HarnessScheduler:
    """
    Scheduler for automated harness runs
    """
    
    def __init__(self):
        self.last_run_time = None
        self.run_interval_seconds = 3600  # 1 hour default
    
    def should_run_harness(self) -> bool:
        """Check if harness should be run based on schedule"""
        if self.last_run_time is None:
            return True
        
        time_since_last = time.time() - self.last_run_time
        return time_since_last >= self.run_interval_seconds
    
    def run_scheduled_harness(self) -> Optional[Dict[str, Any]]:
        """Run harness if scheduled"""
        if self.should_run_harness():
            harness = TestingHarness()
            results = harness.run_full_harness()
            self.last_run_time = time.time()
            return results
        return None