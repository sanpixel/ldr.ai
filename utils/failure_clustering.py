"""
Failure clustering system for Legal Description Reader
Groups failures by pattern to prioritize fixes for automated patch generation
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from utils.fingerprinting import extract_fingerprint_components, group_fingerprints_by_type
import re


class FailureCluster:
    """
    Represents a cluster of similar failures
    """
    
    def __init__(self, cluster_id: str, cluster_type: str):
        self.cluster_id = cluster_id
        self.cluster_type = cluster_type  # 'missing_pattern', 'extra_pattern', 'bucket_mismatch', 'extractor_failure'
        self.failures = []
        self.frequency = 0
        self.pattern = None
        self.extractor_ids = set()
        self.affected_cases = set()
        
    def add_failure(self, failure_data: Dict[str, Any]):
        """Add a failure to this cluster"""
        self.failures.append(failure_data)
        self.frequency += 1
        
        # Track affected cases
        if 'case_index' in failure_data:
            self.affected_cases.add(failure_data['case_index'])
        
        # Track extractor IDs if available
        if 'extractor_id' in failure_data:
            self.extractor_ids.add(failure_data['extractor_id'])
    
    def get_representative_failure(self) -> Dict[str, Any]:
        """Get a representative failure from this cluster"""
        if self.failures:
            return self.failures[0]
        return {}
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary information about this cluster"""
        return {
            'cluster_id': self.cluster_id,
            'cluster_type': self.cluster_type,
            'frequency': self.frequency,
            'pattern': self.pattern,
            'extractor_ids': list(self.extractor_ids),
            'affected_case_count': len(self.affected_cases),
            'representative_failure': self.get_representative_failure()
        }


class FailureClusteringEngine:
    """
    Engine for clustering failures by various patterns
    """
    
    def __init__(self):
        self.clusters = {}
        self.cluster_counter = 0
    
    def cluster_failures(self, harness_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster failures from harness results
        
        Args:
            harness_results: Results from testing harness
            
        Returns:
            Dict with clustering analysis
        """
        self.clusters = {}
        self.cluster_counter = 0
        
        # Extract failed test cases
        failed_cases = [
            result for result in harness_results.get('test_results', [])
            if not result.get('passed', True)
        ]
        
        if not failed_cases:
            return {
                'total_failures': 0,
                'clusters': [],
                'top_clusters': [],
                'clustering_summary': {}
            }
        
        # Cluster by different dimensions
        missing_pattern_clusters = self._cluster_by_missing_patterns(failed_cases)
        extractor_id_clusters = self._cluster_by_extractor_id(failed_cases)
        bucket_mismatch_clusters = self._cluster_by_bucket_mismatch(failed_cases)
        
        # Combine all clusters
        all_clusters = []
        all_clusters.extend(missing_pattern_clusters)
        all_clusters.extend(extractor_id_clusters)
        all_clusters.extend(bucket_mismatch_clusters)
        
        # Sort by frequency and get top clusters
        all_clusters.sort(key=lambda c: c.frequency, reverse=True)
        top_clusters = all_clusters[:3]  # Top 3 as specified in requirements
        
        # Filter out single-occurrence failures as specified
        significant_clusters = [c for c in all_clusters if c.frequency > 1]
        
        return {
            'total_failures': len(failed_cases),
            'total_clusters': len(all_clusters),
            'significant_clusters': len(significant_clusters),
            'clusters': [c.get_cluster_summary() for c in all_clusters],
            'top_clusters': [c.get_cluster_summary() for c in top_clusters],
            'clustering_summary': self._generate_clustering_summary(all_clusters, failed_cases)
        }
    
    def _cluster_by_missing_patterns(self, failed_cases: List[Dict[str, Any]]) -> List[FailureCluster]:
        """
        Cluster failures by missing fingerprint patterns
        
        Args:
            failed_cases: List of failed test cases
            
        Returns:
            List of clusters for missing patterns
        """
        missing_pattern_groups = defaultdict(list)
        
        for case in failed_cases:
            comparison = case.get('comparison', {})
            missing_fingerprints = comparison.get('missing_fingerprints', [])
            
            for fingerprint in missing_fingerprints:
                # Group by exact fingerprint match
                missing_pattern_groups[fingerprint].append({
                    'case_index': case.get('case_index'),
                    'entry_id': case.get('entry_id'),
                    'fingerprint': fingerprint,
                    'source_file': case.get('source_file'),
                    'failure_type': 'missing_pattern'
                })
        
        clusters = []
        for pattern, failures in missing_pattern_groups.items():
            cluster = FailureCluster(
                cluster_id=f"missing_{self.cluster_counter}",
                cluster_type="missing_pattern"
            )
            cluster.pattern = pattern
            
            for failure in failures:
                cluster.add_failure(failure)
            
            clusters.append(cluster)
            self.cluster_counter += 1
        
        return clusters
    
    def _cluster_by_extractor_id(self, failed_cases: List[Dict[str, Any]]) -> List[FailureCluster]:
        """
        Cluster failures by extractor ID for targeted fixes
        
        Args:
            failed_cases: List of failed test cases
            
        Returns:
            List of clusters for extractor failures
        """
        # For now, we don't have extractor_id in the failure data
        # This would be populated when we have more detailed extraction tracking
        
        extractor_groups = defaultdict(list)
        
        for case in failed_cases:
            # Extract potential extractor information from patterns
            comparison = case.get('comparison', {})
            missing_fps = comparison.get('missing_fingerprints', [])
            extra_fps = comparison.get('extra_fingerprints', [])
            
            # Group by fingerprint type (which corresponds to extractor type)
            for fp in missing_fps + extra_fps:
                fp_components = extract_fingerprint_components(fp)
                fp_type = fp_components.get('type', 'unknown')
                
                extractor_groups[fp_type].append({
                    'case_index': case.get('case_index'),
                    'entry_id': case.get('entry_id'),
                    'extractor_type': fp_type,
                    'fingerprint': fp,
                    'source_file': case.get('source_file'),
                    'failure_type': 'extractor_failure'
                })
        
        clusters = []
        for extractor_type, failures in extractor_groups.items():
            if len(failures) > 1:  # Only create cluster if multiple failures
                cluster = FailureCluster(
                    cluster_id=f"extractor_{self.cluster_counter}",
                    cluster_type="extractor_failure"
                )
                cluster.pattern = f"extractor_type:{extractor_type}"
                
                for failure in failures:
                    cluster.add_failure(failure)
                
                clusters.append(cluster)
                self.cluster_counter += 1
        
        return clusters
    
    def _cluster_by_bucket_mismatch(self, failed_cases: List[Dict[str, Any]]) -> List[FailureCluster]:
        """
        Cluster failures by bucket classification mismatches
        
        Args:
            failed_cases: List of failed test cases
            
        Returns:
            List of clusters for bucket mismatches
        """
        bucket_mismatch_groups = defaultdict(list)
        
        for case in failed_cases:
            comparison = case.get('comparison', {})
            
            if not comparison.get('bucket_match', True):
                expected_bucket = case.get('expected_bucket', 'unknown')
                actual_bucket = case.get('actual_bucket', 'unknown')
                mismatch_pattern = f"{actual_bucket}->{expected_bucket}"
                
                bucket_mismatch_groups[mismatch_pattern].append({
                    'case_index': case.get('case_index'),
                    'entry_id': case.get('entry_id'),
                    'expected_bucket': expected_bucket,
                    'actual_bucket': actual_bucket,
                    'mismatch_pattern': mismatch_pattern,
                    'source_file': case.get('source_file'),
                    'failure_type': 'bucket_mismatch'
                })
        
        clusters = []
        for mismatch_pattern, failures in bucket_mismatch_groups.items():
            cluster = FailureCluster(
                cluster_id=f"bucket_{self.cluster_counter}",
                cluster_type="bucket_mismatch"
            )
            cluster.pattern = mismatch_pattern
            
            for failure in failures:
                cluster.add_failure(failure)
            
            clusters.append(cluster)
            self.cluster_counter += 1
        
        return clusters
    
    def _generate_clustering_summary(self, clusters: List[FailureCluster], 
                                   failed_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics about clustering
        
        Args:
            clusters: List of all clusters
            failed_cases: List of failed test cases
            
        Returns:
            Dict with clustering summary
        """
        if not clusters:
            return {
                'total_failures': len(failed_cases),
                'clustered_failures': 0,
                'unclustered_failures': len(failed_cases),
                'cluster_types': {},
                'largest_cluster_size': 0,
                'average_cluster_size': 0.0
            }
        
        # Count failures by cluster type
        cluster_type_counts = Counter(c.cluster_type for c in clusters)
        
        # Calculate cluster size statistics
        cluster_sizes = [c.frequency for c in clusters]
        total_clustered_failures = sum(cluster_sizes)
        
        return {
            'total_failures': len(failed_cases),
            'clustered_failures': total_clustered_failures,
            'unclustered_failures': len(failed_cases) - total_clustered_failures,
            'cluster_types': dict(cluster_type_counts),
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'average_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
            'cluster_count_by_type': {
                'missing_pattern': len([c for c in clusters if c.cluster_type == 'missing_pattern']),
                'extractor_failure': len([c for c in clusters if c.cluster_type == 'extractor_failure']),
                'bucket_mismatch': len([c for c in clusters if c.cluster_type == 'bucket_mismatch'])
            }
        }
    
    def identify_priority_fixes(self, clustering_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify top priority fixes based on clustering analysis
        
        Args:
            clustering_results: Results from cluster_failures
            
        Returns:
            List of priority fix recommendations
        """
        top_clusters = clustering_results.get('top_clusters', [])
        priority_fixes = []
        
        for i, cluster in enumerate(top_clusters):
            cluster_type = cluster.get('cluster_type')
            pattern = cluster.get('pattern')
            frequency = cluster.get('frequency', 0)
            
            fix_recommendation = {
                'priority_rank': i + 1,
                'cluster_id': cluster.get('cluster_id'),
                'cluster_type': cluster_type,
                'pattern': pattern,
                'frequency': frequency,
                'impact_score': self._calculate_impact_score(cluster),
                'recommended_action': self._get_recommended_action(cluster_type, pattern),
                'patch_generation_data': self._prepare_patch_data(cluster)
            }
            
            priority_fixes.append(fix_recommendation)
        
        return priority_fixes
    
    def _calculate_impact_score(self, cluster: Dict[str, Any]) -> float:
        """
        Calculate impact score for a cluster
        
        Args:
            cluster: Cluster summary data
            
        Returns:
            float: Impact score (higher = more impactful)
        """
        frequency = cluster.get('frequency', 0)
        affected_cases = cluster.get('affected_case_count', 0)
        cluster_type = cluster.get('cluster_type', '')
        
        # Base score from frequency
        score = frequency * 10
        
        # Bonus for affecting many different cases
        score += affected_cases * 5
        
        # Type-specific bonuses
        if cluster_type == 'missing_pattern':
            score += 20  # Missing patterns are high priority
        elif cluster_type == 'bucket_mismatch':
            score += 15  # Bucket mismatches affect classification
        elif cluster_type == 'extractor_failure':
            score += 10  # Extractor failures need targeted fixes
        
        return score
    
    def _get_recommended_action(self, cluster_type: str, pattern: str) -> str:
        """
        Get recommended action for a cluster type and pattern
        
        Args:
            cluster_type: Type of cluster
            pattern: Pattern associated with cluster
            
        Returns:
            str: Recommended action
        """
        if cluster_type == 'missing_pattern':
            return f"Add or modify regex rule to capture pattern: {pattern}"
        elif cluster_type == 'bucket_mismatch':
            return f"Review classification logic for pattern: {pattern}"
        elif cluster_type == 'extractor_failure':
            return f"Improve extractor for type: {pattern}"
        else:
            return f"Investigate cluster type: {cluster_type}"
    
    def _prepare_patch_data(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data needed for automated patch generation
        
        Args:
            cluster: Cluster summary data
            
        Returns:
            Dict with patch generation data
        """
        return {
            'cluster_type': cluster.get('cluster_type'),
            'pattern': cluster.get('pattern'),
            'frequency': cluster.get('frequency'),
            'sample_failure': cluster.get('representative_failure', {}),
            'extractor_ids': cluster.get('extractor_ids', []),
            'suggested_regex_modifications': self._suggest_regex_modifications(cluster)
        }
    
    def _suggest_regex_modifications(self, cluster: Dict[str, Any]) -> List[str]:
        """
        Suggest specific regex modifications based on cluster analysis
        
        Args:
            cluster: Cluster summary data
            
        Returns:
            List of suggested modifications
        """
        suggestions = []
        cluster_type = cluster.get('cluster_type')
        pattern = cluster.get('pattern', '')
        
        if cluster_type == 'missing_pattern':
            # Analyze the missing pattern to suggest regex improvements
            if pattern.startswith('course|'):
                suggestions.append("Consider adding more flexible bearing format patterns")
                suggestions.append("Check for alternative degree/minute/second notations")
            elif pattern.startswith('ref_segment|'):
                suggestions.append("Add patterns for reference line variations")
                suggestions.append("Include more directional terms in regex")
            
        elif cluster_type == 'bucket_mismatch':
            if 'explicit_bearings' in pattern:
                suggestions.append("Review criteria for explicit bearing classification")
            elif 'abstract_bearings' in pattern:
                suggestions.append("Improve abstract bearing detection patterns")
        
        return suggestions


def cluster_harness_failures(harness_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to cluster failures from harness results
    
    Args:
        harness_results: Results from testing harness
        
    Returns:
        Dict with clustering analysis and priority fixes
    """
    engine = FailureClusteringEngine()
    clustering_results = engine.cluster_failures(harness_results)
    
    # Add priority fix recommendations
    clustering_results['priority_fixes'] = engine.identify_priority_fixes(clustering_results)
    
    return clustering_results


def analyze_failure_trends(historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze failure trends across multiple harness runs
    
    Args:
        historical_results: List of historical harness results
        
    Returns:
        Dict with trend analysis
    """
    if not historical_results:
        return {'error': 'No historical results provided'}
    
    # Track patterns over time
    pattern_trends = defaultdict(list)
    
    for i, results in enumerate(historical_results):
        clustering_results = cluster_harness_failures(results)
        
        for cluster in clustering_results.get('clusters', []):
            pattern = cluster.get('pattern')
            frequency = cluster.get('frequency', 0)
            
            pattern_trends[pattern].append({
                'run_index': i,
                'frequency': frequency,
                'run_date': results.get('run_at')
            })
    
    # Identify trending patterns
    trending_up = []
    trending_down = []
    
    for pattern, trend_data in pattern_trends.items():
        if len(trend_data) >= 2:
            recent_freq = trend_data[-1]['frequency']
            older_freq = trend_data[-2]['frequency']
            
            if recent_freq > older_freq:
                trending_up.append({
                    'pattern': pattern,
                    'frequency_change': recent_freq - older_freq,
                    'trend': 'increasing'
                })
            elif recent_freq < older_freq:
                trending_down.append({
                    'pattern': pattern,
                    'frequency_change': older_freq - recent_freq,
                    'trend': 'decreasing'
                })
    
    return {
        'total_runs_analyzed': len(historical_results),
        'unique_patterns_tracked': len(pattern_trends),
        'trending_up': sorted(trending_up, key=lambda x: x['frequency_change'], reverse=True),
        'trending_down': sorted(trending_down, key=lambda x: x['frequency_change'], reverse=True),
        'pattern_trends': dict(pattern_trends)
    }