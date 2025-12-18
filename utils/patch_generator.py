"""
GPT Patch Generator for Legal Description Reader
Generates regex rule modifications based on failure analysis
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from utils.auth import get_openai_client
from utils.rules import RulesConfig, RuleEntry
from utils.failure_clustering import FailureClusteringEngine
from utils.patch_prompts import PatchPromptTemplates


@dataclass
class PatchOperation:
    """Represents a single patch operation"""
    operation: str  # add_pattern, modify_pattern, disable_pattern
    extractor_id: str
    new_regex: Optional[str] = None
    new_map: Optional[Dict[str, int]] = None
    reason: str = ""


@dataclass
class PatchSet:
    """Collection of patch operations"""
    patches: List[PatchOperation]
    target_failures: List[str]  # Fingerprints this patch set targets
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'patches': [
                {
                    'operation': p.operation,
                    'extractor_id': p.extractor_id,
                    'new_regex': p.new_regex,
                    'new_map': p.new_map,
                    'reason': p.reason
                }
                for p in self.patches
            ],
            'target_failures': self.target_failures,
            'confidence': self.confidence
        }


class GPTPatchGenerator:
    """
    GPT-powered patch generator for regex rules
    """
    
    def __init__(self):
        self.client = get_openai_client()
        self.clustering_engine = FailureClusteringEngine()
    
    def generate_patches(self, failure_analysis: Dict[str, Any], 
                        current_rules: RulesConfig) -> List[PatchSet]:
        """
        Generate patch sets based on failure analysis
        
        Args:
            failure_analysis: Output from failure clustering analysis
            current_rules: Current regex rules configuration
            
        Returns:
            List of patch sets targeting different failure patterns
        """
        patch_sets = []
        
        # Get top failure clusters
        clusters = failure_analysis.get('clusters', [])
        top_clusters = sorted(clusters, key=lambda x: x.get('frequency', 0), reverse=True)[:3]
        
        for cluster in top_clusters:
            try:
                patch_set = self._generate_patch_for_cluster(cluster, current_rules)
                if patch_set and patch_set.patches:
                    patch_sets.append(patch_set)
            except Exception as e:
                print(f"Error generating patch for cluster: {e}")
                continue
        
        return patch_sets
    
    def _generate_patch_for_cluster(self, cluster: Dict[str, Any], 
                                   current_rules: RulesConfig) -> Optional[PatchSet]:
        """
        Generate a patch set for a specific failure cluster
        
        Args:
            cluster: Failure cluster data
            current_rules: Current rules configuration
            
        Returns:
            PatchSet or None if generation fails
        """
        cluster_type = cluster.get('cluster_type')
        
        if cluster_type == 'missing_pattern':
            return self._generate_missing_pattern_patch(cluster, current_rules)
        elif cluster_type == 'extractor_failure':
            return self._generate_extractor_failure_patch(cluster, current_rules)
        elif cluster_type == 'bucket_mismatch':
            return self._generate_bucket_mismatch_patch(cluster, current_rules)
        else:
            return None
    
    def _generate_missing_pattern_patch(self, cluster: Dict[str, Any], 
                                      current_rules: RulesConfig) -> Optional[PatchSet]:
        """
        Generate patch for missing pattern failures
        
        Args:
            cluster: Missing pattern cluster data
            current_rules: Current rules configuration
            
        Returns:
            PatchSet for missing patterns
        """
        missing_fingerprint = cluster.get('pattern', '')
        sample_texts = cluster.get('sample_texts', [])
        frequency = cluster.get('frequency', 0)
        
        if not missing_fingerprint or not sample_texts:
            return None
        
        # Create prompt for GPT
        prompt = PatchPromptTemplates.missing_pattern_prompt(
            missing_fingerprint, sample_texts, current_rules
        )
        
        # Get GPT response
        gpt_response = self._call_gpt_for_patch(prompt)
        if not gpt_response:
            return None
        
        # Parse response into patch operations
        patches = self._parse_gpt_response(gpt_response)
        
        return PatchSet(
            patches=patches,
            target_failures=[missing_fingerprint],
            confidence=min(0.9, frequency / 10.0)  # Higher frequency = higher confidence
        )
    
    def _generate_extractor_failure_patch(self, cluster: Dict[str, Any], 
                                        current_rules: RulesConfig) -> Optional[PatchSet]:
        """
        Generate patch for extractor failure patterns
        
        Args:
            cluster: Extractor failure cluster data
            current_rules: Current rules configuration
            
        Returns:
            PatchSet for extractor failures
        """
        extractor_id = cluster.get('pattern', '')  # For extractor failures, pattern is the extractor_id
        sample_failures = cluster.get('sample_failures', [])
        frequency = cluster.get('frequency', 0)
        
        if not extractor_id or not sample_failures:
            return None
        
        # Find the current rule for this extractor
        current_rule = None
        for rule in current_rules.rules:
            if rule.extractor_id == extractor_id:
                current_rule = rule
                break
        
        if not current_rule:
            return None
        
        # Create prompt for GPT
        prompt = PatchPromptTemplates.extractor_failure_prompt(
            extractor_id, current_rule, sample_failures, current_rules
        )
        
        # Get GPT response
        gpt_response = self._call_gpt_for_patch(prompt)
        if not gpt_response:
            return None
        
        # Parse response into patch operations
        patches = self._parse_gpt_response(gpt_response)
        
        return PatchSet(
            patches=patches,
            target_failures=[f"extractor_failure:{extractor_id}"],
            confidence=min(0.8, frequency / 15.0)
        )
    
    def _generate_bucket_mismatch_patch(self, cluster: Dict[str, Any], 
                                      current_rules: RulesConfig) -> Optional[PatchSet]:
        """
        Generate patch for bucket classification mismatches
        
        Args:
            cluster: Bucket mismatch cluster data
            current_rules: Current rules configuration
            
        Returns:
            PatchSet for bucket mismatches
        """
        # Bucket mismatches are typically handled by classification model fine-tuning
        # rather than regex rule changes, so we return None for now
        return None
    

    
    def _call_gpt_for_patch(self, prompt: str) -> Optional[str]:
        """
        Call GPT API to generate patch recommendations
        
        Args:
            prompt: Formatted prompt for GPT
            
        Returns:
            GPT response string or None if failed
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": PatchPromptTemplates.system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent, focused responses
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling GPT for patch generation: {e}")
            return None
    
    def _parse_gpt_response(self, gpt_response: str) -> List[PatchOperation]:
        """
        Parse GPT response into patch operations
        
        Args:
            gpt_response: JSON response from GPT
            
        Returns:
            List of PatchOperation objects
        """
        try:
            # Clean up response - remove any markdown formatting
            cleaned_response = gpt_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            response_data = json.loads(cleaned_response)
            patches_data = response_data.get('patches', [])
            
            patches = []
            for patch_data in patches_data:
                operation = patch_data.get('operation')
                extractor_id = patch_data.get('extractor_id')
                
                if not operation or not extractor_id:
                    continue
                
                patch = PatchOperation(
                    operation=operation,
                    extractor_id=extractor_id,
                    new_regex=patch_data.get('new_regex'),
                    new_map=patch_data.get('new_map'),
                    reason=patch_data.get('reason', '')
                )
                
                # Validate the patch
                if self._validate_patch_operation(patch):
                    patches.append(patch)
            
            return patches
            
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT response as JSON: {e}")
            print(f"Response was: {gpt_response}")
            return []
        except Exception as e:
            print(f"Error processing GPT response: {e}")
            return []
    
    def _validate_patch_operation(self, patch: PatchOperation) -> bool:
        """
        Validate a patch operation for correctness
        
        Args:
            patch: PatchOperation to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check operation type
        valid_operations = ['add_pattern', 'modify_pattern', 'disable_pattern']
        if patch.operation not in valid_operations:
            return False
        
        # Check extractor_id
        if not patch.extractor_id or not isinstance(patch.extractor_id, str):
            return False
        
        # For add_pattern and modify_pattern, need regex
        if patch.operation in ['add_pattern', 'modify_pattern']:
            if not patch.new_regex:
                return False
            
            # Try to compile regex to check validity
            try:
                re.compile(patch.new_regex)
            except re.error:
                print(f"Invalid regex in patch: {patch.new_regex}")
                return False
        
        # For disable_pattern, don't need regex
        if patch.operation == 'disable_pattern':
            if patch.new_regex is not None:
                print(f"Disable operation should not have new_regex")
                return False
        
        return True
    
    def generate_patch_json(self, failure_analysis: Dict[str, Any], 
                           current_rules: RulesConfig) -> str:
        """
        Generate patches and return as JSON string
        
        Args:
            failure_analysis: Failure analysis data
            current_rules: Current rules configuration
            
        Returns:
            JSON string with patch recommendations
        """
        patch_sets = self.generate_patches(failure_analysis, current_rules)
        
        result = {
            'patch_sets': [ps.to_dict() for ps in patch_sets],
            'generation_metadata': {
                'total_patch_sets': len(patch_sets),
                'total_operations': sum(len(ps.patches) for ps in patch_sets),
                'target_clusters': len(failure_analysis.get('clusters', [])),
                'confidence_range': {
                    'min': min((ps.confidence for ps in patch_sets), default=0.0),
                    'max': max((ps.confidence for ps in patch_sets), default=0.0)
                }
            }
        }
        
        return json.dumps(result, indent=2)


def create_patch_generator() -> GPTPatchGenerator:
    """
    Factory function to create a patch generator instance
    
    Returns:
        GPTPatchGenerator instance
    """
    return GPTPatchGenerator()


def generate_patches_from_failures(failures_json: str, 
                                  current_rules: RulesConfig) -> str:
    """
    Convenience function to generate patches from failures JSON
    
    Args:
        failures_json: JSON string with failure analysis
        current_rules: Current rules configuration
        
    Returns:
        JSON string with patch recommendations
    """
    try:
        failure_data = json.loads(failures_json)
        generator = create_patch_generator()
        return generator.generate_patch_json(failure_data, current_rules)
    except Exception as e:
        error_result = {
            'error': f"Failed to generate patches: {str(e)}",
            'patch_sets': []
        }
        return json.dumps(error_result, indent=2)