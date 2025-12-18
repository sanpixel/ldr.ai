"""
Patch Application System for Legal Description Reader
Applies GPT-generated patches to regex rules with validation and versioning
"""

import json
import re
import copy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from utils.rules import RulesConfig, RuleEntry
from utils.gcs_rules import GCSRulesManager
from utils.patch_generator import PatchOperation, PatchSet


@dataclass
class PatchApplicationResult:
    """Result of applying a patch set"""
    success: bool
    new_rules: Optional[RulesConfig] = None
    new_version_id: Optional[str] = None
    validation_errors: List[str] = None
    applied_operations: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.applied_operations is None:
            self.applied_operations = []


class PatchApplicationEngine:
    """
    Engine for applying patches to regex rules with validation and versioning
    """
    
    def __init__(self):
        self.gcs_manager = GCSRulesManager()
    
    def apply_patch_set(self, patch_set: PatchSet, 
                       current_rules: RulesConfig) -> PatchApplicationResult:
        """
        Apply a complete patch set to the current rules
        
        Args:
            patch_set: PatchSet containing operations to apply
            current_rules: Current rules configuration
            
        Returns:
            PatchApplicationResult with success status and new rules
        """
        # Create a copy of current rules to modify
        new_rules = copy.deepcopy(current_rules)
        applied_operations = []
        validation_errors = []
        
        # Apply each patch operation
        for patch in patch_set.patches:
            try:
                operation_result = self._apply_single_patch(patch, new_rules)
                if operation_result['success']:
                    applied_operations.append(operation_result['description'])
                else:
                    validation_errors.extend(operation_result['errors'])
            except Exception as e:
                validation_errors.append(f"Error applying patch {patch.extractor_id}: {str(e)}")
        
        # If any operations failed, return failure
        if validation_errors:
            return PatchApplicationResult(
                success=False,
                validation_errors=validation_errors,
                applied_operations=applied_operations
            )
        
        # Validate the complete new rules configuration
        final_validation = self._validate_complete_rules(new_rules)
        if not final_validation['valid']:
            return PatchApplicationResult(
                success=False,
                validation_errors=final_validation['errors'],
                applied_operations=applied_operations
            )
        
        # Save new version to GCS
        try:
            version_id = self.gcs_manager.save_rules_version(new_rules)
            
            return PatchApplicationResult(
                success=True,
                new_rules=new_rules,
                new_version_id=version_id,
                applied_operations=applied_operations
            )
            
        except Exception as e:
            return PatchApplicationResult(
                success=False,
                validation_errors=[f"Failed to save new version: {str(e)}"],
                applied_operations=applied_operations
            )
    
    def _apply_single_patch(self, patch: PatchOperation, 
                           rules: RulesConfig) -> Dict[str, Any]:
        """
        Apply a single patch operation to the rules
        
        Args:
            patch: PatchOperation to apply
            rules: RulesConfig to modify in place
            
        Returns:
            Dict with success status and details
        """
        if patch.operation == 'add_pattern':
            return self._apply_add_pattern(patch, rules)
        elif patch.operation == 'modify_pattern':
            return self._apply_modify_pattern(patch, rules)
        elif patch.operation == 'disable_pattern':
            return self._apply_disable_pattern(patch, rules)
        else:
            return {
                'success': False,
                'errors': [f"Unknown patch operation: {patch.operation}"]
            }
    
    def _apply_add_pattern(self, patch: PatchOperation, 
                          rules: RulesConfig) -> Dict[str, Any]:
        """
        Apply an add_pattern operation
        
        Args:
            patch: Add pattern operation
            rules: Rules to modify
            
        Returns:
            Dict with operation result
        """
        # Check if extractor_id already exists
        existing_ids = [rule.extractor_id for rule in rules.rules]
        if patch.extractor_id in existing_ids:
            return {
                'success': False,
                'errors': [f"Extractor ID '{patch.extractor_id}' already exists"]
            }
        
        # Validate regex
        validation = self._validate_regex_pattern(patch.new_regex)
        if not validation['valid']:
            return {
                'success': False,
                'errors': validation['errors']
            }
        
        # Determine rule type from map fields or use 'course' as default
        rule_type = self._infer_rule_type(patch.new_map)
        
        # Create new rule
        new_rule = RuleEntry(
            extractor_id=patch.extractor_id,
            type=rule_type,
            regex=patch.new_regex,
            map=patch.new_map or {}
        )
        
        # Add to rules
        rules.rules.append(new_rule)
        
        return {
            'success': True,
            'description': f"Added pattern '{patch.extractor_id}' for type '{rule_type}'"
        }
    
    def _apply_modify_pattern(self, patch: PatchOperation, 
                             rules: RulesConfig) -> Dict[str, Any]:
        """
        Apply a modify_pattern operation
        
        Args:
            patch: Modify pattern operation
            rules: Rules to modify
            
        Returns:
            Dict with operation result
        """
        # Find existing rule
        target_rule = None
        for rule in rules.rules:
            if rule.extractor_id == patch.extractor_id:
                target_rule = rule
                break
        
        if not target_rule:
            return {
                'success': False,
                'errors': [f"Extractor ID '{patch.extractor_id}' not found"]
            }
        
        # Validate new regex
        validation = self._validate_regex_pattern(patch.new_regex)
        if not validation['valid']:
            return {
                'success': False,
                'errors': validation['errors']
            }
        
        # Update rule
        old_regex = target_rule.regex
        target_rule.regex = patch.new_regex
        
        if patch.new_map:
            target_rule.map = patch.new_map
        
        return {
            'success': True,
            'description': f"Modified pattern '{patch.extractor_id}' (was: {old_regex[:50]}...)"
        }
    
    def _apply_disable_pattern(self, patch: PatchOperation, 
                              rules: RulesConfig) -> Dict[str, Any]:
        """
        Apply a disable_pattern operation
        
        Args:
            patch: Disable pattern operation
            rules: Rules to modify
            
        Returns:
            Dict with operation result
        """
        # Find and remove existing rule
        original_count = len(rules.rules)
        rules.rules = [rule for rule in rules.rules if rule.extractor_id != patch.extractor_id]
        
        if len(rules.rules) == original_count:
            return {
                'success': False,
                'errors': [f"Extractor ID '{patch.extractor_id}' not found"]
            }
        
        return {
            'success': True,
            'description': f"Disabled pattern '{patch.extractor_id}'"
        }
    
    def _validate_regex_pattern(self, regex_pattern: str) -> Dict[str, Any]:
        """
        Validate a regex pattern for compilation and basic correctness
        
        Args:
            regex_pattern: Regex string to validate
            
        Returns:
            Dict with validation result
        """
        errors = []
        
        if not regex_pattern:
            errors.append("Regex pattern is empty")
            return {'valid': False, 'errors': errors}
        
        # Test regex compilation
        try:
            compiled_regex = re.compile(regex_pattern)
        except re.error as e:
            errors.append(f"Regex compilation error: {str(e)}")
            return {'valid': False, 'errors': errors}
        
        # Check for common issues
        if len(regex_pattern) > 1000:
            errors.append("Regex pattern is too long (>1000 characters)")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'\.\*\.\*',  # Multiple .* can cause catastrophic backtracking
            r'\+\*',      # Invalid quantifier combination
            r'\*\+',      # Invalid quantifier combination
        ]
        
        for dangerous in dangerous_patterns:
            if re.search(dangerous, regex_pattern):
                errors.append(f"Potentially dangerous pattern detected: {dangerous}")
        
        # Test with sample text to ensure it doesn't hang
        try:
            test_text = "N45°30'15\"E 125.50 feet along Duncan Drive northerly"
            compiled_regex.search(test_text)
        except Exception as e:
            errors.append(f"Regex execution error on test text: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_complete_rules(self, rules: RulesConfig) -> Dict[str, Any]:
        """
        Validate the complete rules configuration
        
        Args:
            rules: Complete rules configuration to validate
            
        Returns:
            Dict with validation result
        """
        errors = []
        
        if not rules.rules:
            errors.append("Rules configuration is empty")
            return {'valid': False, 'errors': errors}
        
        # Check for duplicate extractor IDs
        extractor_ids = [rule.extractor_id for rule in rules.rules]
        duplicates = set([x for x in extractor_ids if extractor_ids.count(x) > 1])
        if duplicates:
            errors.append(f"Duplicate extractor IDs found: {list(duplicates)}")
        
        # Validate each rule
        for rule in rules.rules:
            rule_validation = self._validate_single_rule(rule)
            if not rule_validation['valid']:
                errors.extend([f"Rule {rule.extractor_id}: {err}" for err in rule_validation['errors']])
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_single_rule(self, rule: RuleEntry) -> Dict[str, Any]:
        """
        Validate a single rule entry
        
        Args:
            rule: RuleEntry to validate
            
        Returns:
            Dict with validation result
        """
        errors = []
        
        # Check required fields
        if not rule.extractor_id:
            errors.append("Missing extractor_id")
        
        if not rule.type:
            errors.append("Missing type")
        elif rule.type not in ['course', 'curve', 'ref_segment', 'ref_curve']:
            errors.append(f"Invalid type: {rule.type}")
        
        if not rule.regex:
            errors.append("Missing regex")
        else:
            # Validate regex
            regex_validation = self._validate_regex_pattern(rule.regex)
            if not regex_validation['valid']:
                errors.extend(regex_validation['errors'])
        
        # Validate map
        if not isinstance(rule.map, dict):
            errors.append("Map must be a dictionary")
        else:
            # Check that map values are positive integers
            for field, group_num in rule.map.items():
                if not isinstance(group_num, int) or group_num < 1:
                    errors.append(f"Map field '{field}' must be a positive integer, got {group_num}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _infer_rule_type(self, field_map: Dict[str, int]) -> str:
        """
        Infer rule type from field mapping
        
        Args:
            field_map: Mapping of field names to capture groups
            
        Returns:
            Inferred rule type
        """
        if not field_map:
            return 'course'  # Default
        
        fields = set(field_map.keys())
        
        # Check for course/curve fields
        bearing_fields = {'cardinal_ns', 'degrees', 'minutes', 'seconds', 'cardinal_ew', 'distance'}
        if bearing_fields.intersection(fields):
            return 'course'  # Default to course, can be changed to curve manually if needed
        
        # Check for reference fields
        reference_fields = {'reference', 'monument'}
        if reference_fields.intersection(fields):
            return 'ref_segment'  # Default to segment, can be changed to curve manually if needed
        
        return 'course'  # Default fallback
    
    def deploy_new_version(self, version_id: str) -> Dict[str, Any]:
        """
        Deploy a new rules version as current
        
        Args:
            version_id: Version ID to deploy
            
        Returns:
            Dict with deployment result
        """
        try:
            success = self.gcs_manager.deploy_version_as_current(version_id)
            
            if success:
                return {
                    'success': True,
                    'message': f"Version {version_id} deployed as current"
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to deploy version {version_id}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error deploying version {version_id}: {str(e)}"
            }
    
    def rollback_to_version(self, version_id: str) -> Dict[str, Any]:
        """
        Rollback to a previous rules version
        
        Args:
            version_id: Version ID to rollback to
            
        Returns:
            Dict with rollback result
        """
        try:
            success = self.gcs_manager.rollback_to_version(version_id)
            
            if success:
                return {
                    'success': True,
                    'message': f"Rolled back to version {version_id}"
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to rollback to version {version_id}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error rolling back to version {version_id}: {str(e)}"
            }


class PatchValidator:
    """
    Standalone validator for patch operations
    """
    
    @staticmethod
    def validate_patch_set(patch_set: PatchSet) -> Dict[str, Any]:
        """
        Validate a patch set before application
        
        Args:
            patch_set: PatchSet to validate
            
        Returns:
            Dict with validation result
        """
        errors = []
        
        if not patch_set.patches:
            errors.append("Patch set is empty")
            return {'valid': False, 'errors': errors}
        
        # Validate each patch
        for i, patch in enumerate(patch_set.patches):
            patch_errors = PatchValidator.validate_patch_operation(patch)
            if patch_errors:
                errors.extend([f"Patch {i+1}: {err}" for err in patch_errors])
        
        # Check for conflicting operations
        extractor_ids = [p.extractor_id for p in patch_set.patches]
        duplicates = set([x for x in extractor_ids if extractor_ids.count(x) > 1])
        if duplicates:
            errors.append(f"Multiple operations on same extractor IDs: {list(duplicates)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    @staticmethod
    def validate_patch_operation(patch: PatchOperation) -> List[str]:
        """
        Validate a single patch operation
        
        Args:
            patch: PatchOperation to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check operation type
        valid_operations = ['add_pattern', 'modify_pattern', 'disable_pattern']
        if patch.operation not in valid_operations:
            errors.append(f"Invalid operation: {patch.operation}")
        
        # Check extractor_id
        if not patch.extractor_id or not isinstance(patch.extractor_id, str):
            errors.append("Invalid or missing extractor_id")
        
        # Operation-specific validation
        if patch.operation in ['add_pattern', 'modify_pattern']:
            if not patch.new_regex:
                errors.append(f"{patch.operation} requires new_regex")
            else:
                # Test regex compilation
                try:
                    re.compile(patch.new_regex)
                except re.error as e:
                    errors.append(f"Invalid regex: {str(e)}")
        
        if patch.operation == 'disable_pattern':
            if patch.new_regex is not None:
                errors.append("disable_pattern should not have new_regex")
            if patch.new_map is not None:
                errors.append("disable_pattern should not have new_map")
        
        return errors


def apply_patches_from_json(patches_json: str, 
                           current_rules: RulesConfig) -> List[PatchApplicationResult]:
    """
    Convenience function to apply patches from JSON string
    
    Args:
        patches_json: JSON string containing patch sets
        current_rules: Current rules configuration
        
    Returns:
        List of PatchApplicationResult objects
    """
    try:
        patches_data = json.loads(patches_json)
        patch_sets_data = patches_data.get('patch_sets', [])
        
        engine = PatchApplicationEngine()
        results = []
        
        for patch_set_data in patch_sets_data:
            # Convert to PatchSet object
            patches = []
            for patch_data in patch_set_data.get('patches', []):
                patch = PatchOperation(
                    operation=patch_data.get('operation'),
                    extractor_id=patch_data.get('extractor_id'),
                    new_regex=patch_data.get('new_regex'),
                    new_map=patch_data.get('new_map'),
                    reason=patch_data.get('reason', '')
                )
                patches.append(patch)
            
            patch_set = PatchSet(
                patches=patches,
                target_failures=patch_set_data.get('target_failures', []),
                confidence=patch_set_data.get('confidence', 0.0)
            )
            
            # Apply patch set
            result = engine.apply_patch_set(patch_set, current_rules)
            results.append(result)
            
            # If successful, use new rules for next patch set
            if result.success and result.new_rules:
                current_rules = result.new_rules
        
        return results
        
    except Exception as e:
        return [PatchApplicationResult(
            success=False,
            validation_errors=[f"Error processing patches JSON: {str(e)}"]
        )]