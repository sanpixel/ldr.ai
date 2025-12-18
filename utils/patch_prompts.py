"""
Prompt templates for GPT patch generation
Contains structured prompts for different types of failure analysis
"""

import json
from typing import List, Dict, Any
from utils.rules import RuleEntry, RulesConfig


class PatchPromptTemplates:
    """
    Collection of prompt templates for patch generation
    """
    
    @staticmethod
    def missing_pattern_prompt(missing_fingerprint: str, 
                              sample_texts: List[str], 
                              current_rules: RulesConfig) -> str:
        """
        Create GPT prompt for missing pattern analysis
        
        Args:
            missing_fingerprint: The fingerprint pattern that should have been extracted
            sample_texts: Sample legal description texts that should match
            current_rules: Current regex rules
            
        Returns:
            Formatted prompt string
        """
        # Parse fingerprint to understand what type of pattern is missing
        fp_parts = missing_fingerprint.split('|')
        fp_type = fp_parts[0] if fp_parts else 'unknown'
        
        # Get relevant existing rules for this type
        relevant_rules = [rule for rule in current_rules.rules if rule.type == fp_type]
        
        return f"""You are a regex pattern expert helping to improve a legal description parsing system.

TASK: Create a new regex pattern to extract the missing information from legal description texts.

MISSING FINGERPRINT: {missing_fingerprint}
This fingerprint represents the expected extraction that is currently failing.

FINGERPRINT FORMAT REFERENCE:
- course: course|N|45|12|30|E|125.00 (cardinal_ns|degrees|minutes|seconds|cardinal_ew|distance)
- curve: curve|N|45|12|30|E|125.00 (same as course)  
- ref_segment: ref_segment|reference|monument|directional
- ref_curve: ref_curve|reference|monument|directional

SAMPLE TEXTS THAT SHOULD MATCH:
{json.dumps(sample_texts, indent=2)}

EXISTING RULES FOR TYPE '{fp_type}':
{json.dumps([rule.to_dict() for rule in relevant_rules], indent=2)}

ANALYSIS GUIDELINES:
1. Study the sample texts to identify the common pattern that should be extracted
2. Compare with existing rules to avoid conflicts and duplication
3. Focus on the specific format variations that aren't currently handled
4. Consider optional elements, whitespace variations, and alternative spellings

REGEX BEST PRACTICES:
- Use non-capturing groups (?:) for grouping without capturing
- Use word boundaries \\b where appropriate
- Handle optional whitespace with \\s*
- Use character classes [NS] instead of (N|S) for single characters
- Make patterns as specific as possible to avoid false matches
- Consider case insensitivity with (?i) if needed

REQUIREMENTS:
1. Create a new regex pattern that will extract the missing fingerprint from the sample texts
2. Provide the capture group mapping that corresponds to the fingerprint format
3. Ensure the pattern doesn't conflict with existing patterns
4. Make the pattern as specific as possible to avoid false matches
5. Test mentally against the sample texts to ensure it would match

OUTPUT FORMAT (JSON only, no explanations):
{{
  "patches": [
    {{
      "operation": "add_pattern",
      "extractor_id": "descriptive_pattern_name",
      "new_regex": "your_regex_pattern_here",
      "new_map": {{
        "field_name": capture_group_number
      }},
      "reason": "Brief explanation of what this pattern captures"
    }}
  ]
}}"""
    
    @staticmethod
    def extractor_failure_prompt(extractor_id: str, 
                                current_rule: RuleEntry,
                                sample_failures: List[Dict[str, Any]], 
                                current_rules: RulesConfig) -> str:
        """
        Create GPT prompt for extractor failure analysis
        
        Args:
            extractor_id: ID of the failing extractor
            current_rule: The current rule that's failing
            sample_failures: Sample failure cases
            current_rules: All current rules
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a regex pattern expert helping to fix a failing legal description parsing rule.

TASK: Modify or replace the failing regex pattern to handle the sample texts correctly.

FAILING EXTRACTOR: {extractor_id}
CURRENT RULE:
{json.dumps(current_rule.to_dict(), indent=2)}

SAMPLE FAILURE CASES:
{json.dumps(sample_failures, indent=2)}

CONTEXT - OTHER RULES OF SAME TYPE:
{json.dumps([rule.to_dict() for rule in current_rules.rules if rule.type == current_rule.type and rule.extractor_id != extractor_id], indent=2)}

COMMON FAILURE PATTERNS TO CHECK:
1. Too restrictive character classes - e.g., [0-9] instead of \\d, missing decimal points
2. Missing optional elements - whitespace, punctuation, alternative formats
3. Incorrect whitespace handling - \\s+ vs \\s*, missing \\s* between elements
4. Case sensitivity issues - missing (?i) flag or wrong case assumptions
5. Missing alternative formats - "degrees" vs "°", "minutes" vs "'", "feet" vs "ft"
6. Boundary issues - missing word boundaries \\b or anchor points
7. Greedy vs non-greedy matching - .* vs .*?
8. Escape character issues - missing escapes for special characters

ANALYSIS APPROACH:
1. Compare the failing texts with the current regex pattern
2. Identify what specific elements are causing the mismatch
3. Determine if the fix should be a modification or a complete replacement
4. Ensure the fix doesn't break existing successful matches
5. Test the new pattern mentally against both failing and successful cases

REQUIREMENTS:
1. Fix the regex to handle the failing cases
2. Maintain compatibility with existing successful matches (if possible)
3. Ensure capture groups still map correctly to the expected fields
4. Consider if the pattern should be modified or if a new pattern should be added
5. Provide a clear reason for the change

OUTPUT FORMAT (JSON only, no explanations):

For pattern modification:
{{
  "patches": [
    {{
      "operation": "modify_pattern",
      "extractor_id": "{extractor_id}",
      "new_regex": "improved_regex_pattern_here",
      "new_map": {{
        "field_name": capture_group_number
      }},
      "reason": "Brief explanation of the fix"
    }}
  ]
}}

For pattern replacement (if modification would break too many existing matches):
{{
  "patches": [
    {{
      "operation": "disable_pattern",
      "extractor_id": "{extractor_id}",
      "reason": "Pattern too problematic, needs replacement"
    }},
    {{
      "operation": "add_pattern", 
      "extractor_id": "new_pattern_name",
      "new_regex": "replacement_regex_pattern",
      "new_map": {{
        "field_name": capture_group_number
      }},
      "reason": "Replacement for disabled pattern"
    }}
  ]
}}"""
    
    @staticmethod
    def bucket_mismatch_prompt(cluster: Dict[str, Any], 
                              current_rules: RulesConfig) -> str:
        """
        Create GPT prompt for bucket classification mismatch analysis
        
        Args:
            cluster: Bucket mismatch cluster data
            current_rules: Current rules configuration
            
        Returns:
            Formatted prompt string
        """
        mismatch_pattern = cluster.get('pattern', '')
        sample_cases = cluster.get('sample_failures', [])
        frequency = cluster.get('frequency', 0)
        
        return f"""You are an expert in legal description classification helping to fix bucket classification errors.

TASK: Analyze bucket classification mismatches and recommend rule adjustments.

MISMATCH PATTERN: {mismatch_pattern}
This represents the classification error pattern (e.g., "explicit_bearings->abstract_bearings")

SAMPLE MISCLASSIFIED CASES:
{json.dumps(sample_cases, indent=2)}

FREQUENCY: {frequency} occurrences

CURRENT EXTRACTION RULES:
{json.dumps([rule.to_dict() for rule in current_rules.rules], indent=2)}

CLASSIFICATION LOGIC:
- explicit_bearings: Contains numeric course/curve lines with degrees, minutes, seconds, distances
- abstract_bearings: Contains reference lines or incomplete numeric information
- no_bearings: Contains no extractable geometric information

ANALYSIS APPROACH:
1. Examine why the current rules are producing the wrong bucket classification
2. Determine if the issue is:
   - Missing extraction patterns (causing explicit to be classified as abstract)
   - Over-extraction (causing abstract to be classified as explicit)
   - Incorrect extraction patterns (extracting wrong information)
3. Recommend specific rule changes to fix the classification

COMMON ISSUES:
- Patterns extracting reference text as numeric values
- Missing patterns for alternative numeric formats
- Patterns too broad and matching non-geometric text
- Patterns too narrow and missing valid geometric data

REQUIREMENTS:
1. Identify the root cause of the bucket mismatch
2. Recommend specific regex rule changes
3. Ensure changes improve classification without breaking other cases
4. Focus on extraction accuracy rather than classification model changes

OUTPUT FORMAT (JSON only, no explanations):
{{
  "patches": [
    {{
      "operation": "modify_pattern|add_pattern|disable_pattern",
      "extractor_id": "pattern_id",
      "new_regex": "regex_if_applicable",
      "new_map": {{
        "field_name": capture_group_number
      }},
      "reason": "How this fixes the bucket classification issue"
    }}
  ]
}}

Note: If the issue is primarily with the classification model rather than extraction rules, return empty patches array."""
    
    @staticmethod
    def system_prompt() -> str:
        """
        System prompt for GPT patch generation
        
        Returns:
            System prompt string
        """
        return """You are a regex expert specializing in legal description parsing. Your role is to analyze extraction failures and generate precise regex pattern fixes.

CORE PRINCIPLES:
1. Always respond with valid JSON only, no additional text or explanations
2. Focus on minimal, targeted changes that fix specific issues
3. Preserve existing functionality while addressing failures
4. Use proper regex syntax and best practices
5. Ensure capture groups map correctly to expected output fields

REGEX EXPERTISE:
- Legal descriptions contain bearings (N45°30'15"E), distances (125.50 feet), and references (along Duncan Drive)
- Common variations: degrees/minutes/seconds formats, spelled-out directions, various distance units
- Pattern specificity is crucial to avoid false matches in legal text
- Capture groups must align with fingerprint format requirements

OUTPUT REQUIREMENTS:
- JSON format only
- Valid regex patterns that compile successfully
- Correct capture group mappings
- Clear, concise reasons for changes
- Operations: add_pattern, modify_pattern, disable_pattern only"""


def get_prompt_for_cluster_type(cluster_type: str) -> str:
    """
    Get the appropriate prompt template for a cluster type
    
    Args:
        cluster_type: Type of failure cluster
        
    Returns:
        Method name for the appropriate prompt template
    """
    prompt_map = {
        'missing_pattern': 'missing_pattern_prompt',
        'extractor_failure': 'extractor_failure_prompt', 
        'bucket_mismatch': 'bucket_mismatch_prompt'
    }
    
    return prompt_map.get(cluster_type, 'missing_pattern_prompt')


def validate_prompt_inputs(cluster: Dict[str, Any], current_rules: RulesConfig) -> List[str]:
    """
    Validate inputs for prompt generation
    
    Args:
        cluster: Failure cluster data
        current_rules: Current rules configuration
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    if not cluster:
        errors.append("Cluster data is empty")
        return errors
    
    cluster_type = cluster.get('cluster_type')
    if not cluster_type:
        errors.append("Cluster type is missing")
    
    pattern = cluster.get('pattern')
    if not pattern:
        errors.append("Cluster pattern is missing")
    
    frequency = cluster.get('frequency', 0)
    if frequency <= 0:
        errors.append("Cluster frequency must be positive")
    
    if cluster_type == 'missing_pattern':
        sample_texts = cluster.get('sample_texts', [])
        if not sample_texts:
            errors.append("Missing pattern cluster requires sample texts")
    
    elif cluster_type == 'extractor_failure':
        sample_failures = cluster.get('sample_failures', [])
        if not sample_failures:
            errors.append("Extractor failure cluster requires sample failures")
    
    if not current_rules or not current_rules.rules:
        errors.append("Current rules configuration is empty")
    
    return errors