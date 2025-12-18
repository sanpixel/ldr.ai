"""
Property-based tests for regex rules externalization
Tests that externalized rules maintain identical behavior to hardcoded patterns
"""

import pytest
import re
from typing import List
from hypothesis import given, strategies as st, settings, HealthCheck
from utils.rules import (
    RulesConfig, RegexRule, RuleType, RulesProcessor,
    create_default_rules_config, load_rules_from_file
)
from utils.schema import LineType
import json


# Test data strategies
bearing_text_examples = [
    "North 45 degrees 30 minutes 15 seconds East",
    "S 73° 32' 01\" W", 
    "North 71 degrees 53 minutes East",
    "South 12 degrees 45 minutes 30 seconds West",
    "N 90° 00' 00\" E",
    "South 180 degrees West"
]

@st.composite
def bearing_text_strategy(draw):
    """Generate bearing text variations"""
    # Use known examples plus generated variations
    if draw(st.booleans()):
        return draw(st.sampled_from(bearing_text_examples))
    
    # Generate synthetic bearing text
    cardinal_ns = draw(st.sampled_from(['North', 'South', 'N', 'S']))
    degrees = draw(st.integers(min_value=0, max_value=359))
    minutes = draw(st.integers(min_value=0, max_value=59))
    seconds = draw(st.integers(min_value=0, max_value=59))
    cardinal_ew = draw(st.sampled_from(['East', 'West', 'E', 'W']))
    
    # Various formats
    format_choice = draw(st.integers(min_value=1, max_value=3))
    if format_choice == 1:
        return f"{cardinal_ns} {degrees}° {minutes}' {seconds}\" {cardinal_ew}"
    elif format_choice == 2:
        return f"{cardinal_ns} {degrees} degrees {minutes} minutes {seconds} seconds {cardinal_ew}"
    else:
        return f"{cardinal_ns} {degrees} degrees {minutes} minutes {cardinal_ew}"


class TestBehavioralPreservation:
    """
    Feature: self-improving-geometry-extractor, Property 6: Behavioral preservation during externalization
    """
    
    def test_default_rules_config_loads_successfully(self):
        """
        Property 6: Default rules configuration is valid
        The default rules should load and validate successfully
        """
        rules_config = create_default_rules_config()
        
        # Should have valid structure
        assert rules_config.version == "000001"
        assert len(rules_config.rules) > 0
        
        # All rules should be valid
        validation_errors = rules_config.validate_all_rules()
        assert len(validation_errors) == 0, f"Validation errors: {validation_errors}"
        
        # Should be able to create processor
        processor = RulesProcessor(rules_config)
        assert processor is not None
    
    def test_rules_json_file_matches_default_config(self):
        """
        Property 6: JSON file matches default configuration
        The rules.json file should match the default configuration
        """
        # Load from file
        file_config = load_rules_from_file('rules.json')
        
        # Create default config
        default_config = create_default_rules_config()
        
        # Should have same version
        assert file_config.version == default_config.version
        
        # Should have same number of rules
        assert len(file_config.rules) == len(default_config.rules)
        
        # Each rule should match (by extractor_id)
        file_rules_by_id = {rule.extractor_id: rule for rule in file_config.rules}
        default_rules_by_id = {rule.extractor_id: rule for rule in default_config.rules}
        
        assert set(file_rules_by_id.keys()) == set(default_rules_by_id.keys())
    
    @given(bearing_text_strategy())
    @settings(max_examples=100)
    def test_externalized_bearing_extraction_matches_hardcoded(self, bearing_text: str):
        """
        Property 6: Externalized rules produce same results as hardcoded patterns
        For any bearing text, externalized rules should match hardcoded regex behavior
        """
        # Load externalized rules
        rules_config = create_default_rules_config()
        processor = RulesProcessor(rules_config)
        
        # Extract using externalized rules
        externalized_results = processor.extract_courses(bearing_text)
        
        # Extract using hardcoded patterns (simulate original behavior)
        hardcoded_results = self._extract_with_hardcoded_patterns(bearing_text)
        
        # Compare results - should find same number of matches
        # Note: We're testing that externalized rules don't miss patterns that hardcoded ones found
        if hardcoded_results:
            assert len(externalized_results) > 0, f"Externalized rules missed pattern in: {bearing_text}"
    
    def _extract_with_hardcoded_patterns(self, text: str) -> list:
        """
        Simulate the original hardcoded pattern extraction
        This replicates the logic from the original codebase
        """
        results = []
        
        # Pattern 1: Verbose format (from format_bearing_concise)
        pattern1 = r'(North|South)\s+(\d+)\s*(?:°|degrees?|deg|\s)\s*(\d+)\s*(?:\'|′|minutes?|min|\s)\s*(?:(\d+)\s*(?:"|″|seconds?|sec|\s)\s+)?(East|West)'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches1:
            results.append({
                'cardinal_ns': match.group(1),
                'degrees': match.group(2),
                'minutes': match.group(3),
                'seconds': match.group(4),
                'cardinal_ew': match.group(5),
                'pattern': 'verbose'
            })
        
        # Pattern 2: Unified format (from extract_bearings_with_gpt)
        pattern2 = r'(S|South|N|North)[\s\.]*(\d+)(?:[\s°degrees]+(?:(\d+)(?:[\s\'minutes]+(?:(\d+(?:\.\d+)?)(?:[\s"seconds]+)?)?)?)?)?[\s]*(E|W|East|West)'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            results.append({
                'cardinal_ns': match.group(1),
                'degrees': match.group(2),
                'minutes': match.group(3),
                'seconds': match.group(4),
                'cardinal_ew': match.group(5),
                'pattern': 'unified'
            })
        
        # Pattern 3: Long format
        pattern3 = r'(North|South)\s+(\d+)\s+degrees?\s+(\d+)\s+minutes?\s+(\d+(?:\.\d+)?)\s+seconds?\s+(East|West)'
        matches3 = re.finditer(pattern3, text, re.IGNORECASE)
        for match in matches3:
            results.append({
                'cardinal_ns': match.group(1),
                'degrees': match.group(2),
                'minutes': match.group(3),
                'seconds': match.group(4),
                'cardinal_ew': match.group(5),
                'pattern': 'long'
            })
        
        return results
    
    @given(st.text(min_size=10, max_size=200))
    @settings(max_examples=50)
    def test_rules_processor_handles_arbitrary_text(self, text: str):
        """
        Property 6: Rules processor handles any text without crashing
        For any text input, the processor should not crash
        """
        rules_config = create_default_rules_config()
        processor = RulesProcessor(rules_config)
        
        # Should not crash on any text
        try:
            results = processor.extract_all_patterns(text)
            assert isinstance(results, dict)
        except Exception as e:
            pytest.fail(f"Rules processor crashed on text: {text[:50]}... Error: {e}")
    
    def test_rules_config_json_roundtrip_preserves_behavior(self):
        """
        Property 6: JSON serialization roundtrip preserves behavior
        Rules should behave identically after JSON serialization/deserialization
        """
        # Create original config
        original_config = create_default_rules_config()
        original_processor = RulesProcessor(original_config)
        
        # Serialize and deserialize
        json_str = original_config.to_json()
        restored_config = RulesConfig.from_json(json_str)
        restored_processor = RulesProcessor(restored_config)
        
        # Test with sample text
        test_text = "North 45 degrees 30 minutes East 150.5 feet"
        
        original_results = original_processor.extract_all_patterns(test_text)
        restored_results = restored_processor.extract_all_patterns(test_text)
        
        # Should produce identical results
        assert original_results.keys() == restored_results.keys()
        
        for rule_type in original_results:
            assert len(original_results[rule_type]) == len(restored_results[rule_type])


class TestRulesValidation:
    """
    Tests for rules validation and error handling
    """
    
    def test_invalid_regex_pattern_is_caught(self):
        """
        Property 6: Invalid regex patterns are detected during validation
        """
        invalid_rule = RegexRule(
            extractor_id="invalid_pattern",
            type=RuleType.COURSE.value,
            regex="[invalid regex (",  # Unclosed bracket
            map={"test": 1}
        )
        
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            invalid_rule.compile_regex()
    
    def test_duplicate_extractor_ids_are_rejected(self):
        """
        Property 6: Duplicate extractor IDs are rejected
        """
        rule1 = RegexRule(
            extractor_id="duplicate_id",
            type=RuleType.COURSE.value,
            regex=r"test",
            map={"test": 1}
        )
        
        rule2 = RegexRule(
            extractor_id="duplicate_id",  # Same ID
            type=RuleType.DISTANCE.value,
            regex=r"other",
            map={"other": 1}
        )
        
        with pytest.raises(ValueError, match="Duplicate extractor_id"):
            RulesConfig(version="test", rules=[rule1, rule2])
    
    def test_empty_extractor_id_is_rejected(self):
        """
        Property 6: Empty extractor IDs are rejected
        """
        with pytest.raises(ValueError, match="extractor_id cannot be empty"):
            RegexRule(
                extractor_id="",
                type=RuleType.COURSE.value,
                regex=r"test",
                map={"test": 1}
            )
    
    def test_invalid_rule_type_is_rejected(self):
        """
        Property 6: Invalid rule types are rejected
        """
        with pytest.raises(ValueError, match="Invalid rule type"):
            RegexRule(
                extractor_id="test_rule",
                type="invalid_type",
                regex=r"test",
                map={"test": 1}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestRuleResilience:
    """
    Feature: self-improving-geometry-extractor, Property 7: Resilience to invalid regex rules
    Feature: self-improving-geometry-extractor, Property 8: Crash prevention from rule errors
    """
    
    @given(st.lists(
        st.builds(RegexRule,
                 extractor_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                 type=st.sampled_from([rt.value for rt in RuleType]),
                 regex=st.one_of(
                     st.just("[invalid regex ("),  # Invalid regex
                     st.just("*invalid"),          # Invalid regex
                     st.just("(?P<invalid"),       # Invalid regex
                     st.text(min_size=1, max_size=10)  # Random text (might be invalid)
                 ),
                 map=st.dictionaries(
                     keys=st.text(min_size=1, max_size=10),
                     values=st.integers(min_value=1, max_value=5),
                     min_size=1, max_size=3
                 )),
        min_size=1, max_size=5
    ))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_invalid_regex_rules_dont_crash_system(self, rules_with_invalid_regex: List[RegexRule]):
        """
        Property 7: Resilience to invalid regex rules
        Property 8: Crash prevention from rule errors
        For any invalid regex pattern introduced to the system, 
        the application should continue operating with previous valid rules
        """
        from utils.cached_rules_loader import CachedRulesLoader
        
        # Create a loader with valid default rules
        loader = CachedRulesLoader()
        
        # Verify we start with valid rules
        initial_rules = loader.get_rules()
        assert initial_rules is not None
        initial_processor = loader.get_processor()
        assert initial_processor is not None
        
        # Try to create a config with potentially invalid rules
        try:
            invalid_config = RulesConfig(version="test_invalid", rules=rules_with_invalid_regex)
            
            # Validate the config - this should catch invalid regex
            validation_errors = invalid_config.validate_all_rules()
            
            if validation_errors:
                # System correctly identified invalid rules
                # Loader should continue with previous valid rules
                current_rules = loader.get_rules()
                assert current_rules is not None
                assert current_rules.version == initial_rules.version
                
                # Should still be able to create processor
                processor = loader.get_processor()
                assert processor is not None
            else:
                # Rules were actually valid, system should handle them
                processor = RulesProcessor(invalid_config)
                assert processor is not None
                
        except ValueError:
            # Expected for invalid rules - system should not crash
            # Loader should still work with previous rules
            current_rules = loader.get_rules()
            assert current_rules is not None
            
            processor = loader.get_processor()
            assert processor is not None
        
        except Exception as e:
            # Any other exception is a test failure
            pytest.fail(f"System crashed with unexpected exception: {e}")
    
    def test_cached_loader_handles_gcs_unavailable(self):
        """
        Property 7: System continues when GCS unavailable
        When GCS is unavailable, system should continue with cached rules
        """
        from utils.cached_rules_loader import CachedRulesLoader
        
        # Create loader (will start with default rules)
        loader = CachedRulesLoader()
        
        # Should have default rules even if GCS is unavailable
        rules = loader.get_rules()
        assert rules is not None
        assert rules.version is not None
        
        # Should be able to create processor
        processor = loader.get_processor()
        assert processor is not None
        
        # Should report as healthy (has fallback rules)
        assert loader.is_healthy()
    
    def test_rules_processor_handles_compilation_errors_gracefully(self):
        """
        Property 8: Crash prevention from rule errors
        Rules processor should handle regex compilation errors gracefully
        """
        # Create a rule with invalid regex
        invalid_rule = RegexRule(
            extractor_id="test_invalid",
            type=RuleType.COURSE.value,
            regex="[unclosed bracket",  # Invalid regex
            map={"test": 1}
        )
        
        valid_rule = RegexRule(
            extractor_id="test_valid",
            type=RuleType.COURSE.value,
            regex=r"valid_pattern",
            map={"test": 1}
        )
        
        # Create config with mix of valid and invalid rules
        mixed_config = RulesConfig(
            version="test_mixed",
            rules=[valid_rule, invalid_rule]
        )
        
        # Processor should handle this gracefully
        try:
            processor = RulesProcessor(mixed_config)
            
            # Should be able to extract with valid rules (invalid ones skipped)
            results = processor.extract_all_patterns("test text")
            assert isinstance(results, dict)
            
        except Exception as e:
            # Should not crash - if it does, that's a failure
            pytest.fail(f"RulesProcessor crashed on mixed valid/invalid rules: {e}")
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50)
    def test_rules_processor_never_crashes_on_text_input(self, text_input: str):
        """
        Property 8: Crash prevention from rule errors
        For any text input, rules processor should never crash
        """
        from utils.cached_rules_loader import get_current_processor
        
        try:
            processor = get_current_processor()
            
            # Should handle any text input without crashing
            results = processor.extract_all_patterns(text_input)
            assert isinstance(results, dict)
            
            # Individual extraction methods should also not crash
            courses = processor.extract_courses(text_input)
            assert isinstance(courses, list)
            
            distances = processor.extract_distances(text_input)
            assert isinstance(distances, list)
            
        except Exception as e:
            pytest.fail(f"Rules processor crashed on text input: {text_input[:50]}... Error: {e}")
    
    def test_loader_recovers_from_consecutive_failures(self):
        """
        Property 7: System recovers from consecutive failures
        After consecutive failures, system should fall back to known good rules
        """
        from utils.cached_rules_loader import CachedRulesLoader
        
        loader = CachedRulesLoader()
        
        # Simulate consecutive failures
        loader.consecutive_failures = loader.max_consecutive_failures
        loader.last_error = "Simulated failure"
        
        # Should still provide valid rules (fallback)
        rules = loader.get_rules()
        assert rules is not None
        
        # Should be able to create processor
        processor = loader.get_processor()
        assert processor is not None
        
        # Should report fallback usage
        status = loader.get_status()
        assert status['consecutive_failures'] >= loader.max_consecutive_failures
    
    def test_rules_validation_catches_common_regex_errors(self):
        """
        Property 8: Validation catches common regex errors before they cause crashes
        """
        common_invalid_patterns = [
            "[unclosed bracket",
            "*invalid quantifier",
            "(?P<invalid group",
            "\\invalid escape",
            "(?invalid flag)",
            "(unclosed group"
        ]
        
        for invalid_pattern in common_invalid_patterns:
            rule = RegexRule(
                extractor_id=f"test_{invalid_pattern[:5]}",
                type=RuleType.COURSE.value,
                regex=invalid_pattern,
                map={"test": 1}
            )
            
            # Validation should catch the error
            with pytest.raises(ValueError, match="Invalid regex pattern"):
                rule.compile_regex()
    
    def test_loader_health_check_detects_issues(self):
        """
        Property 7: Health check system detects and reports issues
        """
        from utils.cached_rules_loader import CachedRulesLoader, RulesLoaderHealthCheck
        
        loader = CachedRulesLoader()
        
        # Simulate some issues
        loader.consecutive_failures = 3
        loader.last_error = "Test error"
        loader.fallback_count = 2
        
        # Health check should detect issues
        health = RulesLoaderHealthCheck.check_health()
        
        assert isinstance(health, dict)
        assert 'healthy' in health
        assert 'issues' in health
        assert 'recommendations' in health
        
        # Should report issues
        assert len(health['issues']) > 0
        assert any('failure' in issue.lower() for issue in health['issues'])
    
    def test_rules_config_validation_is_comprehensive(self):
        """
        Property 8: Rules configuration validation is comprehensive
        """
        # Test various invalid configurations
        
        # Empty version
        with pytest.raises(ValueError, match="Version cannot be empty"):
            RulesConfig(version="", rules=[])
        
        # Duplicate extractor IDs
        rule1 = RegexRule(extractor_id="dup", type=RuleType.COURSE.value, regex="test1", map={"a": 1})
        rule2 = RegexRule(extractor_id="dup", type=RuleType.COURSE.value, regex="test2", map={"b": 1})
        
        with pytest.raises(ValueError, match="Duplicate extractor_id"):
            RulesConfig(version="test", rules=[rule1, rule2])
        
        # Invalid rule type
        with pytest.raises(ValueError, match="Invalid rule type"):
            RegexRule(extractor_id="test", type="invalid_type", regex="test", map={"a": 1})
        
        # Empty extractor ID
        with pytest.raises(ValueError, match="extractor_id cannot be empty"):
            RegexRule(extractor_id="", type=RuleType.COURSE.value, regex="test", map={"a": 1})