"""
Regex rules management for Legal Description Reader
Handles loading, validation, and processing of externalized regex patterns
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Valid rule types for extraction patterns"""
    COURSE = "course"
    CURVE = "curve"
    DISTANCE = "distance"
    MONUMENT = "monument"
    REFERENCE = "reference"


@dataclass
class RegexRule:
    """
    Individual regex rule for pattern extraction
    """
    extractor_id: str           # Unique identifier for this rule
    type: str                   # RuleType enum value
    regex: str                  # The regex pattern
    map: Dict[str, int]         # Mapping of field names to capture groups
    description: Optional[str] = None  # Human-readable description
    enabled: bool = True        # Whether this rule is active
    
    def __post_init__(self):
        """Validate rule structure"""
        if self.type not in [rt.value for rt in RuleType]:
            raise ValueError(f"Invalid rule type: {self.type}")
        
        if not self.extractor_id:
            raise ValueError("extractor_id cannot be empty")
        
        if not self.regex:
            raise ValueError("regex pattern cannot be empty")
    
    def compile_regex(self) -> re.Pattern:
        """Compile the regex pattern with validation"""
        try:
            return re.compile(self.regex, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in rule {self.extractor_id}: {e}")
    
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract data from text using this rule
        Returns list of matches with mapped field names
        """
        try:
            compiled_pattern = self.compile_regex()
            matches = []
            
            for match in compiled_pattern.finditer(text):
                extracted = {}
                for field_name, group_index in self.map.items():
                    try:
                        extracted[field_name] = match.group(group_index)
                    except IndexError:
                        extracted[field_name] = None
                
                # Add metadata
                extracted['_rule_id'] = self.extractor_id
                extracted['_rule_type'] = self.type
                extracted['_match_text'] = match.group(0)
                extracted['_match_start'] = match.start()
                extracted['_match_end'] = match.end()
                
                matches.append(extracted)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error extracting with rule {self.extractor_id}: {e}")
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegexRule':
        """Create RegexRule from dictionary"""
        return cls(**data)


@dataclass
class RulesConfig:
    """
    Complete configuration of regex rules
    """
    version: str
    rules: List[RegexRule]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate rules configuration"""
        if not self.version:
            raise ValueError("Version cannot be empty")
        
        # Check for duplicate extractor_ids
        extractor_ids = [rule.extractor_id for rule in self.rules]
        if len(extractor_ids) != len(set(extractor_ids)):
            raise ValueError("Duplicate extractor_id found in rules")
    
    def get_rules_by_type(self, rule_type: str) -> List[RegexRule]:
        """Get all enabled rules of a specific type"""
        return [rule for rule in self.rules 
                if rule.type == rule_type and rule.enabled]
    
    def get_rule_by_id(self, extractor_id: str) -> Optional[RegexRule]:
        """Get a specific rule by its extractor_id"""
        for rule in self.rules:
            if rule.extractor_id == extractor_id:
                return rule
        return None
    
    def validate_all_rules(self) -> List[str]:
        """
        Validate all rules and return list of errors
        Returns empty list if all rules are valid
        """
        errors = []
        
        for rule in self.rules:
            try:
                rule.compile_regex()
            except ValueError as e:
                errors.append(f"Rule {rule.extractor_id}: {e}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'version': self.version,
            'rules': [rule.to_dict() for rule in self.rules],
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RulesConfig':
        """Create RulesConfig from dictionary"""
        rules = [RegexRule.from_dict(rule_data) for rule_data in data.get('rules', [])]
        return cls(
            version=data['version'],
            rules=rules,
            metadata=data.get('metadata')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RulesConfig':
        """Create RulesConfig from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class RulesProcessor:
    """
    Processes text using a set of regex rules
    Replaces hardcoded logic with configuration-driven approach
    """
    
    def __init__(self, rules_config: RulesConfig):
        self.rules_config = rules_config
        self._compiled_rules = {}
        self._compile_all_rules()
    
    def _compile_all_rules(self):
        """Pre-compile all regex patterns for performance"""
        for rule in self.rules_config.rules:
            if rule.enabled:
                try:
                    self._compiled_rules[rule.extractor_id] = rule.compile_regex()
                except ValueError as e:
                    logger.warning(f"Skipping invalid rule {rule.extractor_id}: {e}")
    
    def extract_all_patterns(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all patterns from text using all enabled rules
        Returns results grouped by rule type
        """
        results = {}
        
        for rule in self.rules_config.rules:
            if not rule.enabled:
                continue
            
            matches = rule.extract_from_text(text)
            if matches:
                if rule.type not in results:
                    results[rule.type] = []
                results[rule.type].extend(matches)
        
        return results
    
    def extract_by_type(self, text: str, rule_type: str) -> List[Dict[str, Any]]:
        """Extract patterns of a specific type"""
        results = []
        
        for rule in self.rules_config.get_rules_by_type(rule_type):
            matches = rule.extract_from_text(text)
            results.extend(matches)
        
        return results
    
    def extract_courses(self, text: str) -> List[Dict[str, Any]]:
        """Extract course/bearing patterns"""
        return self.extract_by_type(text, RuleType.COURSE.value)
    
    def extract_distances(self, text: str) -> List[Dict[str, Any]]:
        """Extract distance patterns"""
        return self.extract_by_type(text, RuleType.DISTANCE.value)
    
    def extract_monuments(self, text: str) -> List[Dict[str, Any]]:
        """Extract monument/marker patterns"""
        return self.extract_by_type(text, RuleType.MONUMENT.value)


def create_default_rules_config() -> RulesConfig:
    """
    Create default rules configuration based on current hardcoded patterns
    This preserves existing behavior during the transition
    """
    rules = [
        # Bearing extraction rules (from format_bearing_concise)
        RegexRule(
            extractor_id="bearing_verbose_format",
            type=RuleType.COURSE.value,
            regex=r'(North|South)\s+(\d+)\s*(?:°|degrees?|deg|\s)\s*(\d+)\s*(?:\'|′|minutes?|min|\s)\s*(?:(\d+)\s*(?:"|″|seconds?|sec|\s)\s+)?(East|West)',
            map={
                'cardinal_ns': 1,
                'degrees': 2,
                'minutes': 3,
                'seconds': 4,
                'cardinal_ew': 5
            },
            description="Verbose bearing format: North 45 degrees 30 minutes 15 seconds East"
        ),
        
        # Bearing extraction rules (from extract_bearings_with_gpt)
        RegexRule(
            extractor_id="bearing_unified_format",
            type=RuleType.COURSE.value,
            regex=r'(S|South|N|North)[\s\.]*(\d+)(?:[\s°degrees]+(?:(\d+)(?:[\s\'minutes]+(?:(\d+(?:\.\d+)?)(?:[\s"seconds]+)?)?)?)?)?[\s]*(E|W|East|West)',
            map={
                'cardinal_ns': 1,
                'degrees': 2,
                'minutes': 3,
                'seconds': 4,
                'cardinal_ew': 5
            },
            description="Unified bearing format: S 73° 32' 01\" W or North 71 degrees 51 minutes East"
        ),
        
        RegexRule(
            extractor_id="bearing_long_format",
            type=RuleType.COURSE.value,
            regex=r'(North|South)\s+(\d+)\s+degrees?\s+(\d+)\s+minutes?\s+(\d+(?:\.\d+)?)\s+seconds?\s+(East|West)',
            map={
                'cardinal_ns': 1,
                'degrees': 2,
                'minutes': 3,
                'seconds': 4,
                'cardinal_ew': 5
            },
            description="Long bearing format: North 71 degrees 53 minutes 10 seconds East"
        ),
        
        # Distance extraction rule
        RegexRule(
            extractor_id="distance_decimal",
            type=RuleType.DISTANCE.value,
            regex=r'(\d+(?:\.\d+)?)',
            map={
                'distance': 1
            },
            description="Decimal distance values"
        ),
        
        # JSON extraction rule (for GPT output parsing)
        RegexRule(
            extractor_id="json_section",
            type=RuleType.REFERENCE.value,
            regex=r'JSON:\s*(\{.*\})',
            map={
                'json_content': 1
            },
            description="JSON section in GPT responses"
        )
    ]
    
    return RulesConfig(
        version="000001",
        rules=rules,
        metadata={
            'created_from': 'hardcoded_patterns',
            'description': 'Initial rules extracted from existing codebase',
            'compatible_with': 'legacy_extraction_functions'
        }
    )


def save_rules_to_file(rules_config: RulesConfig, filepath: str):
    """Save rules configuration to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(rules_config.to_json())


def load_rules_from_file(filepath: str) -> RulesConfig:
    """Load rules configuration from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return RulesConfig.from_json(f.read())