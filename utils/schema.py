"""
Normalized output schema for Legal Description Reader
Provides consistent data structures for all extraction results
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json


class LineType(Enum):
    """Valid line types for extracted geometry data"""
    COURSE = "course"
    CURVE = "curve"
    REF_SEGMENT = "ref_segment"
    REF_CURVE = "ref_curve"


class BucketClassification(Enum):
    """Valid bucket classifications for legal descriptions"""
    EXPLICIT_BEARINGS = "explicit_bearings"
    ABSTRACT_BEARINGS = "abstract_bearings"
    NO_BEARINGS = "no_bearings"


@dataclass
class LineData:
    """
    Normalized structure for extracted line data
    All fields use null values for missing data rather than omitting them
    """
    type: str  # LineType enum value
    idx: int   # sequence number
    raw: str   # original text
    
    # Type-specific fields (null if not applicable)
    cardinal_ns: Optional[str] = None      # North/South
    degrees: Optional[int] = None          # Degrees component
    minutes: Optional[int] = None          # Minutes component  
    seconds: Optional[float] = None        # Seconds component
    cardinal_ew: Optional[str] = None      # East/West
    distance: Optional[float] = None       # Distance in feet
    monument: Optional[str] = None         # Monument/marker description
    reference: Optional[str] = None        # Reference line/boundary
    
    def __post_init__(self):
        """Validate line type is valid"""
        if self.type not in [lt.value for lt in LineType]:
            raise ValueError(f"Invalid line type: {self.type}. Must be one of {[lt.value for lt in LineType]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with explicit null values"""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if line has minimum required data for its type"""
        if self.type in [LineType.COURSE.value, LineType.CURVE.value]:
            return (self.cardinal_ns is not None and 
                   self.degrees is not None and 
                   self.distance is not None)
        elif self.type in [LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]:
            return self.reference is not None
        return False


@dataclass
class SchemaOutput:
    """
    Normalized output schema for all legal description processing
    Ensures consistent structure regardless of input type or processing path
    """
    bucket: str  # BucketClassification enum value
    lines: List[LineData]
    
    def __post_init__(self):
        """Validate bucket classification is valid"""
        if self.bucket not in [bc.value for bc in BucketClassification]:
            raise ValueError(f"Invalid bucket: {self.bucket}. Must be one of {[bc.value for bc in BucketClassification]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'bucket': self.bucket,
            'lines': [line.to_dict() for line in self.lines]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaOutput':
        """Create SchemaOutput from dictionary"""
        lines = [LineData(**line_data) for line_data in data.get('lines', [])]
        return cls(bucket=data['bucket'], lines=lines)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SchemaOutput':
        """Create SchemaOutput from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def derive_bucket_from_lines(self) -> str:
        """
        Derive bucket classification based on extracted line content
        This implements the requirement that bucket is determined after extraction
        """
        if not self.lines:
            return BucketClassification.NO_BEARINGS.value
        
        # Check for numeric course or curve lines (explicit bearings)
        has_numeric_bearings = any(
            line.type in [LineType.COURSE.value, LineType.CURVE.value] and
            line.degrees is not None and line.distance is not None
            for line in self.lines
        )
        
        if has_numeric_bearings:
            return BucketClassification.EXPLICIT_BEARINGS.value
        
        # Check for reference lines (abstract bearings)
        has_references = any(
            line.type in [LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]
            for line in self.lines
        )
        
        if has_references:
            return BucketClassification.ABSTRACT_BEARINGS.value
        
        return BucketClassification.NO_BEARINGS.value
    
    def update_bucket_classification(self):
        """Update bucket based on current line content"""
        self.bucket = self.derive_bucket_from_lines()
    
    def validate_schema(self) -> List[str]:
        """
        Validate the schema structure and return list of validation errors
        Returns empty list if valid
        """
        errors = []
        
        # Validate bucket
        if self.bucket not in [bc.value for bc in BucketClassification]:
            errors.append(f"Invalid bucket classification: {self.bucket}")
        
        # Validate lines
        if not isinstance(self.lines, list):
            errors.append("Lines must be a list")
        else:
            for i, line in enumerate(self.lines):
                if not isinstance(line, LineData):
                    errors.append(f"Line {i} is not a LineData instance")
                    continue
                
                # Validate line type
                if line.type not in [lt.value for lt in LineType]:
                    errors.append(f"Line {i} has invalid type: {line.type}")
                
                # Validate required fields are present (not necessarily non-null)
                required_fields = ['type', 'idx', 'raw']
                for field in required_fields:
                    if not hasattr(line, field):
                        errors.append(f"Line {i} missing required field: {field}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if schema is valid"""
        return len(self.validate_schema()) == 0


def normalize_legacy_bearing(bearing_dict: Dict[str, Any]) -> LineData:
    """
    Convert legacy bearing dictionary to normalized LineData
    Handles the current bearing format from GPT extraction
    """
    return LineData(
        type=LineType.COURSE.value,
        idx=bearing_dict.get('idx', 0),
        raw=bearing_dict.get('original_text', bearing_dict.get('bearing', '')),
        cardinal_ns=bearing_dict.get('cardinal_ns'),
        degrees=bearing_dict.get('degrees'),
        minutes=bearing_dict.get('minutes'),
        seconds=bearing_dict.get('seconds'),
        cardinal_ew=bearing_dict.get('cardinal_ew'),
        distance=bearing_dict.get('distance'),
        monument=bearing_dict.get('monument')
    )


def normalize_gpt_output(bearings: List[Dict[str, Any]], classification: str = None) -> SchemaOutput:
    """
    Convert current GPT output format to normalized schema
    This provides backward compatibility during the transition
    """
    # Convert bearings to normalized LineData
    lines = []
    for i, bearing in enumerate(bearings):
        line = normalize_legacy_bearing(bearing)
        line.idx = i + 1  # Ensure sequential indexing
        lines.append(line)
    
    # Create schema output
    schema_output = SchemaOutput(
        bucket=BucketClassification.NO_BEARINGS.value,  # Will be derived
        lines=lines
    )
    
    # Derive bucket from line content (not from classification parameter)
    schema_output.update_bucket_classification()
    
    return schema_output


def create_empty_schema() -> SchemaOutput:
    """Create an empty schema with no_bearings classification"""
    return SchemaOutput(
        bucket=BucketClassification.NO_BEARINGS.value,
        lines=[]
    )


def validate_json_schema(data: Dict[str, Any]) -> List[str]:
    """
    Validate that a dictionary conforms to the expected schema structure
    Returns list of validation errors, empty if valid
    """
    errors = []
    
    # Check required top-level fields
    if 'bucket' not in data:
        errors.append("Missing required field: bucket")
    elif data['bucket'] not in [bc.value for bc in BucketClassification]:
        errors.append(f"Invalid bucket value: {data['bucket']}")
    
    if 'lines' not in data:
        errors.append("Missing required field: lines")
    elif not isinstance(data['lines'], list):
        errors.append("Field 'lines' must be an array")
    else:
        # Validate each line
        for i, line in enumerate(data['lines']):
            if not isinstance(line, dict):
                errors.append(f"Line {i} must be an object")
                continue
            
            # Check required line fields
            required_line_fields = ['type', 'idx', 'raw']
            for field in required_line_fields:
                if field not in line:
                    errors.append(f"Line {i} missing required field: {field}")
            
            # Validate line type
            if 'type' in line and line['type'] not in [lt.value for lt in LineType]:
                errors.append(f"Line {i} has invalid type: {line['type']}")
    
    return errors


def extract_bearings_with_normalized_schema(text: str, filename: str, user_email: str, 
                                          file_size: int = None, page_count: int = None) -> SchemaOutput:
    """
    Enhanced version of extract_bearings_with_gpt that returns normalized schema
    This function wraps the existing GPT extraction and normalizes the output
    """
    import streamlit as st
    from main import extract_bearings_with_gpt
    
    try:
        # Use existing GPT extraction function
        bearings, result_text = extract_bearings_with_gpt(
            text, filename, user_email, file_size, page_count
        )
        
        # Convert to normalized schema
        schema_output = normalize_gpt_output(bearings)
        
        # Validate the schema
        validation_errors = schema_output.validate_schema()
        if validation_errors:
            if st.session_state.get('debug_enabled', False):
                st.warning(f"Schema validation errors: {validation_errors}")
            # Return empty schema if validation fails
            return create_empty_schema()
        
        return schema_output
        
    except Exception as e:
        if st.session_state.get('debug_enabled', False):
            st.error(f"Error in normalized extraction: {str(e)}")
        return create_empty_schema()


def apply_schema_validation_defaults(line_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values for schema validation errors
    Ensures all required fields are present with appropriate null values
    """
    defaults = {
        'type': LineType.COURSE.value,
        'idx': 0,
        'raw': '',
        'cardinal_ns': None,
        'degrees': None,
        'minutes': None,
        'seconds': None,
        'cardinal_ew': None,
        'distance': None,
        'monument': None,
        'reference': None
    }
    
    # Apply defaults for missing fields
    for key, default_value in defaults.items():
        if key not in line_data:
            line_data[key] = default_value
    
    return line_data