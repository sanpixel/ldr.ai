"""
Property-based tests for normalized schema consistency
Tests universal properties that should hold across all inputs
"""

import pytest
from hypothesis import given, strategies as st, settings
from utils.schema import (
    SchemaOutput, LineData, LineType, BucketClassification,
    normalize_gpt_output, create_empty_schema, validate_json_schema
)
import json
from typing import Dict, Any, List


# Hypothesis strategies for generating test data
line_types = st.sampled_from([lt.value for lt in LineType])
bucket_types = st.sampled_from([bc.value for bc in BucketClassification])

@st.composite
def line_data_strategy(draw):
    """Generate valid LineData instances"""
    line_type = draw(line_types)
    idx = draw(st.integers(min_value=0, max_value=100))
    raw = draw(st.text(min_size=1, max_size=100))
    
    # Generate type-specific fields
    if line_type in [LineType.COURSE.value, LineType.CURVE.value]:
        cardinal_ns = draw(st.one_of(st.none(), st.sampled_from(['North', 'South'])))
        degrees = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=359)))
        minutes = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=59)))
        seconds = draw(st.one_of(st.none(), st.floats(min_value=0, max_value=59.99)))
        cardinal_ew = draw(st.one_of(st.none(), st.sampled_from(['East', 'West'])))
        distance = draw(st.one_of(st.none(), st.floats(min_value=0.1, max_value=10000)))
        monument = draw(st.one_of(st.none(), st.text(max_size=50)))
        reference = None
    else:  # ref_segment or ref_curve
        cardinal_ns = None
        degrees = None
        minutes = None
        seconds = None
        cardinal_ew = None
        distance = None
        monument = draw(st.one_of(st.none(), st.text(max_size=50)))
        reference = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    
    return LineData(
        type=line_type,
        idx=idx,
        raw=raw,
        cardinal_ns=cardinal_ns,
        degrees=degrees,
        minutes=minutes,
        seconds=seconds,
        cardinal_ew=cardinal_ew,
        distance=distance,
        monument=monument,
        reference=reference
    )

@st.composite
def schema_output_strategy(draw):
    """Generate valid SchemaOutput instances"""
    bucket = draw(bucket_types)
    lines = draw(st.lists(line_data_strategy(), min_size=0, max_size=10))
    return SchemaOutput(bucket=bucket, lines=lines)

@st.composite
def legacy_bearing_strategy(draw):
    """Generate legacy bearing dictionaries for testing normalization"""
    return {
        'cardinal_ns': draw(st.one_of(st.none(), st.sampled_from(['North', 'South']))),
        'degrees': draw(st.one_of(st.none(), st.integers(min_value=0, max_value=359))),
        'minutes': draw(st.one_of(st.none(), st.integers(min_value=0, max_value=59))),
        'seconds': draw(st.one_of(st.none(), st.floats(min_value=0, max_value=59.99))),
        'cardinal_ew': draw(st.one_of(st.none(), st.sampled_from(['East', 'West']))),
        'distance': draw(st.one_of(st.none(), st.floats(min_value=0.1, max_value=10000))),
        'monument': draw(st.one_of(st.none(), st.text(max_size=50))),
        'original_text': draw(st.text(min_size=1, max_size=100)),
        'bearing': draw(st.text(max_size=50))
    }


class TestSchemaConsistency:
    """
    Feature: self-improving-geometry-extractor, Property 1: Schema consistency across all outputs
    """
    
    @given(schema_output_strategy())
    @settings(max_examples=100)
    def test_schema_always_has_bucket_and_lines(self, schema_output: SchemaOutput):
        """
        Property 1: Schema consistency across all outputs
        For any legal description input, the system output should always contain 
        a bucket field and lines array with valid structure
        """
        # Convert to dict to verify structure
        output_dict = schema_output.to_dict()
        
        # Must have bucket field
        assert 'bucket' in output_dict
        assert isinstance(output_dict['bucket'], str)
        assert output_dict['bucket'] in [bc.value for bc in BucketClassification]
        
        # Must have lines field as array
        assert 'lines' in output_dict
        assert isinstance(output_dict['lines'], list)
        
        # Each line must have required structure
        for line in output_dict['lines']:
            assert isinstance(line, dict)
            assert 'type' in line
            assert 'idx' in line
            assert 'raw' in line
            assert line['type'] in [lt.value for lt in LineType]
    
    @given(st.lists(legacy_bearing_strategy(), min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_normalized_gpt_output_has_consistent_schema(self, legacy_bearings: List[Dict[str, Any]]):
        """
        Property 1: Schema consistency for normalized GPT output
        For any legacy bearing list, normalization should produce consistent schema
        """
        normalized = normalize_gpt_output(legacy_bearings)
        
        # Verify schema structure
        assert hasattr(normalized, 'bucket')
        assert hasattr(normalized, 'lines')
        assert isinstance(normalized.bucket, str)
        assert isinstance(normalized.lines, list)
        
        # Verify bucket is valid
        assert normalized.bucket in [bc.value for bc in BucketClassification]
        
        # Verify each line has consistent structure
        for line in normalized.lines:
            assert isinstance(line, LineData)
            assert hasattr(line, 'type')
            assert hasattr(line, 'idx')
            assert hasattr(line, 'raw')
            assert line.type in [lt.value for lt in LineType]
    
    @given(schema_output_strategy())
    @settings(max_examples=100)
    def test_json_serialization_roundtrip_preserves_schema(self, schema_output: SchemaOutput):
        """
        Property 1: JSON serialization preserves schema structure
        For any schema output, JSON roundtrip should preserve structure
        """
        # Serialize to JSON and back
        json_str = schema_output.to_json()
        reconstructed = SchemaOutput.from_json(json_str)
        
        # Verify structure is preserved
        assert reconstructed.bucket == schema_output.bucket
        assert len(reconstructed.lines) == len(schema_output.lines)
        
        for orig_line, recon_line in zip(schema_output.lines, reconstructed.lines):
            assert orig_line.type == recon_line.type
            assert orig_line.idx == recon_line.idx
            assert orig_line.raw == recon_line.raw
    
    @given(schema_output_strategy())
    @settings(max_examples=100)
    def test_schema_validation_is_consistent(self, schema_output: SchemaOutput):
        """
        Property 1: Schema validation is consistent
        For any valid schema output, validation should pass consistently
        """
        # Valid schema should have no validation errors
        validation_errors = schema_output.validate_schema()
        
        # If schema was constructed properly, it should be valid
        if schema_output.is_valid():
            assert len(validation_errors) == 0
        
        # Validation should be deterministic
        validation_errors_2 = schema_output.validate_schema()
        assert validation_errors == validation_errors_2


class TestLineTypeClassification:
    """
    Feature: self-improving-geometry-extractor, Property 2: Line type classification completeness
    """
    
    @given(line_data_strategy())
    @settings(max_examples=100)
    def test_line_type_is_always_valid(self, line_data: LineData):
        """
        Property 2: Line type classification completeness
        For any extracted line, the type field should always be one of: 
        course, curve, ref_segment, or ref_curve
        """
        valid_types = [lt.value for lt in LineType]
        assert line_data.type in valid_types
        
        # Type should be a string
        assert isinstance(line_data.type, str)
        
        # Type should not be empty
        assert len(line_data.type) > 0
    
    @given(st.lists(line_data_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_lines_in_schema_have_valid_types(self, lines: List[LineData]):
        """
        Property 2: All lines in schema have valid types
        For any list of lines, all should have valid type classifications
        """
        schema_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=lines)
        
        valid_types = [lt.value for lt in LineType]
        for line in schema_output.lines:
            assert line.type in valid_types


class TestNullValueConsistency:
    """
    Feature: self-improving-geometry-extractor, Property 3: Null value consistency for missing data
    """
    
    @given(line_data_strategy())
    @settings(max_examples=100)
    def test_missing_fields_are_null_not_omitted(self, line_data: LineData):
        """
        Property 3: Null value consistency for missing data
        For any line with incomplete data, missing fields should be null 
        rather than omitted from the output structure
        """
        line_dict = line_data.to_dict()
        
        # All expected fields should be present
        expected_fields = [
            'type', 'idx', 'raw', 'cardinal_ns', 'degrees', 'minutes', 
            'seconds', 'cardinal_ew', 'distance', 'monument', 'reference'
        ]
        
        for field in expected_fields:
            assert field in line_dict  # Field must be present
            # Value can be None (null) but field must exist
    
    @given(st.dictionaries(
        keys=st.sampled_from(['cardinal_ns', 'degrees', 'minutes', 'seconds', 'cardinal_ew', 'distance', 'monument', 'reference']),
        values=st.one_of(st.none(), st.text(), st.integers(), st.floats()),
        min_size=0,
        max_size=5
    ))
    @settings(max_examples=100)
    def test_partial_data_creates_complete_structure(self, partial_data: Dict[str, Any]):
        """
        Property 3: Partial data creates complete structure with nulls
        For any partial line data, the complete structure should be created with nulls
        """
        # Create line with minimal required data plus partial data
        line_data = LineData(
            type=LineType.COURSE.value,
            idx=1,
            raw="test bearing",
            **partial_data
        )
        
        line_dict = line_data.to_dict()
        
        # All fields should be present
        expected_fields = [
            'type', 'idx', 'raw', 'cardinal_ns', 'degrees', 'minutes', 
            'seconds', 'cardinal_ew', 'distance', 'monument', 'reference'
        ]
        
        for field in expected_fields:
            assert field in line_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestBucketDerivation:
    """
    Feature: self-improving-geometry-extractor, Property 4: Bucket derivation from extraction results
    Feature: self-improving-geometry-extractor, Property 5: Explicit bearings bucket classification
    """
    
    @given(st.lists(line_data_strategy(), min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_bucket_derived_from_line_content_only(self, lines: List[LineData]):
        """
        Property 4: Bucket derivation from extraction results
        For any legal description, bucket classification should be determined 
        solely from extracted line content, not original text
        """
        # Create schema with arbitrary initial bucket
        schema_output = SchemaOutput(bucket=BucketClassification.NO_BEARINGS.value, lines=lines)
        
        # Derive bucket from line content
        derived_bucket = schema_output.derive_bucket_from_lines()
        
        # Update the bucket
        schema_output.update_bucket_classification()
        
        # Bucket should match what was derived from lines
        assert schema_output.bucket == derived_bucket
        
        # Bucket should be valid
        assert schema_output.bucket in [bc.value for bc in BucketClassification]
    
    @given(st.lists(
        st.builds(LineData,
                 type=st.sampled_from([LineType.COURSE.value, LineType.CURVE.value]),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 cardinal_ns=st.sampled_from(['North', 'South']),
                 degrees=st.integers(min_value=0, max_value=359),
                 minutes=st.integers(min_value=0, max_value=59),
                 seconds=st.floats(min_value=0, max_value=59.99),
                 cardinal_ew=st.sampled_from(['East', 'West']),
                 distance=st.floats(min_value=0.1, max_value=10000)),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_numeric_course_curve_lines_create_explicit_bearings(self, numeric_lines: List[LineData]):
        """
        Property 5: Explicit bearings bucket classification
        For any output containing numeric course or curve lines, 
        the bucket should be classified as explicit_bearings
        """
        schema_output = SchemaOutput(bucket=BucketClassification.NO_BEARINGS.value, lines=numeric_lines)
        schema_output.update_bucket_classification()
        
        # Should be classified as explicit bearings
        assert schema_output.bucket == BucketClassification.EXPLICIT_BEARINGS.value
    
    @given(st.lists(
        st.builds(LineData,
                 type=st.sampled_from([LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 reference=st.text(min_size=1, max_size=50)),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_reference_only_lines_create_abstract_bearings(self, reference_lines: List[LineData]):
        """
        Property 5: Reference lines create abstract bearings classification
        For any output containing only reference lines, 
        the bucket should be classified as abstract_bearings
        """
        schema_output = SchemaOutput(bucket=BucketClassification.NO_BEARINGS.value, lines=reference_lines)
        schema_output.update_bucket_classification()
        
        # Should be classified as abstract bearings
        assert schema_output.bucket == BucketClassification.ABSTRACT_BEARINGS.value
    
    def test_empty_lines_create_no_bearings(self):
        """
        Property 5: Empty lines create no_bearings classification
        For any output with no lines, bucket should be no_bearings
        """
        schema_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=[])
        schema_output.update_bucket_classification()
        
        # Should be classified as no bearings
        assert schema_output.bucket == BucketClassification.NO_BEARINGS.value
    
    @given(st.lists(
        st.builds(LineData,
                 type=st.sampled_from([LineType.COURSE.value, LineType.CURVE.value]),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 cardinal_ns=st.just(None),  # Missing numeric data
                 degrees=st.just(None),
                 distance=st.just(None)),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_incomplete_numeric_lines_not_explicit_bearings(self, incomplete_lines: List[LineData]):
        """
        Property 5: Incomplete numeric lines don't create explicit bearings
        For course/curve lines without complete numeric data, 
        should not be classified as explicit_bearings
        """
        schema_output = SchemaOutput(bucket=BucketClassification.NO_BEARINGS.value, lines=incomplete_lines)
        schema_output.update_bucket_classification()
        
        # Should NOT be classified as explicit bearings since data is incomplete
        assert schema_output.bucket != BucketClassification.EXPLICIT_BEARINGS.value
    
    @given(
        st.lists(
            st.builds(LineData,
                     type=st.sampled_from([LineType.COURSE.value, LineType.CURVE.value]),
                     idx=st.integers(min_value=1, max_value=5),
                     raw=st.text(min_size=1, max_size=50),
                     cardinal_ns=st.sampled_from(['North', 'South']),
                     degrees=st.integers(min_value=0, max_value=359),
                     distance=st.floats(min_value=0.1, max_value=10000)),
            min_size=1, max_size=3
        ),
        st.lists(
            st.builds(LineData,
                     type=st.sampled_from([LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]),
                     idx=st.integers(min_value=6, max_value=10),
                     raw=st.text(min_size=1, max_size=50),
                     reference=st.text(min_size=1, max_size=50)),
            min_size=0, max_size=3
        )
    )
    @settings(max_examples=100)
    def test_explicit_bearings_take_precedence(self, numeric_lines: List[LineData], reference_lines: List[LineData]):
        """
        Property 5: Explicit bearings take precedence over abstract
        For any output with both numeric and reference lines,
        should be classified as explicit_bearings
        """
        all_lines = numeric_lines + reference_lines
        schema_output = SchemaOutput(bucket=BucketClassification.NO_BEARINGS.value, lines=all_lines)
        schema_output.update_bucket_classification()
        
        # Should be classified as explicit bearings (takes precedence)
        assert schema_output.bucket == BucketClassification.EXPLICIT_BEARINGS.value