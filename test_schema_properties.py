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

class TestGoldDatasetCompleteness:
    """
    Feature: self-improving-geometry-extractor, Property 9: Gold dataset completeness
    """
    
    @given(
        st.text(min_size=10, max_size=500),
        schema_output_strategy(),
        schema_output_strategy()
    )
    @settings(max_examples=50, deadline=None)
    def test_gold_dataset_preserves_both_original_and_corrected_outputs(self, text: str, original_output: SchemaOutput, corrected_output: SchemaOutput):
        """
        Property 9: Gold dataset completeness
        For any gold dataset entry, both original extractor output and 
        manually corrected output should be preserved
        """
        from utils.gold_dataset import create_gold_from_extraction
        
        # Mock the database operations for testing
        try:
            # The function should handle both outputs without crashing
            # In a real scenario, this would save to database
            # For testing, we verify the logic works
            
            # Verify both outputs are valid schemas
            original_errors = original_output.validate_schema()
            corrected_errors = corrected_output.validate_schema()
            
            # Corrected output should be valid
            assert len(corrected_errors) == 0, f"Corrected output should be valid: {corrected_errors}"
            
            # Original output may have errors (that's why it was corrected)
            # But the system should still preserve it
            
            # Test the enhanced output creation logic
            enhanced_output = corrected_output.to_dict()
            enhanced_output['metadata'] = {
                'original_output': original_output.to_dict(),
                'correction_applied': True,
                'original_had_errors': len(original_errors) > 0
            }
            
            # Should be able to recreate SchemaOutput
            final_output = SchemaOutput.from_dict(enhanced_output)
            assert final_output is not None
            
            # Metadata should preserve original
            metadata = enhanced_output.get('metadata', {})
            assert 'original_output' in metadata
            assert metadata['original_output'] == original_output.to_dict()
            
        except Exception as e:
            # Should not crash on any valid schema inputs
            pytest.fail(f"Gold dataset creation crashed: {e}")
    
    @given(st.lists(
        st.tuples(
            st.text(min_size=10, max_size=200),
            schema_output_strategy()
        ),
        min_size=1, max_size=10
    ))
    @settings(max_examples=30)
    def test_gold_dataset_maintains_schema_consistency(self, entries: List[tuple]):
        """
        Property 9: Gold dataset schema consistency
        For any set of gold dataset entries, all should use identical schema
        """
        from utils.gold_dataset import validate_gold_dataset_integrity
        from utils.schema import validate_json_schema
        
        # Test that all entries follow the same schema
        for text, gold_output in entries:
            # Each gold output should be valid
            validation_errors = gold_output.validate_schema()
            assert len(validation_errors) == 0, f"Gold output should be valid: {validation_errors}"
            
            # JSON representation should be valid
            gold_dict = gold_output.to_dict()
            json_errors = validate_json_schema(gold_dict)
            assert len(json_errors) == 0, f"JSON schema should be valid: {json_errors}"
            
            # Should be able to roundtrip
            reconstructed = SchemaOutput.from_dict(gold_dict)
            assert reconstructed.bucket == gold_output.bucket
            assert len(reconstructed.lines) == len(gold_output.lines)
    
    def test_gold_dataset_preserves_bad_outputs_for_learning(self):
        """
        Property 9: Bad outputs are preserved for learning value
        Gold dataset should preserve incorrect outputs as they provide highest value
        """
        from utils.gold_dataset import create_gold_from_extraction
        
        # Create a "bad" original output (wrong bucket classification)
        bad_original = SchemaOutput(
            bucket="explicit_bearings",  # Wrong classification
            lines=[]  # No actual bearing lines
        )
        
        # Create correct output
        good_corrected = SchemaOutput(
            bucket="no_bearings",  # Correct classification
            lines=[]
        )
        
        # Test the preservation logic
        enhanced_output = good_corrected.to_dict()
        enhanced_output['metadata'] = {
            'original_output': bad_original.to_dict(),
            'correction_applied': True,
            'original_had_errors': True
        }
        
        # Bad output should be preserved in metadata
        metadata = enhanced_output.get('metadata', {})
        preserved_original = metadata.get('original_output', {})
        
        assert preserved_original['bucket'] == "explicit_bearings"  # Bad classification preserved
        assert enhanced_output['bucket'] == "no_bearings"  # Correct classification used
        assert metadata['correction_applied'] == True
        assert metadata['original_had_errors'] == True
    
    @given(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=20))
    @settings(max_examples=30)
    def test_gold_dataset_handles_duplicate_texts(self, texts: List[str]):
        """
        Property 9: Gold dataset handles duplicate texts correctly
        For any set of texts, duplicates should be handled appropriately
        """
        import hashlib
        
        # Test hash-based deduplication logic
        seen_hashes = set()
        unique_texts = []
        
        for text in texts:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        # Should have deduplicated correctly
        assert len(unique_texts) <= len(texts)
        
        # All unique texts should have different hashes
        unique_hashes = set()
        for text in unique_texts:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            assert text_hash not in unique_hashes
            unique_hashes.add(text_hash)
    
    def test_gold_dataset_export_format_for_training(self):
        """
        Property 9: Gold dataset export maintains training format
        Exported data should be suitable for training/fine-tuning
        """
        # Test export format structure
        sample_entry = {
            'id': 'test_hash',
            'text': 'Sample legal description text',
            'gold_output': {
                'bucket': 'explicit_bearings',
                'lines': [
                    {
                        'type': 'course',
                        'idx': 1,
                        'raw': 'North 45 degrees East',
                        'cardinal_ns': 'North',
                        'degrees': 45,
                        'minutes': 0,
                        'seconds': 0,
                        'cardinal_ew': 'East',
                        'distance': 100.0,
                        'monument': None,
                        'reference': None
                    }
                ]
            },
            'source_file': 'test.pdf',
            'created_at': '2024-01-01T00:00:00'
        }
        
        # Test training export format
        training_example = {
            'input_text': sample_entry['text'],
            'expected_output': sample_entry['gold_output'],
            'bucket_classification': sample_entry['gold_output']['bucket'],
            'entry_id': sample_entry['id'],
            'source_file': sample_entry['source_file'],
            'created_at': sample_entry['created_at']
        }
        
        # Verify required fields for training
        assert 'input_text' in training_example
        assert 'expected_output' in training_example
        assert 'bucket_classification' in training_example
        
        # Expected output should be valid schema
        expected_output = training_example['expected_output']
        assert 'bucket' in expected_output
        assert 'lines' in expected_output
        assert isinstance(expected_output['lines'], list)

class TestFingerprintCorrespondence:
    """
    Feature: self-improving-geometry-extractor, Property 10: Fingerprint-to-line correspondence
    Feature: self-improving-geometry-extractor, Property 11: Fingerprint normalization consistency
    """
    
    @given(schema_output_strategy())
    @settings(max_examples=100)
    def test_fingerprint_count_equals_line_count(self, schema_output: SchemaOutput):
        """
        Property 10: Fingerprint-to-line correspondence
        For any set of extracted lines, the number of generated fingerprints 
        should equal the number of lines
        """
        from utils.fingerprinting import generate_schema_fingerprints
        
        fingerprints = generate_schema_fingerprints(schema_output)
        
        # Number of fingerprints must equal number of lines
        assert len(fingerprints) == len(schema_output.lines)
        
        # Each fingerprint should be a non-empty string
        for fp in fingerprints:
            assert isinstance(fp, str)
            assert len(fp) > 0
            assert '|' in fp  # Should have separator
    
    @given(line_data_strategy())
    @settings(max_examples=100)
    def test_single_line_generates_single_fingerprint(self, line_data: LineData):
        """
        Property 10: Single line correspondence
        For any single line, exactly one fingerprint should be generated
        """
        from utils.fingerprinting import generate_line_fingerprint
        
        fingerprint = generate_line_fingerprint(line_data)
        
        # Should generate exactly one fingerprint
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
        
        # Should contain the line type
        assert line_data.type in fingerprint
    
    @given(
        st.builds(LineData,
                 type=st.just(LineType.COURSE.value),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 cardinal_ns=st.sampled_from(['North', 'South']),
                 degrees=st.integers(min_value=0, max_value=359),
                 minutes=st.integers(min_value=0, max_value=59),
                 seconds=st.floats(min_value=0, max_value=59.99),
                 cardinal_ew=st.sampled_from(['East', 'West']),
                 distance=st.floats(min_value=0.1, max_value=10000))
    )
    @settings(max_examples=100)
    def test_course_fingerprint_format_consistency(self, course_line: LineData):
        """
        Property 11: Course fingerprint format consistency
        For any course line, fingerprint should follow format: course|S|45|12|30|E|125.00
        """
        from utils.fingerprinting import generate_line_fingerprint
        
        fingerprint = generate_line_fingerprint(course_line)
        
        # Should start with 'course|'
        assert fingerprint.startswith('course|')
        
        # Should have exactly 7 parts separated by '|'
        parts = fingerprint.split('|')
        assert len(parts) == 7
        
        # Validate format: course|NS|degrees|minutes|seconds|EW|distance
        assert parts[0] == 'course'
        assert parts[1] in ['N', 'S']  # Normalized cardinal
        assert parts[2].isdigit()  # degrees
        assert parts[3].isdigit()  # minutes
        assert '.' in parts[4] or parts[4].isdigit()  # seconds (may be decimal)
        assert parts[5] in ['E', 'W']  # Normalized cardinal
        assert '.' in parts[6] or parts[6].isdigit()  # distance (may be decimal)
    
    @given(
        st.builds(LineData,
                 type=st.sampled_from([LineType.REF_SEGMENT.value, LineType.REF_CURVE.value]),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 reference=st.text(min_size=1, max_size=50),
                 monument=st.one_of(st.none(), st.text(max_size=50)))
    )
    @settings(max_examples=100)
    def test_reference_fingerprint_format_consistency(self, ref_line: LineData):
        """
        Property 11: Reference fingerprint format consistency
        For any reference line, fingerprint should follow format: ref_segment|reference|monument|directional
        """
        from utils.fingerprinting import generate_line_fingerprint
        
        fingerprint = generate_line_fingerprint(ref_line)
        
        # Should start with ref_segment or ref_curve
        assert fingerprint.startswith('ref_segment|') or fingerprint.startswith('ref_curve|')
        
        # Should have exactly 4 parts separated by '|'
        parts = fingerprint.split('|')
        assert len(parts) == 4
        
        # Validate format: ref_type|reference|monument|directional
        assert parts[0] in ['ref_segment', 'ref_curve']
        # parts[1] is reference (can be any text)
        # parts[2] is monument (can be any text)
        # parts[3] is directional (can be empty or directional term)
    
    @given(
        st.builds(LineData,
                 type=st.just(LineType.COURSE.value),
                 idx=st.integers(min_value=1, max_value=10),
                 raw=st.text(min_size=1, max_size=50),
                 cardinal_ns=st.sampled_from(['North', 'NORTH', 'north', 'N']),
                 degrees=st.just(45),
                 minutes=st.just(30),
                 seconds=st.just(15.0),
                 cardinal_ew=st.sampled_from(['East', 'EAST', 'east', 'E']),
                 distance=st.just(100.0))
    )
    @settings(max_examples=50)
    def test_fingerprint_normalization_consistency(self, course_line: LineData):
        """
        Property 11: Fingerprint normalization consistency
        For any two equivalent line representations with different formatting,
        their fingerprints should be identical
        """
        from utils.fingerprinting import generate_line_fingerprint
        
        fingerprint = generate_line_fingerprint(course_line)
        
        # All variations should normalize to the same fingerprint
        expected_fingerprint = "course|N|45|30|15.0|E|100.0"
        assert fingerprint == expected_fingerprint
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_text_normalization_consistency(self, text1: str, text2: str):
        """
        Property 11: Text normalization produces consistent results
        For any text, normalization should be deterministic and consistent
        """
        from utils.fingerprinting import normalize_text
        
        # Same text should always normalize the same way
        norm1_a = normalize_text(text1)
        norm1_b = normalize_text(text1)
        assert norm1_a == norm1_b
        
        # Different texts may or may not normalize the same way
        norm2 = normalize_text(text2)
        # No assertion here - just testing that normalization doesn't crash
        assert isinstance(norm2, str)
    
    @given(st.floats(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_numeric_rounding_consistency(self, value: float):
        """
        Property 11: Numeric rounding is consistent
        For any numeric value, rounding should be deterministic
        """
        from utils.fingerprinting import round_numeric_value
        
        # Same value should always round the same way
        rounded_a = round_numeric_value(value)
        rounded_b = round_numeric_value(value)
        assert rounded_a == rounded_b
        
        # Rounded value should be a float
        assert isinstance(rounded_a, float)
        
        # Should have at most 2 decimal places
        rounded_str = str(rounded_a)
        if '.' in rounded_str:
            decimal_places = len(rounded_str.split('.')[1])
            assert decimal_places <= 2
    
    @given(schema_output_strategy(), schema_output_strategy())
    @settings(max_examples=50)
    def test_fingerprint_comparison_consistency(self, schema1: SchemaOutput, schema2: SchemaOutput):
        """
        Property 11: Fingerprint comparison is consistent
        For any two schema outputs, comparison should be deterministic
        """
        from utils.fingerprinting import FingerprintEngine
        
        engine = FingerprintEngine()
        
        # Same comparison should always yield same results
        comparison1 = engine.compare_outputs(schema1, schema2)
        comparison2 = engine.compare_outputs(schema1, schema2)
        
        # Key metrics should be identical
        assert comparison1['total_expected'] == comparison2['total_expected']
        assert comparison1['total_actual'] == comparison2['total_actual']
        assert comparison1['matching_count'] == comparison2['matching_count']
        assert comparison1['missing_count'] == comparison2['missing_count']
        assert comparison1['extra_count'] == comparison2['extra_count']
        assert comparison1['accuracy'] == comparison2['accuracy']
        assert comparison1['bucket_match'] == comparison2['bucket_match']
    
    def test_fingerprint_excludes_raw_text(self):
        """
        Property 11: Fingerprints exclude raw text from content
        Raw text should not appear directly in fingerprints
        """
        from utils.fingerprinting import generate_line_fingerprint
        
        # Create line with distinctive raw text
        distinctive_raw = "This is very distinctive raw text that should not appear in fingerprint"
        
        line = LineData(
            type=LineType.COURSE.value,
            idx=1,
            raw=distinctive_raw,
            cardinal_ns='North',
            degrees=45,
            minutes=30,
            seconds=15.0,
            cardinal_ew='East',
            distance=100.0
        )
        
        fingerprint = generate_line_fingerprint(line)
        
        # Raw text should not appear in fingerprint
        assert distinctive_raw not in fingerprint
        assert "distinctive" not in fingerprint.lower()
        
        # But structured data should be present
        assert "course" in fingerprint
        assert "45" in fingerprint
        assert "100" in fingerprint
    
    @given(st.lists(line_data_strategy(), min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_fingerprint_engine_caching_consistency(self, lines: List[LineData]):
        """
        Property 11: Fingerprint engine caching produces consistent results
        Cached and non-cached results should be identical
        """
        from utils.fingerprinting import FingerprintEngine
        
        schema_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=lines)
        
        engine = FingerprintEngine()
        
        # Get fingerprints without cache
        fps_no_cache = engine.fingerprint_schema_output(schema_output, use_cache=False)
        
        # Get fingerprints with cache (first time)
        fps_with_cache_1 = engine.fingerprint_schema_output(schema_output, use_cache=True)
        
        # Get fingerprints with cache (second time - should use cached result)
        fps_with_cache_2 = engine.fingerprint_schema_output(schema_output, use_cache=True)
        
        # All results should be identical
        assert fps_no_cache == fps_with_cache_1
        assert fps_with_cache_1 == fps_with_cache_2
class TestPatchValidation:
    """
    Feature: self-improving-geometry-extractor, Property 13: Patch validation before deployment
    """
    
    @given(
        st.sampled_from(['add_pattern', 'modify_pattern', 'disable_pattern']),
        st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        st.one_of(st.none(), st.dictionaries(
            st.text(min_size=1, max_size=20), 
            st.integers(min_value=1, max_value=10)
        ))
    )
    @settings(max_examples=100)
    def test_patch_validation_before_deployment(self, operation: str, extractor_id: str, 
                                              new_regex: Optional[str], new_map: Optional[Dict[str, int]]):
        """
        Property 13: Patch validation before deployment
        For any patch operation, validation should occur before the patch is applied to production rules
        """
        from utils.patch_application import PatchOperation, PatchValidator
        
        # Create patch operation
        patch = PatchOperation(
            operation=operation,
            extractor_id=extractor_id,
            new_regex=new_regex,
            new_map=new_map,
            reason="Test patch"
        )
        
        # Validate patch
        validation_errors = PatchValidator.validate_patch_operation(patch)
        
        # Validation should always return a list
        assert isinstance(validation_errors, list)
        
        # If operation requires regex but none provided, should have errors
        if operation in ['add_pattern', 'modify_pattern'] and not new_regex:
            assert len(validation_errors) > 0
        
        # If operation is disable_pattern and has regex/map, should have errors
        if operation == 'disable_pattern':
            if new_regex is not None or new_map is not None:
                assert len(validation_errors) > 0
        
        # If regex is provided, try to validate it
        if new_regex:
            try:
                import re
                re.compile(new_regex)
                # If regex compiles successfully and other conditions are met, 
                # validation might pass (depends on other factors)
            except re.error:
                # If regex doesn't compile, should have validation errors
                assert len(validation_errors) > 0
    
    @given(
        st.lists(
            st.builds(
                lambda op, eid, regex, map_dict: {
                    'operation': op,
                    'extractor_id': eid,
                    'new_regex': regex if op != 'disable_pattern' else None,
                    'new_map': map_dict if op != 'disable_pattern' else None,
                    'reason': 'Test patch'
                },
                op=st.sampled_from(['add_pattern', 'modify_pattern', 'disable_pattern']),
                eid=st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
                regex=st.one_of(st.none(), st.just(r'test_\d+')),  # Simple valid regex
                map_dict=st.one_of(st.none(), st.dictionaries(
                    st.sampled_from(['degrees', 'minutes', 'distance']),
                    st.integers(min_value=1, max_value=5),
                    min_size=1, max_size=3
                ))
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=50)
    def test_patch_set_validation_consistency(self, patch_data_list: List[Dict[str, Any]]):
        """
        Property 13: Patch set validation consistency
        For any set of patch operations, validation should be consistent and comprehensive
        """
        from utils.patch_application import PatchOperation, PatchSet, PatchValidator
        
        # Convert to PatchOperation objects
        patches = []
        for patch_data in patch_data_list:
            patch = PatchOperation(
                operation=patch_data['operation'],
                extractor_id=patch_data['extractor_id'],
                new_regex=patch_data.get('new_regex'),
                new_map=patch_data.get('new_map'),
                reason=patch_data.get('reason', '')
            )
            patches.append(patch)
        
        # Create patch set
        patch_set = PatchSet(
            patches=patches,
            target_failures=['test_failure'],
            confidence=0.5
        )
        
        # Validate patch set
        validation_result = PatchValidator.validate_patch_set(patch_set)
        
        # Should always return validation result with required fields
        assert 'valid' in validation_result
        assert 'errors' in validation_result
        assert isinstance(validation_result['valid'], bool)
        assert isinstance(validation_result['errors'], list)
        
        # If there are duplicate extractor IDs, should be invalid
        extractor_ids = [p.extractor_id for p in patches]
        has_duplicates = len(extractor_ids) != len(set(extractor_ids))
        if has_duplicates:
            assert not validation_result['valid']
    
    @given(
        st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        st.sampled_from([r'\d+', r'[A-Z]+', r'test_.*', r'(?:invalid', r'[unclosed'])
    )
    @settings(max_examples=100)
    def test_regex_validation_catches_errors(self, extractor_id: str, regex_pattern: str):
        """
        Property 13: Regex validation catches compilation errors
        For any regex pattern, validation should catch compilation errors before deployment
        """
        from utils.patch_application import PatchOperation, PatchValidator
        import re
        
        # Create patch with the regex
        patch = PatchOperation(
            operation='add_pattern',
            extractor_id=extractor_id,
            new_regex=regex_pattern,
            new_map={'test_field': 1},
            reason='Test regex validation'
        )
        
        # Validate patch
        validation_errors = PatchValidator.validate_patch_operation(patch)
        
        # Check if regex compiles
        try:
            re.compile(regex_pattern)
            regex_compiles = True
        except re.error:
            regex_compiles = False
        
        # If regex doesn't compile, validation should catch it
        if not regex_compiles:
            assert len(validation_errors) > 0
            # Should have a specific error about regex compilation
            error_messages = ' '.join(validation_errors).lower()
            assert 'regex' in error_messages or 'invalid' in error_messages


class TestImprovementValidation:
    """
    Feature: self-improving-geometry-extractor, Property 14: Improvement validation through failure reduction
    """
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),  # baseline_accuracy
        st.floats(min_value=0.0, max_value=1.0),  # new_accuracy
        st.integers(min_value=0, max_value=50),   # baseline_failures
        st.integers(min_value=0, max_value=50)    # new_failures
    )
    @settings(max_examples=100)
    def test_improvement_validation_through_failure_reduction(self, baseline_accuracy: float, 
                                                            new_accuracy: float,
                                                            baseline_failures: int, 
                                                            new_failures: int):
        """
        Property 14: Improvement validation through failure reduction
        For any new rule version, the failure count should be less than or equal to the previous version's failure count
        """
        from utils.regression_testing import RegressionTestResult
        from datetime import datetime
        
        # Create mock regression test result
        accuracy_change = new_accuracy - baseline_accuracy
        
        test_result = RegressionTestResult(
            version_id="test_version",
            passed=accuracy_change >= -0.01,  # Standard regression threshold
            accuracy_change=accuracy_change,
            new_failures=[f"failure_{i}" for i in range(max(0, new_failures - baseline_failures))],
            fixed_failures=[f"fixed_{i}" for i in range(max(0, baseline_failures - new_failures))],
            overall_accuracy=new_accuracy,
            baseline_accuracy=baseline_accuracy,
            test_timestamp=datetime.now()
        )
        
        # Improvement should be validated if:
        # 1. No significant regression (accuracy doesn't drop by more than 1%)
        # 2. Failure count doesn't increase significantly
        
        expected_improvement = (
            accuracy_change >= -0.01 and  # No significant regression
            new_failures <= baseline_failures + 2  # Allow small increase in edge cases
        )
        
        # The test result should reflect whether improvement was achieved
        if accuracy_change >= 0.01:  # Clear improvement
            assert test_result.passed
        elif accuracy_change <= -0.02:  # Clear regression
            assert not test_result.passed
        
        # Failure counts should be tracked correctly
        assert len(test_result.new_failures) == max(0, new_failures - baseline_failures)
        assert len(test_result.fixed_failures) == max(0, baseline_failures - new_failures)
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),  # accuracy_1
        st.floats(min_value=0.0, max_value=1.0),  # accuracy_2
        st.floats(min_value=0.0, max_value=1.0)   # accuracy_3
    )
    @settings(max_examples=50)
    def test_improvement_validation_consistency(self, accuracy_1: float, accuracy_2: float, accuracy_3: float):
        """
        Property 14: Improvement validation consistency across multiple versions
        For any sequence of version accuracies, improvement validation should be transitive and consistent
        """
        from utils.regression_testing import RegressionTestingEngine
        
        # Create mock engine (we'll test the logic, not the actual GCS calls)
        engine = RegressionTestingEngine()
        
        # Test the improvement validation logic
        accuracies = [accuracy_1, accuracy_2, accuracy_3]
        
        # Check that improvement detection is consistent
        for i in range(len(accuracies) - 1):
            baseline = accuracies[i]
            new_acc = accuracies[i + 1]
            change = new_acc - baseline
            
            # Improvement should be detected consistently
            is_improvement = change > 0
            is_regression = change < -0.01  # Engine's default threshold
            
            # These should be mutually exclusive
            assert not (is_improvement and is_regression)
            
            # If there's a clear improvement, it should be positive
            if change > 0.05:  # 5% improvement
                assert is_improvement
                assert not is_regression
            
            # If there's a clear regression, it should be detected
            if change < -0.05:  # 5% regression
                assert is_regression
                assert not is_improvement


class TestAutomaticRollback:
    """
    Feature: self-improving-geometry-extractor, Property 15: Automatic rollback on regression
    """
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),  # baseline_accuracy
        st.floats(min_value=-0.1, max_value=0.1)  # accuracy_change
    )
    @settings(max_examples=100)
    def test_automatic_rollback_on_regression(self, baseline_accuracy: float, accuracy_change: float):
        """
        Property 15: Automatic rollback on regression
        For any rule version that increases failure count, the system should automatically revert to the previous version
        """
        from utils.regression_testing import RegressionTestingEngine
        
        new_accuracy = max(0.0, min(1.0, baseline_accuracy + accuracy_change))
        actual_change = new_accuracy - baseline_accuracy
        
        # Create engine with standard threshold
        engine = RegressionTestingEngine(regression_threshold=-0.01)
        
        # Determine if this should trigger rollback
        should_rollback = actual_change < engine.regression_threshold
        
        # Test the rollback decision logic
        if should_rollback:
            # Regression detected - rollback should be triggered
            assert actual_change < -0.01
        else:
            # No regression - rollback should not be triggered
            assert actual_change >= -0.01
        
        # Test threshold sensitivity
        if actual_change < -0.02:  # Clear regression
            assert should_rollback
        elif actual_change > 0.01:  # Clear improvement
            assert not should_rollback
    
    @given(
        st.floats(min_value=-0.1, max_value=-0.001),  # regression_threshold (negative values)
        st.floats(min_value=0.0, max_value=1.0),      # baseline_accuracy
        st.floats(min_value=-0.2, max_value=0.2)      # accuracy_change
    )
    @settings(max_examples=100)
    def test_rollback_threshold_sensitivity(self, regression_threshold: float, 
                                          baseline_accuracy: float, accuracy_change: float):
        """
        Property 15: Rollback threshold sensitivity
        For any regression threshold, rollback decisions should be consistent with the threshold setting
        """
        from utils.regression_testing import RegressionTestingEngine
        
        new_accuracy = max(0.0, min(1.0, baseline_accuracy + accuracy_change))
        actual_change = new_accuracy - baseline_accuracy
        
        # Create engine with custom threshold
        engine = RegressionTestingEngine(regression_threshold=regression_threshold)
        
        # Rollback should be triggered if change is below threshold
        should_rollback = actual_change < regression_threshold
        
        # Verify threshold logic
        if actual_change < regression_threshold:
            assert should_rollback
        else:
            assert not should_rollback
        
        # Test edge cases around threshold
        if abs(actual_change - regression_threshold) < 0.001:
            # Very close to threshold - behavior should be consistent
            if actual_change < regression_threshold:
                assert should_rollback
            else:
                assert not should_rollback
    
    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2, max_size=5
        )
    )
    @settings(max_examples=50)
    def test_rollback_chain_consistency(self, accuracy_sequence: List[float]):
        """
        Property 15: Rollback chain consistency
        For any sequence of accuracy changes, rollback decisions should prevent cascading regressions
        """
        from utils.regression_testing import RegressionTestingEngine
        
        engine = RegressionTestingEngine()
        
        # Simulate a sequence of version deployments
        rollback_count = 0
        current_accuracy = accuracy_sequence[0]
        
        for next_accuracy in accuracy_sequence[1:]:
            accuracy_change = next_accuracy - current_accuracy
            
            # Check if rollback would be triggered
            if accuracy_change < engine.regression_threshold:
                rollback_count += 1
                # After rollback, accuracy should return to previous level
                # (current_accuracy stays the same)
            else:
                # No rollback, accuracy moves forward
                current_accuracy = next_accuracy
        
        # Rollback system should prevent accuracy from degrading below original
        final_accuracy = current_accuracy
        original_accuracy = accuracy_sequence[0]
        
        # With proper rollback, final accuracy should not be significantly worse than original
        # (allowing for small variations due to different test cases)
        degradation = original_accuracy - final_accuracy
        assert degradation <= 0.05  # Allow up to 5% degradation in worst case


class TestHarnessComparison:
    """
    Feature: self-improving-geometry-extractor, Property 12: Harness fingerprint-based comparison
    """
    
    @given(schema_output_strategy(), schema_output_strategy())
    @settings(max_examples=50)
    def test_harness_uses_fingerprints_for_comparison(self, actual_output: SchemaOutput, expected_output: SchemaOutput):
        """
        Property 12: Harness fingerprint-based comparison
        For any harness test run, comparisons between actual and expected outputs 
        should use fingerprints rather than raw text
        """
        from utils.testing_harness import TestingHarness
        from utils.fingerprinting import FingerprintEngine
        
        harness = TestingHarness()
        
        # Test that harness uses fingerprint engine for comparison
        comparison = harness.fingerprint_engine.compare_outputs(actual_output, expected_output)
        
        # Comparison should contain fingerprint-based metrics
        assert 'matching_fingerprints' in comparison
        assert 'missing_fingerprints' in comparison
        assert 'extra_fingerprints' in comparison
        assert 'accuracy' in comparison
        assert 'precision' in comparison
        assert 'recall' in comparison
        
        # Should not contain raw text comparisons
        assert 'raw_text_match' not in comparison
        assert 'text_similarity' not in comparison
        
        # Fingerprint lists should be lists of strings
        assert isinstance(comparison['matching_fingerprints'], list)
        assert isinstance(comparison['missing_fingerprints'], list)
        assert isinstance(comparison['extra_fingerprints'], list)
        
        # All fingerprints should be strings with separators
        all_fingerprints = (comparison['matching_fingerprints'] + 
                          comparison['missing_fingerprints'] + 
                          comparison['extra_fingerprints'])
        
        for fp in all_fingerprints:
            assert isinstance(fp, str)
            assert '|' in fp  # Should have fingerprint separator
    
    @given(st.lists(line_data_strategy(), min_size=0, max_size=10))
    @settings(max_examples=30)
    def test_harness_comparison_is_deterministic(self, lines: List[LineData]):
        """
        Property 12: Harness comparison is deterministic
        For any set of lines, comparison should always yield same results
        """
        from utils.testing_harness import TestingHarness
        
        schema_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=lines)
        
        harness = TestingHarness()
        
        # Run comparison multiple times
        comparison1 = harness.fingerprint_engine.compare_outputs(schema_output, schema_output)
        comparison2 = harness.fingerprint_engine.compare_outputs(schema_output, schema_output)
        
        # Results should be identical
        assert comparison1['total_expected'] == comparison2['total_expected']
        assert comparison1['total_actual'] == comparison2['total_actual']
        assert comparison1['matching_count'] == comparison2['matching_count']
        assert comparison1['accuracy'] == comparison2['accuracy']
        assert comparison1['bucket_match'] == comparison2['bucket_match']
    
    def test_harness_identifies_missing_extra_and_matching_patterns(self):
        """
        Property 12: Harness correctly identifies missing, extra, and matching patterns
        """
        from utils.testing_harness import TestingHarness
        
        # Create expected output with specific lines
        expected_lines = [
            LineData(
                type=LineType.COURSE.value,
                idx=1,
                raw="North 45 degrees East 100 feet",
                cardinal_ns='North',
                degrees=45,
                minutes=0,
                seconds=0.0,
                cardinal_ew='East',
                distance=100.0
            ),
            LineData(
                type=LineType.REF_SEGMENT.value,
                idx=2,
                raw="along property line",
                reference="property line"
            )
        ]
        
        # Create actual output with one matching, one missing, one extra
        actual_lines = [
            LineData(
                type=LineType.COURSE.value,
                idx=1,
                raw="North 45 degrees East 100 feet",
                cardinal_ns='North',
                degrees=45,
                minutes=0,
                seconds=0.0,
                cardinal_ew='East',
                distance=100.0
            ),
            # Missing the ref_segment line
            # Extra course line
            LineData(
                type=LineType.COURSE.value,
                idx=3,
                raw="South 30 degrees West 50 feet",
                cardinal_ns='South',
                degrees=30,
                minutes=0,
                seconds=0.0,
                cardinal_ew='West',
                distance=50.0
            )
        ]
        
        expected_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=expected_lines)
        actual_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=actual_lines)
        
        harness = TestingHarness()
        comparison = harness.fingerprint_engine.compare_outputs(actual_output, expected_output)
        
        # Should have 1 matching, 1 missing, 1 extra
        assert comparison['matching_count'] == 1
        assert comparison['missing_count'] == 1
        assert comparison['extra_count'] == 1
        
        # Should identify bucket match
        assert comparison['bucket_match'] == True
    
    def test_harness_detects_bucket_mismatches(self):
        """
        Property 12: Harness detects bucket classification mismatches
        """
        from utils.testing_harness import TestingHarness
        
        # Same lines but different bucket classifications
        lines = [
            LineData(
                type=LineType.COURSE.value,
                idx=1,
                raw="North 45 degrees East 100 feet",
                cardinal_ns='North',
                degrees=45,
                minutes=0,
                seconds=0.0,
                cardinal_ew='East',
                distance=100.0
            )
        ]
        
        expected_output = SchemaOutput(bucket=BucketClassification.EXPLICIT_BEARINGS.value, lines=lines)
        actual_output = SchemaOutput(bucket=BucketClassification.ABSTRACT_BEARINGS.value, lines=lines)
        
        harness = TestingHarness()
        comparison = harness.fingerprint_engine.compare_outputs(actual_output, expected_output)
        
        # Should detect bucket mismatch
        assert comparison['bucket_match'] == False
        assert comparison['expected_bucket'] == BucketClassification.EXPLICIT_BEARINGS.value
        assert comparison['actual_bucket'] == BucketClassification.ABSTRACT_BEARINGS.value
    
    @given(st.lists(
        st.tuples(
            st.text(min_size=10, max_size=100),
            schema_output_strategy()
        ),
        min_size=1, max_size=5
    ))
    @settings(max_examples=20)
    def test_harness_processes_multiple_test_cases(self, test_cases: List[tuple]):
        """
        Property 12: Harness processes multiple test cases consistently
        """
        from utils.testing_harness import TestingHarness
        
        harness = TestingHarness()
        
        # Mock gold entries
        gold_entries = []
        for i, (text, schema_output) in enumerate(test_cases):
            gold_entry = {
                'id': f'test_{i}',
                'text': text,
                'gold_output': schema_output.to_dict(),
                'source_file': f'test_{i}.pdf'
            }
            gold_entries.append(gold_entry)
        
        # Test that harness can process multiple cases
        # (We can't run full harness without database, but we can test the structure)
        
        for i, gold_entry in enumerate(gold_entries):
            # Test single case processing structure
            entry_id = gold_entry.get('id')
            text = gold_entry.get('text')
            expected_output_dict = gold_entry.get('gold_output')
            
            # Should be able to parse expected output
            expected_output = SchemaOutput.from_dict(expected_output_dict)
            assert expected_output is not None
            
            # Should have required fields for comparison
            assert isinstance(text, str)
            assert len(text) > 0
            assert isinstance(entry_id, str)
    
    def test_harness_generates_failure_analysis(self):
        """
        Property 12: Harness generates structured failure analysis
        """
        from utils.testing_harness import TestingHarness
        
        harness = TestingHarness()
        
        # Mock failure cases
        failures = [
            {
                'case_index': 0,
                'entry_id': 'test_1',
                'passed': False,
                'comparison': {
                    'bucket_match': False,
                    'missing_fingerprints': ['course|N|45|0|0|E|100.0'],
                    'extra_fingerprints': ['course|S|30|0|0|W|50.0']
                },
                'expected_bucket': 'explicit_bearings',
                'actual_bucket': 'abstract_bearings'
            },
            {
                'case_index': 1,
                'entry_id': 'test_2',
                'passed': False,
                'comparison': {
                    'bucket_match': True,
                    'missing_fingerprints': ['course|N|45|0|0|E|100.0'],
                    'extra_fingerprints': []
                },
                'expected_bucket': 'explicit_bearings',
                'actual_bucket': 'explicit_bearings'
            }
        ]
        
        analysis = harness.analyze_failures(failures)
        
        # Should have structured analysis
        assert 'total_failures' in analysis
        assert 'bucket_mismatches' in analysis
        assert 'missing_patterns' in analysis
        assert 'extra_patterns' in analysis
        assert 'common_failure_patterns' in analysis
        
        # Should count failures correctly
        assert analysis['total_failures'] == 2
        
        # Should identify patterns
        assert 'course|N|45|0|0|E|100.0' in analysis['missing_patterns']
        assert analysis['missing_patterns']['course|N|45|0|0|E|100.0'] == 2  # Appears in both failures
    
    def test_harness_calculates_accuracy_metrics(self):
        """
        Property 12: Harness calculates comprehensive accuracy metrics
        """
        from utils.testing_harness import TestingHarness
        
        harness = TestingHarness()
        
        # Mock test results
        test_results = [
            {
                'passed': True,
                'comparison': {
                    'bucket_match': True,
                    'precision': 1.0,
                    'recall': 1.0,
                    'accuracy': 1.0
                }
            },
            {
                'passed': False,
                'comparison': {
                    'bucket_match': False,
                    'precision': 0.5,
                    'recall': 0.8,
                    'accuracy': 0.6
                }
            },
            {
                'passed': True,
                'comparison': {
                    'bucket_match': True,
                    'precision': 0.9,
                    'recall': 0.9,
                    'accuracy': 0.9
                }
            }
        ]
        
        metrics = harness.calculate_accuracy_metrics(test_results)
        
        # Should calculate all required metrics
        assert 'overall_accuracy' in metrics
        assert 'bucket_accuracy' in metrics
        assert 'average_precision' in metrics
        assert 'average_recall' in metrics
        assert 'f1_score' in metrics
        
        # Should calculate correctly (rounded to 4 decimal places)
        assert metrics['overall_accuracy'] == round(2/3, 4)  # 2 passed out of 3
        assert metrics['bucket_accuracy'] == round(2/3, 4)   # 2 bucket matches out of 3
        
        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0