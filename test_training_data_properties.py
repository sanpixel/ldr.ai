"""
Property-based tests for training data export format compliance
Tests that exported training data follows OpenAI fine-tuning JSONL format
"""

import pytest
import json
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Dict, Any, List
from utils.fine_tuning import (
    export_training_data_as_jsonl, 
    create_training_example,
    validate_jsonl_format,
    TrainingDataExporter
)
from utils.schema import SchemaOutput, LineData, BucketClassification
from utils.gold_dataset import export_gold_dataset_for_training


# Hypothesis strategies for generating test data
bucket_classifications = st.sampled_from([bc.value for bc in BucketClassification])

@st.composite
def training_entry_strategy(draw):
    """Generate training data entries"""
    text = draw(st.text(min_size=10, max_size=100))
    bucket = draw(bucket_classifications)
    
    # Simplified lines generation
    lines = []
    if bucket != BucketClassification.NO_BEARINGS.value:
        # Just create one simple line
        line = {
            'type': 'course',
            'idx': 1,
            'raw': 'North 45 degrees East 100 feet',
            'cardinal_ns': 'North',
            'degrees': 45,
            'minutes': 0,
            'seconds': 0,
            'cardinal_ew': 'East',
            'distance': 100.0,
            'monument': None,
            'reference': None
        }
        lines.append(line)
    
    gold_output = {
        'bucket': bucket,
        'lines': lines
    }
    
    return {
        'id': 'test_id',
        'text': text,
        'gold_output': gold_output,
        'source_file': 'test.pdf',
        'created_at': '2024-01-01T00:00:00Z'
    }

@st.composite
def jsonl_message_strategy(draw):
    """Generate JSONL message format"""
    user_content = draw(st.text(min_size=10, max_size=200))
    assistant_content = draw(st.text(min_size=5, max_size=100))
    
    return {
        'messages': [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content}
        ]
    }


class TestTrainingDataFormatCompliance:
    """
    Feature: self-improving-geometry-extractor, Property 16: Training data export format compliance
    """
    
    @given(st.lists(training_entry_strategy(), min_size=1, max_size=3))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_exported_training_data_has_valid_jsonl_format(self, training_entries: List[Dict[str, Any]]):
        """
        Property 16: Training data export format compliance
        For any exported training data, the format should be valid JSONL with 
        proper message structure for OpenAI fine-tuning
        """
        # Export training data
        exporter = TrainingDataExporter()
        jsonl_lines = exporter.export_entries_to_jsonl(training_entries)
        
        # Each line should be valid JSON
        for line in jsonl_lines:
            assert isinstance(line, str)
            assert len(line.strip()) > 0
            
            # Should be parseable as JSON
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in JSONL line: {line[:100]}... Error: {e}")
            
            # Should have messages array
            assert 'messages' in json_obj
            assert isinstance(json_obj['messages'], list)
            assert len(json_obj['messages']) >= 2  # At least user and assistant
            
            # Check message structure
            for message in json_obj['messages']:
                assert isinstance(message, dict)
                assert 'role' in message
                assert 'content' in message
                assert message['role'] in ['user', 'assistant', 'system']
                assert isinstance(message['content'], str)
                assert len(message['content']) > 0
    
    @given(training_entry_strategy())
    @settings(max_examples=100)
    def test_single_training_example_creates_valid_jsonl_entry(self, training_entry: Dict[str, Any]):
        """
        Property 16: Single training example format compliance
        For any single training entry, the JSONL output should be valid
        """
        # Create training example
        jsonl_entry = create_training_example(training_entry)
        
        # Should be valid JSON string
        assert isinstance(jsonl_entry, str)
        
        try:
            json_obj = json.loads(jsonl_entry)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in training example: {jsonl_entry[:100]}... Error: {e}")
        
        # Should have required structure
        assert 'messages' in json_obj
        messages = json_obj['messages']
        assert isinstance(messages, list)
        assert len(messages) == 2  # User and assistant
        
        # Check user message
        user_msg = messages[0]
        assert user_msg['role'] == 'user'
        assert isinstance(user_msg['content'], str)
        assert len(user_msg['content']) > 0
        assert training_entry['text'] in user_msg['content']  # Should contain original text
        
        # Check assistant message
        assistant_msg = messages[1]
        assert assistant_msg['role'] == 'assistant'
        assert isinstance(assistant_msg['content'], str)
        assert len(assistant_msg['content']) > 0
        
        # Assistant content should contain classification
        bucket = training_entry['gold_output']['bucket']
        assert bucket in assistant_msg['content']
    
    @given(st.lists(jsonl_message_strategy(), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_jsonl_validation_accepts_valid_format(self, valid_messages: List[Dict[str, Any]]):
        """
        Property 16: JSONL validation accepts valid formats
        For any valid JSONL message format, validation should pass
        """
        # Convert to JSONL lines
        jsonl_lines = []
        for msg in valid_messages:
            jsonl_lines.append(json.dumps(msg))
        
        # Validation should pass
        validation_errors = validate_jsonl_format(jsonl_lines)
        assert len(validation_errors) == 0, f"Valid JSONL rejected: {validation_errors}"
    
    def test_jsonl_validation_rejects_invalid_format(self):
        """
        Property 16: JSONL validation rejects invalid formats
        Invalid JSONL formats should be caught by validation
        """
        invalid_examples = [
            '{"invalid": "no messages array"}',
            '{"messages": "not an array"}',
            '{"messages": [{"role": "user"}]}',  # Missing content
            '{"messages": [{"content": "no role"}]}',  # Missing role
            '{"messages": [{"role": "invalid_role", "content": "test"}]}',  # Invalid role
            'invalid json {',
            '',
            '{"messages": []}'  # Empty messages
        ]
        
        for invalid_line in invalid_examples:
            validation_errors = validate_jsonl_format([invalid_line])
            assert len(validation_errors) > 0, f"Invalid JSONL accepted: {invalid_line}"
    
    @given(st.lists(training_entry_strategy(), min_size=1, max_size=5))
    @settings(max_examples=30)
    def test_training_data_export_preserves_all_entries(self, training_entries: List[Dict[str, Any]]):
        """
        Property 16: Training data export preserves all valid entries
        For any list of training entries, all valid entries should be exported
        """
        exporter = TrainingDataExporter()
        jsonl_lines = exporter.export_entries_to_jsonl(training_entries)
        
        # Should have same number of lines as input entries
        assert len(jsonl_lines) == len(training_entries)
        
        # Each line should correspond to an entry
        for i, (entry, jsonl_line) in enumerate(zip(training_entries, jsonl_lines)):
            json_obj = json.loads(jsonl_line)
            
            # Should contain the original text
            user_content = json_obj['messages'][0]['content']
            assert entry['text'] in user_content
            
            # Should contain the bucket classification
            assistant_content = json_obj['messages'][1]['content']
            expected_bucket = entry['gold_output']['bucket']
            assert expected_bucket in assistant_content
    
    def test_training_data_export_handles_empty_input(self):
        """
        Property 16: Training data export handles empty input gracefully
        Empty input should produce empty output without errors
        """
        exporter = TrainingDataExporter()
        jsonl_lines = exporter.export_entries_to_jsonl([])
        
        assert isinstance(jsonl_lines, list)
        assert len(jsonl_lines) == 0
    
    @given(training_entry_strategy())
    @settings(max_examples=50)
    def test_training_example_includes_classification_prompt(self, training_entry: Dict[str, Any]):
        """
        Property 16: Training examples include proper classification prompt
        Each training example should include the classification prompt in user message
        """
        jsonl_entry = create_training_example(training_entry)
        json_obj = json.loads(jsonl_entry)
        
        user_content = json_obj['messages'][0]['content']
        
        # Should include classification instruction
        classification_keywords = [
            'classify', 'classification', 'determine', 'bearings',
            'explicit_bearings', 'abstract_bearings', 'no_bearings'
        ]
        
        # At least one classification keyword should be present
        assert any(keyword in user_content.lower() for keyword in classification_keywords), \
            f"No classification keywords found in: {user_content}"
        
        # Should include the actual text to classify
        assert training_entry['text'] in user_content
    
    @given(training_entry_strategy())
    @settings(max_examples=50)
    def test_assistant_response_format_is_consistent(self, training_entry: Dict[str, Any]):
        """
        Property 16: Assistant response format is consistent
        Assistant responses should follow a consistent format
        """
        jsonl_entry = create_training_example(training_entry)
        json_obj = json.loads(jsonl_entry)
        
        assistant_content = json_obj['messages'][1]['content']
        expected_bucket = training_entry['gold_output']['bucket']
        
        # Should contain the classification
        assert expected_bucket in assistant_content
        
        # Should follow expected format (e.g., "classification: explicit_bearings")
        expected_formats = [
            f"classification: {expected_bucket}",
            f"CLASSIFICATION: {expected_bucket}",
            expected_bucket
        ]
        
        format_found = any(fmt in assistant_content for fmt in expected_formats)
        assert format_found, f"Expected format not found in: {assistant_content}"
    
    def test_gold_dataset_export_produces_valid_training_format(self):
        """
        Property 16: Gold dataset export produces valid training format
        The gold dataset export should produce training data in valid JSONL format
        """
        # This tests the actual gold dataset export function
        try:
            training_data = export_gold_dataset_for_training()
            
            # Should return a list
            assert isinstance(training_data, list)
            
            # If we have data, test the format
            if training_data:
                exporter = TrainingDataExporter()
                jsonl_lines = exporter.export_entries_to_jsonl(training_data)
                
                # Should produce valid JSONL
                validation_errors = validate_jsonl_format(jsonl_lines)
                assert len(validation_errors) == 0, f"Gold dataset export invalid: {validation_errors}"
                
                # Each line should be valid
                for line in jsonl_lines:
                    json_obj = json.loads(line)
                    assert 'messages' in json_obj
                    assert len(json_obj['messages']) >= 2
                    
        except Exception as e:
            # If gold dataset is not available, that's okay for this test
            # We're testing the format compliance, not the data availability
            if "database" not in str(e).lower():
                pytest.fail(f"Gold dataset export failed unexpectedly: {e}")
    
    @given(st.lists(training_entry_strategy(), min_size=1, max_size=3))
    @settings(max_examples=20)
    def test_batch_export_maintains_order(self, training_entries: List[Dict[str, Any]]):
        """
        Property 16: Batch export maintains entry order
        Exported JSONL lines should correspond to input entries in order
        """
        exporter = TrainingDataExporter()
        jsonl_lines = exporter.export_entries_to_jsonl(training_entries)
        
        # Should maintain order
        for i, (entry, jsonl_line) in enumerate(zip(training_entries, jsonl_lines)):
            json_obj = json.loads(jsonl_line)
            user_content = json_obj['messages'][0]['content']
            
            # Should contain the text from the corresponding entry
            assert entry['text'] in user_content, \
                f"Entry {i} text not found in corresponding JSONL line"
    
    def test_training_data_format_matches_openai_specification(self):
        """
        Property 16: Training data format matches OpenAI specification
        The format should match OpenAI's fine-tuning requirements exactly
        """
        # Test with a known good example
        sample_entry = {
            'id': 'test_001',
            'text': 'North 45 degrees 30 minutes East 150.00 feet to an iron pin',
            'gold_output': {
                'bucket': 'explicit_bearings',
                'lines': [
                    {
                        'type': 'course',
                        'idx': 1,
                        'raw': 'North 45 degrees 30 minutes East 150.00 feet',
                        'cardinal_ns': 'North',
                        'degrees': 45,
                        'minutes': 30,
                        'seconds': 0,
                        'cardinal_ew': 'East',
                        'distance': 150.0,
                        'monument': 'iron pin',
                        'reference': None
                    }
                ]
            },
            'source_file': 'test.pdf',
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        jsonl_entry = create_training_example(sample_entry)
        json_obj = json.loads(jsonl_entry)
        
        # Must have exactly the structure OpenAI expects
        assert list(json_obj.keys()) == ['messages']
        assert len(json_obj['messages']) == 2
        
        # User message structure
        user_msg = json_obj['messages'][0]
        assert user_msg['role'] == 'user'
        assert 'content' in user_msg
        assert len(user_msg) == 2  # Only role and content
        
        # Assistant message structure
        assistant_msg = json_obj['messages'][1]
        assert assistant_msg['role'] == 'assistant'
        assert 'content' in assistant_msg
        assert len(assistant_msg) == 2  # Only role and content
        
        # Content should be strings
        assert isinstance(user_msg['content'], str)
        assert isinstance(assistant_msg['content'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])