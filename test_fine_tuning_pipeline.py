#!/usr/bin/env python3
"""
Test script for fine-tuning pipeline functionality
"""

import json
import os
from utils.fine_tuning import (
    FineTuningDataPipeline,
    TrainingDataExporter,
    create_training_example,
    validate_jsonl_format
)


def test_training_data_pipeline():
    """Test the fine-tuning data pipeline with mock data"""
    
    print("Testing Fine-Tuning Data Pipeline...")
    
    # Create mock training data
    mock_training_data = [
        {
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
            'source_file': 'test1.pdf',
            'created_at': '2024-01-01T00:00:00Z'
        },
        {
            'id': 'test_002',
            'text': 'Lot 5, Block 3, Happy Valley Subdivision as recorded in Plat Book 42',
            'gold_output': {
                'bucket': 'external_ref',
                'lines': []
            },
            'source_file': 'test2.pdf',
            'created_at': '2024-01-02T00:00:00Z'
        },
        {
            'id': 'test_003',
            'text': 'Following the existing fence line in a northerly direction',
            'gold_output': {
                'bucket': 'abstract_bearings',
                'lines': [
                    {
                        'type': 'ref_segment',
                        'idx': 1,
                        'raw': 'Following the existing fence line in a northerly direction',
                        'cardinal_ns': None,
                        'degrees': None,
                        'minutes': None,
                        'seconds': None,
                        'cardinal_ew': None,
                        'distance': None,
                        'monument': 'fence line',
                        'reference': 'northerly direction'
                    }
                ]
            },
            'source_file': 'test3.pdf',
            'created_at': '2024-01-03T00:00:00Z'
        }
    ]
    
    # Test TrainingDataExporter
    print("\n1. Testing TrainingDataExporter...")
    exporter = TrainingDataExporter()
    jsonl_lines = exporter.export_entries_to_jsonl(mock_training_data)
    
    print(f"   ✅ Exported {len(jsonl_lines)} JSONL lines")
    
    # Test JSONL validation
    print("\n2. Testing JSONL validation...")
    validation_errors = validate_jsonl_format(jsonl_lines)
    
    if validation_errors:
        print(f"   ❌ Validation errors: {validation_errors}")
        return False
    else:
        print("   ✅ JSONL format validation passed")
    
    # Test individual training example creation
    print("\n3. Testing individual training example creation...")
    for i, entry in enumerate(mock_training_data):
        jsonl_entry = create_training_example(entry)
        
        try:
            json_obj = json.loads(jsonl_entry)
            assert 'messages' in json_obj
            assert len(json_obj['messages']) == 2
            assert json_obj['messages'][0]['role'] == 'user'
            assert json_obj['messages'][1]['role'] == 'assistant'
            
            # Check that text is included
            user_content = json_obj['messages'][0]['content']
            assert entry['text'] in user_content
            
            # Check that classification is included
            assistant_content = json_obj['messages'][1]['content']
            expected_bucket = entry['gold_output']['bucket']
            assert expected_bucket in assistant_content
            
            print(f"   ✅ Training example {i+1} created successfully")
            
        except Exception as e:
            print(f"   ❌ Training example {i+1} failed: {e}")
            return False
    
    # Test file export
    print("\n4. Testing file export...")
    test_filename = "test_training_export.jsonl"
    
    try:
        success = exporter.export_to_file(mock_training_data, test_filename)
        
        if success:
            print(f"   ✅ Successfully exported to {test_filename}")
            
            # Verify file contents
            with open(test_filename, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
            
            if len(file_lines) == len(mock_training_data):
                print(f"   ✅ File contains correct number of lines ({len(file_lines)})")
            else:
                print(f"   ❌ File line count mismatch: expected {len(mock_training_data)}, got {len(file_lines)}")
                return False
            
            # Clean up test file
            os.remove(test_filename)
            print(f"   ✅ Cleaned up test file")
            
        else:
            print("   ❌ File export failed")
            return False
            
    except Exception as e:
        print(f"   ❌ File export error: {e}")
        return False
    
    # Test pipeline class
    print("\n5. Testing FineTuningDataPipeline...")
    pipeline = FineTuningDataPipeline()
    
    # Test merge function with mock data (simulating empty database)
    merged_data = pipeline.merge_training_sources(
        include_classifications=False,  # Skip database calls
        include_gold_dataset=False      # Skip database calls
    )
    
    print(f"   ✅ Pipeline merge completed (empty result expected): {len(merged_data)} entries")
    
    print("\n🎉 All tests passed! Fine-tuning pipeline is working correctly.")
    return True


def show_sample_output():
    """Show sample JSONL output"""
    print("\n" + "="*60)
    print("SAMPLE JSONL OUTPUT")
    print("="*60)
    
    sample_entry = {
        'id': 'sample',
        'text': 'North 45 degrees 30 minutes East 150.00 feet to an iron pin',
        'gold_output': {
            'bucket': 'explicit_bearings',
            'lines': []
        },
        'source_file': 'sample.pdf',
        'created_at': '2024-01-01T00:00:00Z'
    }
    
    jsonl_output = create_training_example(sample_entry)
    json_obj = json.loads(jsonl_output)
    
    print(json.dumps(json_obj, indent=2))
    print("\nThis format is compatible with OpenAI fine-tuning requirements.")


if __name__ == "__main__":
    success = test_training_data_pipeline()
    
    if success:
        show_sample_output()
    else:
        print("\n❌ Tests failed!")
        exit(1)