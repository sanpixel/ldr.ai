"""
Gold dataset management for Legal Description Reader
Handles creation, storage, and management of verified correct outputs for testing
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from utils.auth import get_supabase_client
from utils.schema import SchemaOutput, validate_json_schema
from datetime import datetime
import json
import hashlib


def save_gold_entry(text: str, gold_output: SchemaOutput, 
                   source_file: str = None, verified_by: str = None) -> bool:
    """
    Save a gold dataset entry with verified correct output
    
    Args:
        text: Original legal description text
        gold_output: Manually verified correct SchemaOutput
        source_file: Optional source filename
        verified_by: Optional user who verified the output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Validate the gold output schema
        validation_errors = gold_output.validate_schema()
        if validation_errors:
            st.error(f"Gold output validation failed: {validation_errors}")
            return False
        
        # Generate unique ID based on text hash
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Prepare insert data
        insert_data = {
            'id': text_hash,
            'text': text,
            'gold_output': gold_output.to_dict(),
            'source_file': source_file,
            'verified_by': verified_by or st.session_state.get('user', {}).get('email', 'system')
        }
        
        # Insert or update the gold entry
        result = supabase.table('gold_dataset').upsert(insert_data).execute()
        
        if result.data:
            return True
        else:
            st.error("Failed to save gold dataset entry")
            return False
            
    except Exception as e:
        st.error(f"Error saving gold dataset entry: {str(e)}")
        return False


def load_all_gold_entries() -> List[Dict[str, Any]]:
    """
    Load all gold dataset entries
    
    Returns:
        List of gold dataset entries
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('gold_dataset')\
            .select('*')\
            .order('created_at', desc=True)\
            .execute()
        
        return result.data or []
        
    except Exception as e:
        st.error(f"Error loading gold dataset: {str(e)}")
        return []


def get_gold_entry_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific gold dataset entry by ID
    
    Args:
        entry_id: Entry identifier (text hash)
    
    Returns:
        Dict with entry data or None if not found
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('gold_dataset')\
            .select('*')\
            .eq('id', entry_id)\
            .execute()
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception as e:
        st.error(f"Error getting gold entry {entry_id}: {str(e)}")
        return None


def get_gold_entry_by_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Get gold dataset entry by matching text
    
    Args:
        text: Legal description text to match
    
    Returns:
        Dict with entry data or None if not found
    """
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return get_gold_entry_by_id(text_hash)


def update_gold_entry(entry_id: str, gold_output: SchemaOutput, 
                     verified_by: str = None) -> bool:
    """
    Update an existing gold dataset entry with corrected output
    
    Args:
        entry_id: Entry ID to update
        gold_output: New verified correct output
        verified_by: User making the correction
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Validate the gold output schema
        validation_errors = gold_output.validate_schema()
        if validation_errors:
            st.error(f"Gold output validation failed: {validation_errors}")
            return False
        
        update_data = {
            'gold_output': gold_output.to_dict(),
            'verified_by': verified_by or st.session_state.get('user', {}).get('email', 'system'),
            'updated_at': datetime.now().isoformat()
        }
        
        result = supabase.table('gold_dataset')\
            .update(update_data)\
            .eq('id', entry_id)\
            .execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        st.error(f"Error updating gold entry: {str(e)}")
        return False


def create_gold_from_extraction(text: str, original_output: SchemaOutput, 
                               corrected_output: SchemaOutput,
                               source_file: str = None) -> bool:
    """
    Create gold dataset entry from original and corrected extraction results
    Preserves both original (potentially incorrect) and corrected outputs
    
    Args:
        text: Original legal description text
        original_output: Original extractor output
        corrected_output: Manually corrected output
        source_file: Optional source filename
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate both outputs
        original_errors = original_output.validate_schema()
        corrected_errors = corrected_output.validate_schema()
        
        if original_errors:
            st.warning(f"Original output has validation errors: {original_errors}")
        
        if corrected_errors:
            st.error(f"Corrected output validation failed: {corrected_errors}")
            return False
        
        # Create enhanced gold output with both versions
        enhanced_gold_output = corrected_output.to_dict()
        enhanced_gold_output['metadata'] = {
            'original_output': original_output.to_dict(),
            'correction_applied': True,
            'original_had_errors': len(original_errors) > 0,
            'corrected_at': datetime.now().isoformat()
        }
        
        # Convert back to SchemaOutput for validation
        final_gold_output = SchemaOutput.from_dict(enhanced_gold_output)
        
        return save_gold_entry(text, final_gold_output, source_file)
        
    except Exception as e:
        st.error(f"Error creating gold entry from extraction: {str(e)}")
        return False


def get_gold_dataset_stats() -> Dict[str, Any]:
    """
    Get statistics about the gold dataset
    
    Returns:
        Dict with dataset statistics
    """
    try:
        supabase = get_supabase_client()
        
        # Get total count
        total_result = supabase.table('gold_dataset').select('id', count='exact').execute()
        total_count = total_result.count or 0
        
        # Get all entries for detailed stats
        all_entries = load_all_gold_entries()
        
        # Calculate statistics
        bucket_counts = {}
        source_file_counts = {}
        verifier_counts = {}
        
        for entry in all_entries:
            gold_output = entry.get('gold_output', {})
            bucket = gold_output.get('bucket', 'unknown')
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            
            source_file = entry.get('source_file', 'unknown')
            source_file_counts[source_file] = source_file_counts.get(source_file, 0) + 1
            
            verifier = entry.get('verified_by', 'unknown')
            verifier_counts[verifier] = verifier_counts.get(verifier, 0) + 1
        
        return {
            'total_entries': total_count,
            'bucket_distribution': bucket_counts,
            'source_file_distribution': source_file_counts,
            'verifier_distribution': verifier_counts,
            'last_updated': max([entry.get('created_at', '') for entry in all_entries]) if all_entries else None
        }
        
    except Exception as e:
        st.error(f"Error getting gold dataset stats: {str(e)}")
        return {
            'total_entries': 0,
            'bucket_distribution': {},
            'source_file_distribution': {},
            'verifier_distribution': {},
            'last_updated': None
        }


def validate_gold_dataset_integrity() -> Dict[str, Any]:
    """
    Validate the integrity of the entire gold dataset
    
    Returns:
        Dict with validation results
    """
    try:
        all_entries = load_all_gold_entries()
        
        validation_results = {
            'total_entries': len(all_entries),
            'valid_entries': 0,
            'invalid_entries': 0,
            'validation_errors': [],
            'schema_compliance': True
        }
        
        for entry in all_entries:
            entry_id = entry.get('id', 'unknown')
            
            # Check required fields
            if not entry.get('text'):
                validation_results['validation_errors'].append(f"Entry {entry_id}: Missing text field")
                validation_results['invalid_entries'] += 1
                continue
            
            if not entry.get('gold_output'):
                validation_results['validation_errors'].append(f"Entry {entry_id}: Missing gold_output field")
                validation_results['invalid_entries'] += 1
                continue
            
            # Validate gold output schema
            try:
                gold_output_dict = entry['gold_output']
                schema_errors = validate_json_schema(gold_output_dict)
                
                if schema_errors:
                    validation_results['validation_errors'].append(f"Entry {entry_id}: Schema errors: {schema_errors}")
                    validation_results['invalid_entries'] += 1
                    validation_results['schema_compliance'] = False
                else:
                    validation_results['valid_entries'] += 1
                    
            except Exception as e:
                validation_results['validation_errors'].append(f"Entry {entry_id}: Schema validation exception: {e}")
                validation_results['invalid_entries'] += 1
                validation_results['schema_compliance'] = False
        
        return validation_results
        
    except Exception as e:
        return {
            'total_entries': 0,
            'valid_entries': 0,
            'invalid_entries': 0,
            'validation_errors': [f"Validation failed: {e}"],
            'schema_compliance': False
        }


def export_gold_dataset_for_training() -> List[Dict[str, Any]]:
    """
    Export gold dataset in format suitable for training/fine-tuning
    
    Returns:
        List of training examples
    """
    try:
        all_entries = load_all_gold_entries()
        training_data = []
        
        for entry in all_entries:
            text = entry.get('text', '')
            gold_output = entry.get('gold_output', {})
            
            if text and gold_output:
                training_example = {
                    'input_text': text,
                    'expected_output': gold_output,
                    'bucket_classification': gold_output.get('bucket', 'no_bearings'),
                    'entry_id': entry.get('id'),
                    'source_file': entry.get('source_file'),
                    'created_at': entry.get('created_at')
                }
                training_data.append(training_example)
        
        return training_data
        
    except Exception as e:
        st.error(f"Error exporting gold dataset: {str(e)}")
        return []


def collect_real_descriptions_for_gold(limit: int = 40) -> List[Dict[str, Any]]:
    """
    Collect real legal descriptions from classification_data for gold dataset creation
    
    Args:
        limit: Maximum number of descriptions to collect
    
    Returns:
        List of candidate descriptions with their classification data
    """
    try:
        supabase = get_supabase_client()
        
        # Get diverse set of classifications
        result = supabase.table('classification_data')\
            .select('reasoning_data')\
            .order('inserted_at', desc=True)\
            .limit(limit * 2)\
            .execute()  # Get more than needed to allow filtering
        
        candidates = []
        seen_hashes = set()
        
        for row in result.data or []:
            reasoning_data = row.get('reasoning_data', {})
            input_text = reasoning_data.get('input_text', '')
            
            if not input_text or len(input_text) < 50:  # Skip very short texts
                continue
            
            # Avoid duplicates
            text_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            
            # Check if already in gold dataset
            existing_gold = get_gold_entry_by_text(input_text)
            if existing_gold:
                continue
            
            candidate = {
                'text': input_text,
                'classification': reasoning_data.get('classification', 'unknown'),
                'confidence': reasoning_data.get('confidence', 'unknown'),
                'filename': reasoning_data.get('filename', 'unknown'),
                'reasoning_data': reasoning_data,
                'text_hash': text_hash
            }
            candidates.append(candidate)
            
            if len(candidates) >= limit:
                break
        
        return candidates
        
    except Exception as e:
        st.error(f"Error collecting descriptions for gold dataset: {str(e)}")
        return []


class GoldDatasetManager:
    """
    High-level manager for gold dataset operations
    """
    
    def __init__(self):
        self.stats = None
        self.last_stats_update = None
    
    def get_cached_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached statistics, refreshing if needed"""
        now = datetime.now()
        
        if (force_refresh or 
            self.stats is None or 
            self.last_stats_update is None or
            (now - self.last_stats_update).total_seconds() > 300):  # 5 minute cache
            
            self.stats = get_gold_dataset_stats()
            self.last_stats_update = now
        
        return self.stats
    
    def is_ready_for_harness(self) -> bool:
        """Check if gold dataset has enough entries for harness testing"""
        stats = self.get_cached_stats()
        return stats.get('total_entries', 0) >= 10  # Minimum for meaningful testing
    
    def get_recommended_additions(self) -> List[str]:
        """Get recommendations for improving gold dataset coverage"""
        stats = self.get_cached_stats()
        recommendations = []
        
        total = stats.get('total_entries', 0)
        if total < 40:
            recommendations.append(f"Add more entries (current: {total}, target: 40)")
        
        bucket_dist = stats.get('bucket_distribution', {})
        for bucket_type in ['explicit_bearings', 'abstract_bearings', 'no_bearings']:
            count = bucket_dist.get(bucket_type, 0)
            if count < 10:
                recommendations.append(f"Add more {bucket_type} examples (current: {count})")
        
        return recommendations
    
    def create_from_candidates(self, candidates: List[Dict[str, Any]]) -> int:
        """
        Create gold entries from candidate descriptions
        Returns number of entries created
        """
        created_count = 0
        
        for candidate in candidates:
            try:
                # This would typically involve running the current extractor
                # and then allowing manual correction
                # For now, we'll create a placeholder entry
                
                text = candidate['text']
                
                # Create a basic schema output based on classification
                classification = candidate.get('classification', 'no_bearings')
                
                if classification == 'explicit_bearings':
                    # Would run actual extraction here
                    gold_output = SchemaOutput(bucket='explicit_bearings', lines=[])
                elif classification == 'abstract_bearings':
                    gold_output = SchemaOutput(bucket='abstract_bearings', lines=[])
                else:
                    gold_output = SchemaOutput(bucket='no_bearings', lines=[])
                
                if save_gold_entry(
                    text=text,
                    gold_output=gold_output,
                    source_file=candidate.get('filename'),
                    verified_by='auto_import'
                ):
                    created_count += 1
                    
            except Exception as e:
                st.warning(f"Failed to create gold entry: {e}")
                continue
        
        return created_count