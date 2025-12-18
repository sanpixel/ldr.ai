"""
GPT Fine-Tuning Integration for Legal Description Reader
Manages training data collection, JSONL export, and fine-tuning workflows
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from utils.auth import get_openai_client


@dataclass
class TrainingExample:
    """Represents a single training example for fine-tuning"""
    input_text: str
    expected_classification: str
    entry_id: Optional[str] = None
    source_file: Optional[str] = None
    created_at: Optional[str] = None


class TrainingDataExporter:
    """Handles export of training data to JSONL format for OpenAI fine-tuning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_entries_to_jsonl(self, training_entries: List[Dict[str, Any]]) -> List[str]:
        """
        Export training entries to JSONL format
        
        Args:
            training_entries: List of training data entries
            
        Returns:
            List of JSONL strings, one per entry
        """
        jsonl_lines = []
        
        for entry in training_entries:
            try:
                jsonl_line = create_training_example(entry)
                jsonl_lines.append(jsonl_line)
            except Exception as e:
                self.logger.error(f"Failed to create training example: {e}")
                continue
        
        return jsonl_lines
    
    def export_to_file(self, training_entries: List[Dict[str, Any]], filename: str) -> bool:
        """
        Export training data to JSONL file
        
        Args:
            training_entries: List of training data entries
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            jsonl_lines = self.export_entries_to_jsonl(training_entries)
            
            with open(filename, 'w', encoding='utf-8') as f:
                for line in jsonl_lines:
                    f.write(line + '\n')
            
            self.logger.info(f"Exported {len(jsonl_lines)} training examples to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export training data: {e}")
            return False


def create_training_example(training_entry: Dict[str, Any]) -> str:
    """
    Create a single JSONL training example from a training entry
    
    Args:
        training_entry: Dictionary containing text, gold_output, etc.
        
    Returns:
        JSONL string for this training example
    """
    text = training_entry.get('text', '')
    gold_output = training_entry.get('gold_output', {})
    bucket_classification = gold_output.get('bucket', 'no_bearings')
    
    # Create the user prompt for classification
    user_content = f"""Determine if bearings are present or not in the following legal description:
{text}"""
    
    # Create the assistant response with classification
    assistant_content = f"classification: {bucket_classification}"
    
    # Create the JSONL message structure
    jsonl_obj = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    
    return json.dumps(jsonl_obj, ensure_ascii=False)


def validate_jsonl_format(jsonl_lines: List[str]) -> List[str]:
    """
    Validate JSONL format for OpenAI fine-tuning compliance
    
    Args:
        jsonl_lines: List of JSONL strings to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    for i, line in enumerate(jsonl_lines):
        line_num = i + 1
        
        # Check if line is empty
        if not line.strip():
            errors.append(f"Line {line_num}: Empty line")
            continue
        
        # Check if valid JSON
        try:
            json_obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: Invalid JSON - {e}")
            continue
        
        # Check if has messages array
        if 'messages' not in json_obj:
            errors.append(f"Line {line_num}: Missing 'messages' field")
            continue
        
        messages = json_obj['messages']
        if not isinstance(messages, list):
            errors.append(f"Line {line_num}: 'messages' must be an array")
            continue
        
        if len(messages) == 0:
            errors.append(f"Line {line_num}: 'messages' array cannot be empty")
            continue
        
        # Validate each message
        for j, message in enumerate(messages):
            msg_num = j + 1
            
            if not isinstance(message, dict):
                errors.append(f"Line {line_num}, Message {msg_num}: Must be an object")
                continue
            
            # Check required fields
            if 'role' not in message:
                errors.append(f"Line {line_num}, Message {msg_num}: Missing 'role' field")
                continue
            
            if 'content' not in message:
                errors.append(f"Line {line_num}, Message {msg_num}: Missing 'content' field")
                continue
            
            # Validate role
            valid_roles = ['user', 'assistant', 'system']
            if message['role'] not in valid_roles:
                errors.append(f"Line {line_num}, Message {msg_num}: Invalid role '{message['role']}', must be one of {valid_roles}")
            
            # Validate content
            if not isinstance(message['content'], str):
                errors.append(f"Line {line_num}, Message {msg_num}: 'content' must be a string")
            elif len(message['content'].strip()) == 0:
                errors.append(f"Line {line_num}, Message {msg_num}: 'content' cannot be empty")
    
    return errors


def export_training_data_as_jsonl(training_data: List[Dict[str, Any]]) -> List[str]:
    """
    Export training data as JSONL format
    
    Args:
        training_data: List of training data dictionaries
        
    Returns:
        List of JSONL strings
    """
    exporter = TrainingDataExporter()
    return exporter.export_entries_to_jsonl(training_data)


class FineTuningDataPipeline:
    """
    Complete pipeline for collecting, processing, and exporting training data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exporter = TrainingDataExporter()
    
    def collect_training_data_from_classifications(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Collect training data from classification_data table
        
        Args:
            limit: Maximum number of entries to collect (None for all)
            
        Returns:
            List of training data entries
        """
        try:
            from utils.classification import load_all_classification_data
            
            classification_data = load_all_classification_data()
            training_entries = []
            
            # Apply limit if specified
            if limit:
                classification_data = classification_data[:limit]
            
            for entry in classification_data:
                reasoning_data = entry.get('reasoning_data', {})
                
                # Extract relevant fields
                text = reasoning_data.get('input_text', '')
                classification = reasoning_data.get('classification', 'no_bearings')
                
                if text and classification:
                    # Create gold output structure
                    gold_output = {
                        'bucket': classification,
                        'lines': []  # Classification data doesn't include extracted lines
                    }
                    
                    training_entry = {
                        'id': entry.get('id'),
                        'text': text,
                        'gold_output': gold_output,
                        'source_file': reasoning_data.get('filename', 'unknown'),
                        'created_at': entry.get('created_at'),
                        'confidence': reasoning_data.get('confidence', 'medium'),
                        'user_email': reasoning_data.get('user_email', 'anonymous')
                    }
                    training_entries.append(training_entry)
            
            self.logger.info(f"Collected {len(training_entries)} training entries from classification data")
            return training_entries
            
        except Exception as e:
            self.logger.error(f"Failed to collect training data from classifications: {e}")
            return []
    
    def collect_training_data_from_gold_dataset(self) -> List[Dict[str, Any]]:
        """
        Collect training data from gold dataset
        
        Returns:
            List of training data entries
        """
        try:
            from utils.gold_dataset import export_gold_dataset_for_training
            
            gold_data = export_gold_dataset_for_training()
            self.logger.info(f"Collected {len(gold_data)} training entries from gold dataset")
            return gold_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect training data from gold dataset: {e}")
            return []
    
    def merge_training_sources(self, 
                              include_classifications: bool = True,
                              include_gold_dataset: bool = True,
                              classification_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Merge training data from multiple sources
        
        Args:
            include_classifications: Include data from classification_data table
            include_gold_dataset: Include data from gold dataset
            classification_limit: Limit on classification data entries
            
        Returns:
            Combined list of training data entries
        """
        all_training_data = []
        
        if include_classifications:
            classification_data = self.collect_training_data_from_classifications(classification_limit)
            all_training_data.extend(classification_data)
        
        if include_gold_dataset:
            gold_data = self.collect_training_data_from_gold_dataset()
            all_training_data.extend(gold_data)
        
        # Remove duplicates based on text content
        seen_texts = set()
        unique_training_data = []
        
        for entry in all_training_data:
            text = entry.get('text', '')
            text_hash = hash(text)
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_training_data.append(entry)
        
        self.logger.info(f"Merged training data: {len(all_training_data)} total, {len(unique_training_data)} unique")
        return unique_training_data
    
    def export_training_data_to_file(self, 
                                   output_filename: str,
                                   include_classifications: bool = True,
                                   include_gold_dataset: bool = True,
                                   classification_limit: Optional[int] = None) -> bool:
        """
        Export training data to JSONL file
        
        Args:
            output_filename: Output file path
            include_classifications: Include classification data
            include_gold_dataset: Include gold dataset
            classification_limit: Limit on classification entries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Collect training data
            training_data = self.merge_training_sources(
                include_classifications=include_classifications,
                include_gold_dataset=include_gold_dataset,
                classification_limit=classification_limit
            )
            
            if not training_data:
                self.logger.warning("No training data collected")
                return False
            
            # Export to file
            success = self.exporter.export_to_file(training_data, output_filename)
            
            if success:
                self.logger.info(f"Successfully exported {len(training_data)} training examples to {output_filename}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to export training data: {e}")
            return False
    
    def create_fine_tuning_dataset(self, 
                                 output_filename: str = "training_data.jsonl",
                                 min_examples_per_class: int = 10) -> Tuple[bool, Dict[str, int]]:
        """
        Create a balanced fine-tuning dataset
        
        Args:
            output_filename: Output file path
            min_examples_per_class: Minimum examples per classification class
            
        Returns:
            Tuple of (success, class_counts)
        """
        try:
            # Collect all available training data
            all_training_data = self.merge_training_sources()
            
            # Group by classification
            class_groups = {}
            for entry in all_training_data:
                classification = entry['gold_output']['bucket']
                if classification not in class_groups:
                    class_groups[classification] = []
                class_groups[classification].append(entry)
            
            # Balance the dataset
            balanced_data = []
            class_counts = {}
            
            for classification, entries in class_groups.items():
                # Take up to min_examples_per_class from each class
                selected_entries = entries[:min_examples_per_class] if len(entries) >= min_examples_per_class else entries
                balanced_data.extend(selected_entries)
                class_counts[classification] = len(selected_entries)
            
            if not balanced_data:
                self.logger.warning("No balanced training data available")
                return False, {}
            
            # Export balanced dataset
            success = self.exporter.export_to_file(balanced_data, output_filename)
            
            if success:
                self.logger.info(f"Created balanced dataset with {len(balanced_data)} examples: {class_counts}")
            
            return success, class_counts
            
        except Exception as e:
            self.logger.error(f"Failed to create fine-tuning dataset: {e}")
            return False, {}


def collect_training_data_from_classifications() -> List[Dict[str, Any]]:
    """
    Collect training data from classification_data table
    
    Returns:
        List of training data entries
    """
    pipeline = FineTuningDataPipeline()
    return pipeline.collect_training_data_from_classifications()


def upload_training_data_to_openai(jsonl_filename: str) -> Optional[str]:
    """
    Upload training data file to OpenAI for fine-tuning
    
    Args:
        jsonl_filename: Path to JSONL training data file
        
    Returns:
        File ID if successful, None otherwise
    """
    try:
        client = get_openai_client()
        if not client:
            logging.error("OpenAI client not available")
            return None
        
        # Upload file
        with open(jsonl_filename, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        logging.info(f"Uploaded training data file: {file_id}")
        return file_id
        
    except Exception as e:
        logging.error(f"Failed to upload training data: {e}")
        return None


def create_fine_tuning_job(training_file_id: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """
    Create a fine-tuning job with OpenAI
    
    Args:
        training_file_id: ID of uploaded training file
        model: Base model to fine-tune
        
    Returns:
        Fine-tuning job ID if successful, None otherwise
    """
    try:
        client = get_openai_client()
        if not client:
            logging.error("OpenAI client not available")
            return None
        
        # Create fine-tuning job
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model
        )
        
        job_id = response.id
        logging.info(f"Created fine-tuning job: {job_id}")
        return job_id
        
    except Exception as e:
        logging.error(f"Failed to create fine-tuning job: {e}")
        return None


def get_fine_tuning_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of a fine-tuning job
    
    Args:
        job_id: Fine-tuning job ID
        
    Returns:
        Job status dictionary if successful, None otherwise
    """
    try:
        client = get_openai_client()
        if not client:
            logging.error("OpenAI client not available")
            return None
        
        response = client.fine_tuning.jobs.retrieve(job_id)
        
        return {
            'id': response.id,
            'status': response.status,
            'model': response.model,
            'fine_tuned_model': response.fine_tuned_model,
            'created_at': response.created_at,
            'finished_at': response.finished_at,
            'training_file': response.training_file,
            'validation_file': response.validation_file,
            'result_files': response.result_files
        }
        
    except Exception as e:
        logging.error(f"Failed to get fine-tuning job status: {e}")
        return None


def store_training_data_entry(text: str, 
                            bucket_classification: str,
                            extracted_lines: List[Dict[str, Any]] = None,
                            source_file: str = "unknown",
                            user_email: str = "anonymous") -> bool:
    """
    Store a training data entry for future fine-tuning
    
    Args:
        text: Raw legal description text
        bucket_classification: Final bucket classification
        extracted_lines: Extracted line data (optional)
        source_file: Source filename
        user_email: User who processed this
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from utils.classification import save_classification_data
        
        # Create training data entry in the format expected by classification_data table
        training_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_text': text,
            'classification': bucket_classification,
            'confidence': 'high',  # Assume high confidence for stored training data
            'filename': source_file,
            'user_email': user_email,
            'extracted_lines': extracted_lines or [],
            'training_data': True,  # Flag to identify this as training data
            'bucket_classification': bucket_classification,
            'processing_method': 'training_pipeline'
        }
        
        # Save to classification_data table (maintains backward compatibility)
        success = save_classification_data(training_entry)
        
        if success:
            logging.info(f"Stored training data entry for classification: {bucket_classification}")
        
        return success
        
    except Exception as e:
        logging.error(f"Failed to store training data entry: {e}")
        return False


def get_training_data_statistics() -> Dict[str, Any]:
    """
    Get statistics about available training data
    
    Returns:
        Dictionary with training data statistics
    """
    try:
        pipeline = FineTuningDataPipeline()
        
        # Collect data from both sources
        classification_data = pipeline.collect_training_data_from_classifications()
        gold_data = pipeline.collect_training_data_from_gold_dataset()
        
        # Count by classification type
        classification_counts = {}
        total_entries = 0
        
        for data_source, source_name in [(classification_data, 'classification_data'), (gold_data, 'gold_dataset')]:
            for entry in data_source:
                bucket = entry['gold_output']['bucket']
                if bucket not in classification_counts:
                    classification_counts[bucket] = {'classification_data': 0, 'gold_dataset': 0, 'total': 0}
                
                classification_counts[bucket][source_name] += 1
                classification_counts[bucket]['total'] += 1
                total_entries += 1
        
        return {
            'total_entries': total_entries,
            'classification_data_entries': len(classification_data),
            'gold_dataset_entries': len(gold_data),
            'classification_counts': classification_counts,
            'ready_for_fine_tuning': total_entries >= 50,  # OpenAI minimum recommendation
            'recommended_min_per_class': 10
        }
        
    except Exception as e:
        logging.error(f"Failed to get training data statistics: {e}")
        return {
            'total_entries': 0,
            'classification_data_entries': 0,
            'gold_dataset_entries': 0,
            'classification_counts': {},
            'ready_for_fine_tuning': False,
            'error': str(e)
        }


def export_for_openai_fine_tuning(output_filename: str = "ldr_training_data.jsonl") -> Tuple[bool, str]:
    """
    Export training data in OpenAI fine-tuning format
    
    Args:
        output_filename: Output JSONL filename
        
    Returns:
        Tuple of (success, message)
    """
    try:
        pipeline = FineTuningDataPipeline()
        
        # Create balanced dataset
        success, class_counts = pipeline.create_fine_tuning_dataset(output_filename)
        
        if success:
            total_examples = sum(class_counts.values())
            message = f"Successfully exported {total_examples} training examples to {output_filename}. Class distribution: {class_counts}"
            
            # Validate the exported file
            with open(output_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            validation_errors = validate_jsonl_format(lines)
            if validation_errors:
                message += f" Warning: {len(validation_errors)} validation errors found."
            else:
                message += " File validated successfully for OpenAI fine-tuning."
            
            return True, message
        else:
            return False, "Failed to create training dataset"
            
    except Exception as e:
        error_msg = f"Failed to export training data: {e}"
        logging.error(error_msg)
        return False, error_msg