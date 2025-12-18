#!/usr/bin/env python3
"""
CLI script to export training data for OpenAI fine-tuning
"""

import argparse
import sys
from utils.fine_tuning import (
    FineTuningDataPipeline, 
    export_for_openai_fine_tuning,
    get_training_data_statistics
)


def main():
    parser = argparse.ArgumentParser(description='Export training data for OpenAI fine-tuning')
    parser.add_argument('--output', '-o', default='ldr_training_data.jsonl', 
                       help='Output JSONL filename (default: ldr_training_data.jsonl)')
    parser.add_argument('--stats', action='store_true', 
                       help='Show training data statistics')
    parser.add_argument('--classification-limit', type=int, 
                       help='Limit number of classification data entries')
    parser.add_argument('--min-per-class', type=int, default=10,
                       help='Minimum examples per classification class (default: 10)')
    
    args = parser.parse_args()
    
    if args.stats:
        print("Getting training data statistics...")
        stats = get_training_data_statistics()
        
        print(f"\nTraining Data Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Classification data entries: {stats['classification_data_entries']}")
        print(f"Gold dataset entries: {stats['gold_dataset_entries']}")
        print(f"Ready for fine-tuning: {stats['ready_for_fine_tuning']}")
        
        print(f"\nClassification breakdown:")
        for classification, counts in stats['classification_counts'].items():
            print(f"  {classification}: {counts['total']} total "
                  f"({counts['classification_data']} from classification_data, "
                  f"{counts['gold_dataset']} from gold_dataset)")
        
        if not stats['ready_for_fine_tuning']:
            print(f"\nNote: OpenAI recommends at least 50 total examples for fine-tuning.")
        
        return
    
    print(f"Exporting training data to {args.output}...")
    
    try:
        pipeline = FineTuningDataPipeline()
        
        # Create balanced dataset
        success, class_counts = pipeline.create_fine_tuning_dataset(
            output_filename=args.output,
            min_examples_per_class=args.min_per_class
        )
        
        if success:
            total_examples = sum(class_counts.values())
            print(f"✅ Successfully exported {total_examples} training examples")
            print(f"📊 Class distribution: {class_counts}")
            print(f"📁 Output file: {args.output}")
            
            # Show next steps
            print(f"\nNext steps:")
            print(f"1. Review the exported file: {args.output}")
            print(f"2. Upload to OpenAI: openai api fine_tuning.jobs.create -t {args.output} -m gpt-3.5-turbo")
            print(f"3. Monitor training progress in OpenAI dashboard")
            
        else:
            print("❌ Failed to export training data")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()