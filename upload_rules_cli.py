#!/usr/bin/env python3

import json
from google.cloud import storage
import os

def upload_rules():
    try:
        # Load rules from local file
        with open('gcs_rules_current.json', 'r') as f:
            rules_data = json.load(f)

        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket('ldr-ai')

        # Upload to GCS
        blob = bucket.blob('rules/current.json')
        blob.upload_from_string(
            json.dumps(rules_data, indent=2),
            content_type='application/json'
        )

        print('✅ Rules uploaded to GCS successfully')
        return True
        
    except Exception as e:
        print(f'❌ Error uploading rules: {e}')
        return False

if __name__ == '__main__':
    upload_rules()