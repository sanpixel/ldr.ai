#!/usr/bin/env python3
"""
Upload rules.json to GCS for the first time
"""

import sys
import os
sys.path.append('.')

from utils.gcs_rules import GCSRulesManager
from utils.rules import RulesConfig
import json

def main():
    print("🔄 Uploading rules.json to GCS...")
    
    try:
        # Load local rules file
        with open('gcs_rules_current.json', 'r') as f:
            rules_data = json.load(f)
        
        # Create RulesConfig object
        rules_config = RulesConfig.from_dict(rules_data)
        print(f"✅ Loaded {len(rules_data.get('rules', []))} rules from gcs_rules_current.json")
        
        # Create GCS manager
        manager = GCSRulesManager()
        
        # Upload as first version
        version_id = manager.save_rules_version(rules_config)
        print(f"✅ Uploaded rules as version: {version_id}")
        
        print("✅ Rules automatically set as current during save_rules_version")
            
        print("🎉 Rules upload complete!")
        
    except Exception as e:
        print(f"❌ Error uploading rules: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())