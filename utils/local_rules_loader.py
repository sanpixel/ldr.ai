"""
Local file-based rules loader as fallback when GCS is not available
"""

import json
import os
from typing import Optional
from utils.rules import RulesConfig, RulesProcessor
import streamlit as st

class LocalRulesLoader:
    """Load rules from local rules.json file"""
    
    def __init__(self, rules_file: str = "rules.json"):
        self.rules_file = rules_file
        self._cached_processor = None
    
    def get_current_processor(self) -> Optional[RulesProcessor]:
        """Get current rules processor from local file"""
        try:
            if not os.path.exists(self.rules_file):
                if st.session_state.get('debug_enabled', False):
                    st.warning(f"🔍 DEBUG: Rules file not found: {self.rules_file}")
                return None
            
            with open(self.rules_file, 'r') as f:
                rules_data = json.load(f)
            
            rules_config = RulesConfig.from_dict(rules_data)
            processor = RulesProcessor(rules_config)
            
            if st.session_state.get('debug_enabled', False):
                st.write(f"🔍 DEBUG: Loaded {len(rules_config.rules)} rules from {self.rules_file}")
            
            return processor
            
        except Exception as e:
            if st.session_state.get('debug_enabled', False):
                st.error(f"🔍 DEBUG: Failed to load local rules: {str(e)}")
            return None

# Global instance
_local_loader = LocalRulesLoader()

def get_current_processor() -> Optional[RulesProcessor]:
    """Get current rules processor (local fallback)"""
    return _local_loader.get_current_processor()