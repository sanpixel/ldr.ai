"""
Classification database operations for Legal Description Reader
Handles saving, loading, and managing classification reasoning data in Supabase
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from utils.auth import get_supabase_client
import json
from datetime import datetime


def save_classification_data(classification_entry: Dict[str, Any], pdf_preview_bytes: Optional[bytes] = None) -> bool:
    """
    Save a single classification entry to the database
    
    Args:
        classification_entry: Dictionary containing classification data with keys:
            - classification: str (explicit_bearings, abstract_bearings, external_ref)
            - confidence: str (high, medium, low)
            - input_text: str
            - reasoning: str
            - evidence: str (optional)
            - alternatives: str (optional)
            - timestamp: str (ISO format)
        pdf_preview_bytes: Optional bytes of the PDF preview image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare insert data
        insert_data = {'reasoning_data': classification_entry}
        
        # Add pdf_preview if provided (encode as base64 for JSON compatibility)
        if pdf_preview_bytes:
            import base64
            insert_data['pdf_preview'] = base64.b64encode(pdf_preview_bytes).decode('utf-8')
        
        # Insert the classification data
        result = supabase.table('classification_data').insert(insert_data).execute()
        
        if result.data:
            return True
        else:
            st.error("Failed to save classification data")
            return False
            
    except Exception as e:
        st.error(f"Error saving classification data: {str(e)}")
        return False


def load_all_classification_data() -> List[Dict[str, Any]]:
    """
    Load all classification data from the database
    
    Returns:
        List[Dict]: List of classification entries, empty list if error
    """
    try:
        supabase = get_supabase_client()
        
        # Query all classification data, ordered by insertion time (newest first)
        result = supabase.table('classification_data')\
            .select('reasoning_data')\
            .order('inserted_at', desc=True)\
            .execute()
        
        if result.data:
            # Extract the reasoning_data from each row
            return [row['reasoning_data'] for row in result.data]
        else:
            return []
            
    except Exception as e:
        st.error(f"Error loading classification data: {str(e)}")
        return []


def get_filtered_classification_data(
    classification_filter: Optional[str] = None,
    confidence_filter: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get filtered classification data from database
    
    Args:
        classification_filter: Filter by classification type (optional)
        confidence_filter: Filter by confidence level (optional) 
        limit: Maximum number of results (optional)
    
    Returns:
        List[Dict]: Filtered classification entries
    """
    try:
        supabase = get_supabase_client()
        
        # Start with base query
        query = supabase.table('classification_data').select('reasoning_data')
        
        # Apply filters using the generated columns
        if classification_filter and classification_filter != "All":
            query = query.eq('classification', classification_filter)
            
        if confidence_filter and confidence_filter != "All":
            query = query.eq('confidence', confidence_filter)
        
        # Order and limit
        query = query.order('inserted_at', desc=True)
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        
        if result.data:
            return [row['reasoning_data'] for row in result.data]
        else:
            return []
            
    except Exception as e:
        st.error(f"Error filtering classification data: {str(e)}")
        return []


def clear_all_classification_data() -> bool:
    """
    Clear all classification data from the database
    WARNING: This deletes all records permanently
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Delete all records from the table
        result = supabase.table('classification_data').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        
        return True
        
    except Exception as e:
        st.error(f"Error clearing classification data: {str(e)}")
        return False


def get_classification_stats() -> Dict[str, Any]:
    """
    Get summary statistics about classification data
    
    Returns:
        Dict with stats: total_count, classification_counts, confidence_counts, etc.
    """
    try:
        supabase = get_supabase_client()
        
        # Get all data for stats calculation
        result = supabase.table('classification_data')\
            .select('classification, confidence, created_at')\
            .execute()
        
        if not result.data:
            return {
                'total_count': 0,
                'classification_counts': {},
                'confidence_counts': {},
                'today_count': 0
            }
        
        # Calculate stats
        data = result.data
        total_count = len(data)
        
        # Count by classification type
        classification_counts = {}
        for row in data:
            classification = row.get('classification', 'Unknown')
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        # Count by confidence
        confidence_counts = {}
        for row in data:
            confidence = row.get('confidence', 'Unknown')
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        # Count today's classifications
        today = datetime.now().date()
        today_count = 0
        for row in data:
            if row.get('created_at'):
                try:
                    entry_date = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')).date()
                    if entry_date == today:
                        today_count += 1
                except:
                    continue
        
        return {
            'total_count': total_count,
            'classification_counts': classification_counts,
            'confidence_counts': confidence_counts,
            'today_count': today_count
        }
        
    except Exception as e:
        st.error(f"Error getting classification stats: {str(e)}")
        return {
            'total_count': 0,
            'classification_counts': {},
            'confidence_counts': {},
            'today_count': 0
        }


def test_database_connection() -> bool:
    """
    Test if database connection and table exists
    
    Returns:
        bool: True if connection works, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Try a simple count query
        result = supabase.table('classification_data').select('id', count='exact').limit(1).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Database connection test failed: {str(e)}")
        return False


def get_latest_pdf_preview() -> Optional[bytes]:
    """
    Get the most recent PDF preview from the database
    
    Returns:
        bytes: PDF preview image bytes, or None if not found
    """
    try:
        supabase = get_supabase_client()
        
        # Get the most recent entry with a pdf_preview
        result = supabase.table('classification_data')\
            .select('pdf_preview')\
            .not_.is_('pdf_preview', 'null')\
            .order('inserted_at', desc=True)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            pdf_preview_data = result.data[0].get('pdf_preview')
            if pdf_preview_data:
                # Supabase returns BYTEA as hex string (starts with \x)
                if isinstance(pdf_preview_data, str) and pdf_preview_data.startswith('\\x'):
                    # Remove \x prefix and decode from hex
                    return bytes.fromhex(pdf_preview_data[2:])
                # If already bytes, return as-is
                return pdf_preview_data
        
        return None
        
    except Exception as e:
        st.error(f"Error retrieving PDF preview: {str(e)}")
        return None
