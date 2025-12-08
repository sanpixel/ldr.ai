"""
Google Cloud Storage operations for PDF preview images
"""

import os
import json
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
from typing import Optional
import uuid


def get_gcs_client():
    """Get authenticated GCS client using service account key from environment"""
    try:
        # Get service account key from environment variable
        sa_key_json = os.getenv('GCS_SERVICE_ACCOUNT_KEY')
        if not sa_key_json:
            st.error("GCS_SERVICE_ACCOUNT_KEY not found in environment")
            return None
        
        # Parse JSON key
        sa_key_dict = json.loads(sa_key_json)
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(sa_key_dict)
        
        # Create client
        client = storage.Client(credentials=credentials, project=sa_key_dict.get('project_id'))
        
        return client
    except Exception as e:
        st.error(f"Error creating GCS client: {str(e)}")
        return None


def upload_pdf_preview(image_bytes: bytes, filename_prefix: str = "preview") -> Optional[str]:
    """
    Upload PDF preview image to GCS bucket and return signed URL
    
    Args:
        image_bytes: PNG image bytes
        filename_prefix: Prefix for the filename (default: "preview")
    
    Returns:
        str: Signed URL for the uploaded image, or None if upload fails
    """
    try:
        client = get_gcs_client()
        if not client:
            return None
        
        bucket_name = "ldr-ai"
        bucket = client.bucket(bucket_name)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        blob_name = f"{filename_prefix}_{unique_id}.png"
        
        # Upload image
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type='image/png')
        
        # Generate signed URL (valid for 7 days)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET"
        )
        
        return url
        
    except Exception as e:
        st.error(f"Error uploading to GCS: {str(e)}")
        return None


def get_latest_pdf_preview_url() -> Optional[str]:
    """
    Get the most recent PDF preview URL from the database
    
    Returns:
        str: GCS URL of the PDF preview, or None if not found
    """
    try:
        from utils.auth import get_supabase_client
        supabase = get_supabase_client()
        
        # Get the most recent entry with a pdf_preview
        result = supabase.table('classification_data')\
            .select('pdf_preview')\
            .not_.is_('pdf_preview', 'null')\
            .order('inserted_at', desc=True)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0].get('pdf_preview')
        
        return None
        
    except Exception as e:
        st.error(f"Error retrieving PDF preview URL: {str(e)}")
        return None
