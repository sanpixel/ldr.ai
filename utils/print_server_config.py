"""
Print Server Configuration Management
Handles storing and retrieving the print server URL from Supabase.
"""

from utils.auth import get_supabase_client
import logging

def save_print_server_url(url: str) -> bool:
    """Save or update the print server URL in Supabase."""
    try:
        supabase = get_supabase_client()
        
        # Upsert - insert if not exists, update if exists
        result = supabase.table('print_server_config').upsert({
            'id': '00000000-0000-0000-0000-000000000001',  # Fixed ID for singleton pattern
            'print_server_url': url
        }).execute()
        
        if result.data:
            logging.info(f"Print server URL updated: {url}")
            return True
        return False
    except Exception as e:
        logging.error(f"Failed to save print server URL: {e}")
        return False


def get_print_server_url() -> str:
    """Get the latest print server URL from Supabase."""
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('print_server_config')\
            .select('print_server_url')\
            .order('updated_at', desc=True)\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['print_server_url']
        
        # Fallback to default if no URL found
        return "https://f9c54cb3a24a.ngrok-free.app"
    except Exception as e:
        logging.error(f"Failed to get print server URL: {e}")
        return "https://f9c54cb3a24a.ngrok-free.app"


def test_database_connection() -> bool:
    """Test if the database connection works."""
    try:
        supabase = get_supabase_client()
        result = supabase.table('print_server_config').select('id', count='exact').limit(1).execute()
        return True
    except Exception as e:
        logging.error(f"Database connection test failed: {e}")
        return False
