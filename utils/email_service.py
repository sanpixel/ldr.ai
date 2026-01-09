"""
Email Service using Resend API
Handles sending emails with file attachments.
"""

import os
import json
import base64
import logging
from typing import Optional

try:
    from resend import Resend
except ImportError:
    Resend = None


def get_resend_api_key() -> Optional[str]:
    """Get Resend API key from environment or local file."""
    # First try environment variable (works for both production and local .env)
    env_key = os.getenv("RESEND_API_KEY")
    if env_key:
        return env_key
    
    # Fallback to local JSON file (legacy local development - keeps existing setup working)
    local_key_file = r"C:\dev\openai-key.json"
    try:
        if os.path.exists(local_key_file):
            with open(local_key_file, 'r') as f:
                key_data = json.load(f)
                return key_data.get('RESEND_API_KEY')
    except Exception as e:
        logging.warning(f"Could not read Resend key from {local_key_file}: {e}")
    
    return None


def send_email_with_attachments(
    recipient_email: str,
    dxf_bytes: bytes,
    highlighted_pdf_bytes: bytes,
    report_pdf_bytes: bytes,
    filename_prefix: str = "legal_description"
) -> bool:
    """
    Send email with DXF and PDF attachments using Resend.
    
    Args:
        recipient_email: Email address to send to
        dxf_bytes: DXF file content as bytes
        highlighted_pdf_bytes: Highlighted PDF content as bytes
        report_pdf_bytes: Report PDF content as bytes
        filename_prefix: Prefix for attachment filenames
    
    Returns:
        True if email sent successfully, False otherwise
    """
    if not Resend:
        logging.error("Resend library not installed")
        return False
    
    api_key = get_resend_api_key()
    if not api_key:
        logging.error("Resend API key not configured")
        return False
    
    try:
        client = Resend(api_key=api_key)
        
        # Prepare attachments
        attachments = [
            {
                "filename": f"{filename_prefix}.dxf",
                "content": base64.b64encode(dxf_bytes).decode('utf-8'),
            },
            {
                "filename": f"{filename_prefix}_highlighted.pdf",
                "content": base64.b64encode(highlighted_pdf_bytes).decode('utf-8'),
            },
            {
                "filename": f"{filename_prefix}_report.pdf",
                "content": base64.b64encode(report_pdf_bytes).decode('utf-8'),
            }
        ]
        
        # Send email
        response = client.emails.send({
            "from": "noreply@ldr.clocknumbers.com",
            "to": recipient_email,
            "subject": "Your Legal Description Analysis Results",
            "html": f"""
            <h2>Legal Description Analysis Results</h2>
            <p>Your analysis is complete. Please find the attached files:</p>
            <ul>
                <li><strong>{filename_prefix}.dxf</strong> - AutoCAD drawing file</li>
                <li><strong>{filename_prefix}_highlighted.pdf</strong> - Highlighted legal description</li>
                <li><strong>{filename_prefix}_report.pdf</strong> - Analysis report</li>
            </ul>
            <p>Thank you for using Legal Description Reader.</p>
            """,
            "attachments": attachments
        })
        
        logging.info(f"Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False
