#!/usr/bin/env python3
"""
Webhook Print Server
A lightweight HTTP server that accepts authenticated POST requests and prints documents to the default printer.
"""

import os
import logging
import secrets
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
    import win32print
    import win32api
    WINDOWS_PRINTING = True
except ImportError:
    WINDOWS_PRINTING = False
    logging.warning("win32print not available - printing functionality will be limited")

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration for the webhook print server."""
    PORT: int
    HOST: str
    API_KEY: str
    LOG_LEVEL: str
    MAX_DOCUMENT_SIZE: int
    
    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with defaults."""
        # Get port with default
        port = int(os.getenv('PORT', '8000'))
        
        # Get host with default
        host = os.getenv('HOST', '127.0.0.1')
        
        # Get or generate API key
        api_key = os.getenv('API_KEY')
        if not api_key:
            # Generate a secure random API key (32 bytes = 64 hex characters)
            api_key = secrets.token_hex(32)
            logging.info(f"Generated API Key: {api_key}")
            logging.info("IMPORTANT: Save this API key - it will be needed for all requests!")
        
        # Get log level with default
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Get max document size with default (in MB)
        max_doc_size = int(os.getenv('MAX_DOCUMENT_SIZE', '10'))
        
        return cls(
            PORT=port,
            HOST=host,
            API_KEY=api_key,
            LOG_LEVEL=log_level,
            MAX_DOCUMENT_SIZE=max_doc_size
        )


def setup_logging(log_level: str):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


from flask import Flask, request, jsonify
from functools import wraps
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
config = None  # Will be set in main
print_manager = None  # Will be set in main


@dataclass
class Document:
    """Represents a document to be printed."""
    content: bytes
    format: str
    filename: str
    metadata: Dict[str, Any]


class DocumentHandler:
    """Handles document extraction and validation."""
    
    SUPPORTED_FORMATS = ['pdf', 'text']
    
    @staticmethod
    def extract_document(request_data: dict) -> Document:
        """
        Extract document from request JSON.
        Raises ValueError if document data is missing or invalid.
        """
        if not request_data:
            raise ValueError("Request body is empty")
        
        if 'document' not in request_data:
            raise ValueError("Missing 'document' field in request")
        
        if 'format' not in request_data:
            raise ValueError("Missing 'format' field in request")
        
        doc_format = request_data['format'].lower()
        
        # Validate format
        if doc_format not in DocumentHandler.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {doc_format}. Supported formats: {', '.join(DocumentHandler.SUPPORTED_FORMATS)}")
        
        # Decode base64 document content
        try:
            doc_content = base64.b64decode(request_data['document'])
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        filename = request_data.get('filename', f'document.{doc_format}')
        metadata = request_data.get('metadata', {})
        
        return Document(
            content=doc_content,
            format=doc_format,
            filename=filename,
            metadata=metadata
        )
    
    @staticmethod
    def validate_format(document: Document) -> bool:
        """Validate that document format is supported."""
        return document.format in DocumentHandler.SUPPORTED_FORMATS


@dataclass
class PrintResult:
    """Result of a print operation."""
    success: bool
    job_id: Optional[int]
    error_message: Optional[str]
    timestamp: datetime


class PrintManager:
    """Manages printing documents to the system printer."""
    
    def __init__(self):
        """Initialize the print manager."""
        self.default_printer = self.get_default_printer()
    
    def get_default_printer(self) -> str:
        """Get the system's default printer name."""
        if WINDOWS_PRINTING:
            try:
                return win32print.GetDefaultPrinter()
            except Exception as e:
                logging.error(f"Failed to get default printer: {e}")
                return None
        else:
            logging.warning("Windows printing not available")
            return None
    
    def print_document(self, document: Document) -> PrintResult:
        """
        Print a document to the default printer.
        Routes to appropriate handler based on document format.
        """
        if document.format == 'pdf':
            return self.print_pdf(document.content, document.filename)
        elif document.format == 'text':
            return self.print_text(document.content.decode('utf-8', errors='replace'), document.filename)
        else:
            return PrintResult(
                success=False,
                job_id=None,
                error_message=f"Unsupported format: {document.format}",
                timestamp=datetime.utcnow()
            )
    
    def print_pdf(self, pdf_data: bytes, filename: str = "document.pdf") -> PrintResult:
        """Print PDF data to the default printer."""
        if not self.default_printer:
            return PrintResult(
                success=False,
                job_id=None,
                error_message="No default printer available",
                timestamp=datetime.utcnow()
            )
        
        if not WINDOWS_PRINTING:
            return PrintResult(
                success=False,
                job_id=None,
                error_message="Windows printing not available",
                timestamp=datetime.utcnow()
            )
        
        try:
            # Write PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_data)
                temp_path = temp_file.name
            
            # Print using Windows API
            win32api.ShellExecute(
                0,
                "print",
                temp_path,
                f'/d:"{self.default_printer}"',
                ".",
                0
            )
            
            logging.info(f"PDF print job sent: {filename} to {self.default_printer}")
            
            # Clean up temp file after a delay (Windows needs time to read it)
            # Note: In production, you might want a better cleanup strategy
            
            return PrintResult(
                success=True,
                job_id=None,  # Windows ShellExecute doesn't return job ID
                error_message=None,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Failed to print PDF: {e}")
            return PrintResult(
                success=False,
                job_id=None,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
    
    def print_text(self, text_content: str, filename: str = "document.txt") -> PrintResult:
        """Print text content to the default printer."""
        if not self.default_printer:
            return PrintResult(
                success=False,
                job_id=None,
                error_message="No default printer available",
                timestamp=datetime.utcnow()
            )
        
        if not WINDOWS_PRINTING:
            return PrintResult(
                success=False,
                job_id=None,
                error_message="Windows printing not available",
                timestamp=datetime.utcnow()
            )
        
        try:
            # Write text to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_file:
                temp_file.write(text_content)
                temp_path = temp_file.name
            
            # Print using Windows API
            win32api.ShellExecute(
                0,
                "print",
                temp_path,
                f'/d:"{self.default_printer}"',
                ".",
                0
            )
            
            logging.info(f"Text print job sent: {filename} to {self.default_printer}")
            
            return PrintResult(
                success=True,
                job_id=None,
                error_message=None,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Failed to print text: {e}")
            return PrintResult(
                success=False,
                job_id=None,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )


def require_api_key(f):
    """
    Decorator to validate API key in request headers.
    Checks for 'X-API-Key' header and validates against configured key.
    Returns 401 if missing, 403 if invalid.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logging.warning(f"Authentication failed: Missing API key from {request.remote_addr}")
            return jsonify({
                'status': 'error',
                'error': 'Missing API key',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 401
        
        if api_key != config.API_KEY:
            logging.warning(f"Authentication failed: Invalid API key from {request.remote_addr}")
            return jsonify({
                'status': 'error',
                'error': 'Invalid API key',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 403
        
        logging.info(f"Authentication successful from {request.remote_addr}")
        return f(*args, **kwargs)
    
    return decorated_function


def log_request():
    """Middleware to log all incoming requests."""
    logging.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.before_request
def before_request():
    """Log all requests before processing."""
    log_request()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - no authentication required."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.route('/print', methods=['POST'])
@require_api_key
def print_endpoint():
    """Main endpoint for receiving print jobs - authentication required."""
    try:
        # Extract document from request
        request_data = request.get_json()
        document = DocumentHandler.extract_document(request_data)
        
        logging.info(f"Document received from {request.remote_addr}: {document.filename} ({document.format}), size: {len(document.content)} bytes")
        
        # Send document to printer
        result = print_manager.print_document(document)
        
        if result.success:
            logging.info(f"Print job successful: {document.filename} at {result.timestamp}")
            return jsonify({
                'status': 'success',
                'message': 'Document queued for printing',
                'filename': document.filename,
                'format': document.format,
                'job_id': result.job_id,
                'timestamp': result.timestamp.isoformat() + 'Z'
            }), 200
        else:
            logging.error(f"Print job failed: {result.error_message}")
            # Check if it's a printer availability issue
            if "No default printer" in result.error_message or "not available" in result.error_message:
                return jsonify({
                    'status': 'error',
                    'error': result.error_message,
                    'timestamp': result.timestamp.isoformat() + 'Z'
                }), 503
            else:
                return jsonify({
                    'status': 'error',
                    'error': result.error_message,
                    'timestamp': result.timestamp.isoformat() + 'Z'
                }), 500
        
    except ValueError as e:
        logging.error(f"Invalid request from {request.remote_addr}: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 400
    except Exception as e:
        logging.error(f"Error processing request from {request.remote_addr}: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


# Main entry point
if __name__ == "__main__":
    config = Config.load_from_env()
    setup_logging(config.LOG_LEVEL)
    
    # Initialize print manager
    print_manager = PrintManager()
    
    # Display startup banner
    print("\n" + "=" * 60)
    print("  Webhook Print Server")
    print("=" * 60)
    logging.info("Webhook Print Server starting...")
    logging.info(f"Server will listen on {config.HOST}:{config.PORT}")
    logging.info(f"Default printer: {print_manager.default_printer or 'None detected'}")
    
    if not print_manager.default_printer:
        logging.warning("WARNING: No default printer detected. Print jobs will fail.")
    
    logging.info(f"API Key: {'(using configured key)' if os.getenv('API_KEY') else '(generated - see above)'}")
    print(f"\n  Health check: http://{config.HOST}:{config.PORT}/health")
    print(f"  Print endpoint: http://{config.HOST}:{config.PORT}/print")
    print(f"  Authentication: X-API-Key header required")
    print("=" * 60 + "\n")
    
    try:
        logging.info(f"Starting Flask server on {config.HOST}:{config.PORT}...")
        app.run(host=config.HOST, port=config.PORT, debug=False)
    except OSError as e:
        if "address already in use" in str(e).lower() or "winerror 10048" in str(e).lower():
            logging.error(f"ERROR: Port {config.PORT} is already in use!")
            logging.error(f"Please choose a different port by setting PORT environment variable.")
            logging.error(f"Example: set PORT=8001")
            print(f"\n❌ ERROR: Port {config.PORT} is already in use!")
            print(f"   Set a different port: set PORT=8001\n")
        else:
            logging.error(f"Failed to start server: {e}")
            print(f"\n❌ ERROR: Failed to start server: {e}\n")
    except KeyboardInterrupt:
        logging.info("Server shutdown requested by user")
        print("\n\n👋 Server stopped by user\n")
    except Exception as e:
        logging.error(f"Unexpected server error: {e}")
        print(f"\n❌ ERROR: {e}\n")
