# Design Document: Webhook Print Server

## Overview

The Webhook Print Server is a lightweight Python HTTP server that accepts authenticated POST requests containing document data and automatically sends them to the system's default printer. The server will run locally, validate incoming requests using an API key, and support common document formats including PDF and plain text.

## Architecture

### High-Level Architecture

```
External App → HTTP POST → Webhook Server → Document Handler → System Printer
                              ↓
                         Authentication
                              ↓
                           Logging
```

### Technology Stack

- **HTTP Server**: Flask (lightweight, simple routing, easy to deploy)
- **Printing**: 
  - Windows: `win32print` library for direct printer access
  - Cross-platform fallback: System commands (`lpr` on Unix, `print` on Windows)
- **Document Handling**: 
  - PDF: Direct printing via printer driver
  - Text: Convert to printable format
- **Configuration**: Environment variables via `python-dotenv`
- **Logging**: Python `logging` module

### Deployment Model

- Standalone Python script that runs locally
- Can be started manually or as a background service
- Listens on localhost by default (configurable to 0.0.0.0 for network access)

## Components and Interfaces

### 1. HTTP Server Component

**Responsibilities:**
- Accept incoming HTTP POST requests
- Route requests to appropriate handlers
- Return HTTP responses

**Interface:**
```python
class WebhookServer:
    def __init__(self, port: int, api_key: str)
    def start(self) -> None
    def stop(self) -> None
```

**Endpoints:**
- `POST /print` - Main endpoint for receiving print jobs
- `GET /health` - Health check endpoint (no auth required)

### 2. Authentication Middleware

**Responsibilities:**
- Validate API key from request headers
- Reject unauthorized requests
- Log authentication attempts

**Interface:**
```python
def require_api_key(func):
    """Decorator to validate API key in request headers"""
    # Checks for 'X-API-Key' header
    # Returns 401 if missing, 403 if invalid
```

### 3. Document Handler Component

**Responsibilities:**
- Extract document data from request payload
- Determine document format
- Prepare document for printing

**Interface:**
```python
class DocumentHandler:
    def extract_document(self, request_data: dict) -> Document
    def validate_format(self, document: Document) -> bool
    def prepare_for_print(self, document: Document) -> bytes
```

**Document Data Structure:**
```python
@dataclass
class Document:
    content: bytes          # Raw document data
    format: str            # 'pdf', 'text', etc.
    filename: str          # Optional filename
    metadata: dict         # Additional metadata
```

### 4. Print Manager Component

**Responsibilities:**
- Interface with system printer
- Queue print jobs
- Handle print errors
- Report print status

**Interface:**
```python
class PrintManager:
    def __init__(self)
    def get_default_printer(self) -> str
    def print_document(self, document: Document) -> PrintResult
    def print_pdf(self, pdf_data: bytes) -> PrintResult
    def print_text(self, text_content: str) -> PrintResult
```

**PrintResult Data Structure:**
```python
@dataclass
class PrintResult:
    success: bool
    job_id: Optional[int]
    error_message: Optional[str]
    timestamp: datetime
```

### 5. Configuration Manager

**Responsibilities:**
- Load configuration from environment variables
- Provide default values
- Generate random API key if not specified

**Interface:**
```python
class Config:
    PORT: int
    API_KEY: str
    HOST: str
    LOG_LEVEL: str
    
    @classmethod
    def load_from_env(cls) -> Config
```

### 6. Logger Component

**Responsibilities:**
- Log all server activity
- Log print job status
- Log authentication attempts
- Format log messages with timestamps

**Configuration:**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Data Models

### Request Payload Format

The server expects POST requests with the following JSON structure:

```json
{
  "document": "base64_encoded_document_data",
  "format": "pdf|text",
  "filename": "optional_filename.pdf",
  "metadata": {
    "source": "app_name",
    "timestamp": "2025-12-08T10:30:00Z"
  }
}
```

### Response Format

**Success Response (200):**
```json
{
  "status": "success",
  "message": "Document queued for printing",
  "job_id": 12345,
  "timestamp": "2025-12-08T10:30:01Z"
}
```

**Error Response (4xx/5xx):**
```json
{
  "status": "error",
  "error": "Error description",
  "timestamp": "2025-12-08T10:30:01Z"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Authentication enforcement
*For any* incoming request to the `/print` endpoint, if the API key is missing or invalid, the server should reject the request with HTTP 401 or 403 status and not process the document.
**Validates: Requirements 2.3, 2.4**

### Property 2: Successful print job response
*For any* valid authenticated request with a supported document format, the server should return HTTP 200 status and include a confirmation message in the response body.
**Validates: Requirements 6.1, 6.2**

### Property 3: Document format validation
*For any* request payload, if the document format is not supported (not PDF or text), the server should return HTTP 400 status with format details.
**Validates: Requirements 6.5**

### Property 4: Configuration defaults
*For any* server startup where PORT environment variable is not set, the server should default to port 8000.
**Validates: Requirements 4.3**

### Property 5: API key generation
*For any* server startup where API_KEY environment variable is not set, the server should generate a random API key and display it in the startup logs.
**Validates: Requirements 4.5**

### Property 6: Logging completeness
*For any* processed request (successful or failed), the server should log the request details including timestamp, source, and outcome.
**Validates: Requirements 5.5**

## Error Handling

### Error Categories

1. **Authentication Errors**
   - Missing API key → 401 Unauthorized
   - Invalid API key → 403 Forbidden
   - Log: "Authentication failed: [reason]"

2. **Request Validation Errors**
   - Missing document data → 400 Bad Request
   - Invalid JSON → 400 Bad Request
   - Unsupported format → 400 Bad Request
   - Log: "Invalid request: [details]"

3. **Printing Errors**
   - Printer not available → 503 Service Unavailable
   - Print job failed → 500 Internal Server Error
   - Document format error → 400 Bad Request
   - Log: "Print error: [details]"

4. **Server Errors**
   - Port already in use → Exit with error message
   - Configuration error → Exit with error message
   - Log: "Server error: [details]"

### Error Response Strategy

- All errors return JSON with consistent structure
- Include helpful error messages for debugging
- Log all errors with full context
- Never expose sensitive information (API keys) in responses

## Testing Strategy

### Unit Testing

**Framework**: pytest

**Test Coverage:**

1. **Authentication Tests**
   - Test valid API key acceptance
   - Test missing API key rejection (401)
   - Test invalid API key rejection (403)

2. **Document Handler Tests**
   - Test PDF document extraction
   - Test text document extraction
   - Test invalid format rejection
   - Test base64 decoding

3. **Print Manager Tests**
   - Test default printer detection
   - Test print job submission (mocked printer)
   - Test error handling for unavailable printer

4. **Configuration Tests**
   - Test environment variable loading
   - Test default port value (8000)
   - Test API key generation when not provided

5. **Endpoint Tests**
   - Test `/print` endpoint with valid request
   - Test `/health` endpoint
   - Test error responses

### Property-Based Testing

**Framework**: Hypothesis (Python property-based testing library)

**Configuration**: Each property test should run a minimum of 100 iterations.

**Test Tagging**: Each property-based test must include a comment with the format:
`# Feature: webhook-print-server, Property {number}: {property_text}`

**Property Tests:**

1. **Property 1: Authentication enforcement**
   - Generate random API keys and request headers
   - Verify that mismatched keys always result in 401/403
   - Tag: `# Feature: webhook-print-server, Property 1: Authentication enforcement`

2. **Property 2: Successful print job response**
   - Generate random valid documents (PDF and text)
   - Verify all valid requests return 200 with confirmation
   - Tag: `# Feature: webhook-print-server, Property 2: Successful print job response`

3. **Property 3: Document format validation**
   - Generate random unsupported format strings
   - Verify all unsupported formats return 400
   - Tag: `# Feature: webhook-print-server, Property 3: Document format validation`

4. **Property 4: Configuration defaults**
   - Test server initialization without PORT env var
   - Verify port always defaults to 8000
   - Tag: `# Feature: webhook-print-server, Property 4: Configuration defaults`

5. **Property 5: API key generation**
   - Test server initialization without API_KEY env var
   - Verify a random key is always generated and logged
   - Tag: `# Feature: webhook-print-server, Property 5: API key generation`

6. **Property 6: Logging completeness**
   - Generate random requests (valid and invalid)
   - Verify all requests produce log entries with required fields
   - Tag: `# Feature: webhook-print-server, Property 6: Logging completeness`

### Integration Testing

- Test full request flow from HTTP POST to print job
- Test with actual printer (manual verification)
- Test with various document formats and sizes
- Test concurrent requests

### Manual Testing

- Send test documents from external app
- Verify physical printer output
- Test error scenarios (printer offline, invalid documents)
- Verify log output is readable and complete

## Security Considerations

### API Key Security

- API key transmitted in HTTP header (X-API-Key)
- Consider HTTPS for production use
- API key should be strong (minimum 32 characters if generated)
- Never log the actual API key value

### Network Security

- Default to localhost binding (127.0.0.1)
- Only bind to 0.0.0.0 if explicitly configured
- Consider firewall rules for network access
- No authentication bypass for any endpoint except `/health`

### Input Validation

- Validate all request data before processing
- Limit document size to prevent memory exhaustion
- Sanitize filenames to prevent path traversal
- Validate base64 encoding before decoding

## Deployment

### Installation

```bash
# Install dependencies
pip install flask python-dotenv pywin32  # Windows
pip install flask python-dotenv          # Unix/Linux

# Create .env file
echo "PORT=8000" > .env
echo "API_KEY=your-secret-key-here" >> .env

# Run server
python webhook_print_server.py
```

### Running as a Service

**Windows:**
- Use NSSM (Non-Sucking Service Manager) to create Windows service
- Or use Task Scheduler to run on startup

**Linux:**
- Create systemd service unit file
- Enable and start service

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8000 | Server port |
| HOST | 127.0.0.1 | Bind address |
| API_KEY | (generated) | Authentication key |
| LOG_LEVEL | INFO | Logging verbosity |
| MAX_DOCUMENT_SIZE | 10MB | Maximum document size |

## Future Enhancements

- Support for additional document formats (DOCX, images)
- Print queue management (view, cancel jobs)
- Multiple printer support (specify printer in request)
- Print job history and statistics
- Web UI for monitoring
- HTTPS support with self-signed certificates
- Rate limiting to prevent abuse
- Webhook callbacks for print job completion
