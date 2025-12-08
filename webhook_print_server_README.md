# Webhook Print Server

A lightweight Python HTTP server that accepts authenticated POST requests containing document data and automatically sends them to your system's default printer.

## Features

- 🔐 API key authentication
- 📄 Supports PDF and text documents
- 🖨️ Prints to system default printer
- 📝 Comprehensive logging
- ⚙️ Environment-based configuration
- 🚀 Easy to deploy and run

## Requirements

- Python 3.7+
- Windows OS (for printing functionality)
- Default printer configured in Windows

## Installation

1. Install dependencies:

```bash
pip install flask python-dotenv pywin32
```

2. Create a `.env` file (optional):

```bash
PORT=8000
HOST=127.0.0.1
API_KEY=your-secret-api-key-here
LOG_LEVEL=INFO
```

If you don't set an API_KEY, one will be generated automatically at startup.

## Usage

### Starting the Server

```bash
python webhook_print_server.py
```

The server will display:
- The listening address and port
- The default printer name
- The API key (if generated)
- Available endpoints

### Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `HOST` | 127.0.0.1 | Bind address (use 0.0.0.0 for network access) |
| `API_KEY` | (generated) | Authentication key |
| `LOG_LEVEL` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `MAX_DOCUMENT_SIZE` | 10 | Maximum document size in MB |

### API Endpoints

#### Health Check

```bash
GET /health
```

Returns server status. No authentication required.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-08T10:30:00Z"
}
```

#### Print Document

```bash
POST /print
```

Sends a document to the default printer. Requires authentication.

**Headers:**
- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body:**
```json
{
  "document": "base64_encoded_document_data",
  "format": "pdf",
  "filename": "document.pdf",
  "metadata": {
    "source": "my_app",
    "timestamp": "2025-12-08T10:30:00Z"
  }
}
```

**Supported Formats:**
- `pdf` - PDF documents
- `text` - Plain text documents

**Success Response (200):**
```json
{
  "status": "success",
  "message": "Document queued for printing",
  "filename": "document.pdf",
  "format": "pdf",
  "job_id": null,
  "timestamp": "2025-12-08T10:30:01Z"
}
```

**Error Responses:**
- `401` - Missing API key
- `403` - Invalid API key
- `400` - Invalid request (bad format, missing fields, etc.)
- `503` - Printer not available
- `500` - Internal server error

## Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Print a PDF
curl -X POST http://localhost:8000/print \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "document": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvTWVkaWFCb3hbMCAwIDYxMiA3OTJdL1BhcmVudCAyIDAgUi9SZXNvdXJjZXM8PD4+Pj4KZW5kb2JqCnhyZWYKMCA0CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAxNSAwMDAwMCBuIAowMDAwMDAwMDY0IDAwMDAwIG4gCjAwMDAwMDAxMTUgMDAwMDAgbiAKdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoyMDQKJSVFT0YK",
    "format": "pdf",
    "filename": "test.pdf"
  }'

# Print text
curl -X POST http://localhost:8000/print \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "document": "SGVsbG8sIFdvcmxkIQ==",
    "format": "text",
    "filename": "hello.txt"
  }'
```

### Using Python

```python
import requests
import base64

# Read a PDF file
with open('document.pdf', 'rb') as f:
    pdf_data = f.read()

# Encode to base64
pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')

# Send to print server
response = requests.post(
    'http://localhost:8000/print',
    headers={'X-API-Key': 'your-api-key-here'},
    json={
        'document': pdf_b64,
        'format': 'pdf',
        'filename': 'document.pdf'
    }
)

print(response.json())
```

See `example_client.py` for a complete example.

## Exposing to the Internet

The server runs locally by default. To accept requests from the internet, you have several options:

### Option 1: ngrok (Easiest)

```bash
# Install ngrok from https://ngrok.com/
ngrok http 8000
```

This creates a public URL that tunnels to your local server.

### Option 2: Port Forwarding

Configure your router to forward a port to your machine's local IP and port 8000.

### Option 3: Cloudflare Tunnel

```bash
# Install cloudflared
cloudflared tunnel --url http://localhost:8000
```

### Option 4: Network Binding

Set `HOST=0.0.0.0` to allow connections from your local network:

```bash
set HOST=0.0.0.0
python webhook_print_server.py
```

**Security Note:** Always use a strong API key when exposing the server to the internet.

## Troubleshooting

### No Printer Detected

**Problem:** Server starts but shows "None detected" for default printer.

**Solution:**
- Ensure you have a printer installed in Windows
- Set a default printer in Windows Settings → Devices → Printers & scanners
- Restart the server after configuring the printer

### Port Already in Use

**Problem:** Error message "Port 8000 is already in use"

**Solution:**
- Choose a different port: `set PORT=8001`
- Or stop the process using port 8000

### Print Job Not Printing

**Problem:** Server returns success but nothing prints.

**Solution:**
- Check printer is online and has paper
- Check Windows print queue for errors
- Verify the document format is correct
- Check server logs for error messages

### Authentication Failures

**Problem:** Getting 401 or 403 errors

**Solution:**
- Ensure you're sending the `X-API-Key` header
- Verify the API key matches the one shown at server startup
- Check for typos in the API key

## Development

### Running Tests

```bash
pip install pytest hypothesis
pytest test_webhook_print_server.py -v
```

### Project Structure

```
webhook_print_server.py    # Main server application
test_webhook_print_server.py  # Test suite
.env.example               # Example configuration
webhook_print_server_README.md  # This file
example_client.py          # Example client script
```

## Security Considerations

- **API Key:** Use a strong, randomly generated API key
- **HTTPS:** Consider using HTTPS in production (via reverse proxy)
- **Network:** Bind to 127.0.0.1 unless network access is required
- **Firewall:** Configure firewall rules appropriately
- **Document Size:** Limit document sizes to prevent abuse

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions, please check the troubleshooting section above.
