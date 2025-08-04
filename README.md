
# Line Drawing Application - Deployment Guide

## CORS Configuration
- The application has CORS (Cross-Origin Resource Sharing) enabled in `.streamlit/config.toml` to allow file uploads from different origins
- This is necessary for the PDF upload feature to work across different browsers
- Keep in mind that while CORS is enabled for development, you may want to restrict it in production

## Security Checklist Before Deployment
1. **File Upload Security**
   - The app accepts PDF files only
   - Implement file size limits if needed for your use case
   - Consider adding file validation

2. **Browser Compatibility**
   - Test uploads in different browsers (Chrome, Edge, Firefox)
   - Clear browser cache if experiencing issues
   - Check for browser extensions that might block requests

3. **Resource Usage**
   - Monitor memory usage when processing large PDFs
   - Consider implementing rate limiting for file uploads
   - Watch CPU usage during PDF text extraction

4. **Error Handling**
   - The app includes error handling for PDF processing
   - Check logs for any unhandled exceptions
   - Ensure proper error messages are shown to users

## Deployment Configuration
The app is configured to run on port 5000 and listens on 0.0.0.0 for deployment.
Required settings in `.streamlit/config.toml`:
```toml
[server]
enableCORS = true
enableXsrfProtection = false
```

## Dependencies
Ensure all required packages are installed:
- streamlit
- numpy
- pandas
- plotly
- ezdxf
- pdf2image
- pytesseract
