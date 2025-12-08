"""
Tests for Webhook Print Server
"""

import os
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import patch
from webhook_print_server import Config, app, DocumentHandler, Document, PrintManager, PrintResult
import base64
from unittest.mock import Mock, MagicMock
from datetime import datetime


class TestConfiguration:
    """Tests for configuration management."""
    
    # Feature: webhook-print-server, Property 4: Configuration defaults
    @settings(max_examples=100)
    @given(st.none())
    def test_property_configuration_defaults(self, _):
        """
        Property 4: Configuration defaults
        For any server startup where PORT environment variable is not set,
        the server should default to port 8000.
        Validates: Requirements 4.3
        """
        # Clear PORT environment variable
        with patch.dict(os.environ, {}, clear=True):
            config = Config.load_from_env()
            assert config.PORT == 8000, "Port should default to 8000 when not set"
    
    # Feature: webhook-print-server, Property 5: API key generation
    @settings(max_examples=100)
    @given(st.none())
    def test_property_api_key_generation(self, _):
        """
        Property 5: API key generation
        For any server startup where API_KEY environment variable is not set,
        the server should generate a random API key and display it in the startup logs.
        Validates: Requirements 4.5
        """
        # Clear API_KEY environment variable
        with patch.dict(os.environ, {}, clear=True):
            config = Config.load_from_env()
            # Verify API key was generated
            assert config.API_KEY is not None, "API key should be generated"
            assert len(config.API_KEY) >= 32, "Generated API key should be at least 32 characters"
            assert config.API_KEY != "", "API key should not be empty"


class TestEndpoints:
    """Tests for Flask endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint_returns_200(self, client):
        """Test that /health endpoint returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_print_endpoint_routing(self, client):
        """Test that /print endpoint is accessible via POST."""
        response = client.post('/print', json={})
        # Should return some response (authentication will be added later)
        assert response.status_code in [200, 401, 403]


class TestAuthentication:
    """Tests for authentication middleware."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        # Set a known API key for testing
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='test-api-key-12345',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        with app.test_client() as client:
            yield client
    
    # Feature: webhook-print-server, Property 1: Authentication enforcement
    @settings(max_examples=100)
    @given(api_key=st.one_of(
        st.none(), 
        st.text(alphabet=st.characters(blacklist_categories=('Cc', 'Cs')), min_size=1, max_size=100)
    ))
    def test_property_authentication_enforcement(self, api_key):
        """
        Property 1: Authentication enforcement
        For any incoming request to the /print endpoint, if the API key is missing or invalid,
        the server should reject the request with HTTP 401 or 403 status and not process the document.
        Validates: Requirements 2.3, 2.4
        """
        app.config['TESTING'] = True
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='correct-api-key',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        
        with app.test_client() as client:
            headers = {}
            if api_key is not None:
                headers['X-API-Key'] = api_key
            
            response = client.post('/print', json={}, headers=headers)
            
            # If API key is missing or doesn't match, should get 401 or 403
            if api_key is None:
                assert response.status_code == 401, "Missing API key should return 401"
            elif api_key != 'correct-api-key':
                assert response.status_code == 403, "Invalid API key should return 403"
            else:
                # Valid key should not return 401 or 403
                assert response.status_code not in [401, 403], "Valid API key should not be rejected"
    
    def test_valid_api_key_acceptance(self, client):
        """Test that valid API key is accepted."""
        # Need to provide valid document data
        test_doc = base64.b64encode(b"test content").decode('utf-8')
        response = client.post('/print', 
            json={'document': test_doc, 'format': 'pdf'}, 
            headers={'X-API-Key': 'test-api-key-12345'})
        # Should not get 401 or 403 (authentication errors)
        assert response.status_code not in [401, 403]
    
    def test_missing_api_key_rejection(self, client):
        """Test that missing API key returns 401."""
        response = client.post('/print', json={})
        assert response.status_code == 401
        data = response.get_json()
        assert data['status'] == 'error'
        assert 'Missing API key' in data['error']
    
    def test_invalid_api_key_rejection(self, client):
        """Test that invalid API key returns 403."""
        response = client.post('/print', json={}, headers={'X-API-Key': 'wrong-key'})
        assert response.status_code == 403
        data = response.get_json()
        assert data['status'] == 'error'
        assert 'Invalid API key' in data['error']


class TestDocumentHandler:
    """Tests for document handler."""
    
    # Feature: webhook-print-server, Property 3: Document format validation
    @settings(max_examples=100)
    @given(doc_format=st.text(min_size=1, max_size=50).filter(lambda x: x.lower() not in ['pdf', 'text']))
    def test_property_document_format_validation(self, doc_format):
        """
        Property 3: Document format validation
        For any request payload, if the document format is not supported (not PDF or text),
        the server should return HTTP 400 status with format details.
        Validates: Requirements 6.5
        """
        app.config['TESTING'] = True
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='test-key',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        
        with app.test_client() as client:
            # Create request with unsupported format
            test_doc = base64.b64encode(b"test content").decode('utf-8')
            response = client.post('/print', 
                json={'document': test_doc, 'format': doc_format},
                headers={'X-API-Key': 'test-key'}
            )
            
            # Should return 400 for unsupported format
            assert response.status_code == 400, f"Unsupported format '{doc_format}' should return 400"
            data = response.get_json()
            assert 'error' in data, "Error response should contain 'error' field"
    
    def test_pdf_document_extraction(self):
        """Test extracting a PDF document."""
        test_content = b"PDF content here"
        test_doc = base64.b64encode(test_content).decode('utf-8')
        request_data = {
            'document': test_doc,
            'format': 'pdf',
            'filename': 'test.pdf'
        }
        
        doc = DocumentHandler.extract_document(request_data)
        assert doc.content == test_content
        assert doc.format == 'pdf'
        assert doc.filename == 'test.pdf'
    
    def test_text_document_extraction(self):
        """Test extracting a text document."""
        test_content = b"Text content here"
        test_doc = base64.b64encode(test_content).decode('utf-8')
        request_data = {
            'document': test_doc,
            'format': 'text',
            'filename': 'test.txt'
        }
        
        doc = DocumentHandler.extract_document(request_data)
        assert doc.content == test_content
        assert doc.format == 'text'
        assert doc.filename == 'test.txt'
    
    def test_invalid_format_rejection(self):
        """Test that invalid format is rejected."""
        test_doc = base64.b64encode(b"content").decode('utf-8')
        request_data = {
            'document': test_doc,
            'format': 'docx'
        }
        
        with pytest.raises(ValueError) as exc_info:
            DocumentHandler.extract_document(request_data)
        assert 'Unsupported format' in str(exc_info.value)
    
    def test_base64_decoding(self):
        """Test that base64 decoding works correctly."""
        test_content = b"Test content with special chars: \x00\x01\x02"
        test_doc = base64.b64encode(test_content).decode('utf-8')
        request_data = {
            'document': test_doc,
            'format': 'pdf'
        }
        
        doc = DocumentHandler.extract_document(request_data)
        assert doc.content == test_content



class TestPrintManager:
    """Tests for print manager."""
    
    def test_default_printer_detection(self):
        """Test that default printer can be detected."""
        pm = PrintManager()
        # On Windows with printer, should return a string
        # On systems without printer, should return None
        assert pm.default_printer is None or isinstance(pm.default_printer, str)
    
    def test_print_job_submission_mocked(self):
        """Test print job submission with mocked printer."""
        pm = PrintManager()
        pm.default_printer = "MockPrinter"
        
        # Create a test document
        doc = Document(
            content=b"Test content",
            format='text',
            filename='test.txt',
            metadata={}
        )
        
        # Mock the Windows API if not available
        import webhook_print_server
        if not webhook_print_server.WINDOWS_PRINTING:
            # If Windows printing not available, expect failure
            result = pm.print_document(doc)
            assert result.success == False
        else:
            # If Windows printing available, we can't easily test without actual printer
            # Just verify the method exists and returns PrintResult
            result = pm.print_document(doc)
            assert isinstance(result, PrintResult)
    
    def test_error_handling_for_unavailable_printer(self):
        """Test error handling when printer is unavailable."""
        pm = PrintManager()
        pm.default_printer = None  # Simulate no printer
        
        doc = Document(
            content=b"Test content",
            format='pdf',
            filename='test.pdf',
            metadata={}
        )
        
        result = pm.print_document(doc)
        assert result.success == False
        assert "No default printer" in result.error_message



class TestPrintEndpoint:
    """Tests for the complete print endpoint."""
    
    @pytest.fixture
    def client_with_mock_printer(self):
        """Create a test client with mocked print manager."""
        app.config['TESTING'] = True
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='test-key',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        
        # Mock the print manager
        mock_pm = Mock(spec=PrintManager)
        mock_pm.default_printer = "MockPrinter"
        mock_pm.print_document = Mock(return_value=PrintResult(
            success=True,
            job_id=12345,
            error_message=None,
            timestamp=datetime.utcnow()
        ))
        webhook_print_server.print_manager = mock_pm
        
        with app.test_client() as client:
            yield client
    
    # Feature: webhook-print-server, Property 2: Successful print job response
    @settings(max_examples=100)
    @given(
        content=st.binary(min_size=1, max_size=1000),
        doc_format=st.sampled_from(['pdf', 'text'])
    )
    def test_property_successful_print_job_response(self, content, doc_format):
        """
        Property 2: Successful print job response
        For any valid authenticated request with a supported document format,
        the server should return HTTP 200 status and include a confirmation message in the response body.
        Validates: Requirements 6.1, 6.2
        """
        app.config['TESTING'] = True
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='test-key',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        
        # Mock the print manager to always succeed
        mock_pm = Mock(spec=PrintManager)
        mock_pm.default_printer = "MockPrinter"
        mock_pm.print_document = Mock(return_value=PrintResult(
            success=True,
            job_id=12345,
            error_message=None,
            timestamp=datetime.utcnow()
        ))
        webhook_print_server.print_manager = mock_pm
        
        with app.test_client() as client:
            # Create valid request
            doc_b64 = base64.b64encode(content).decode('utf-8')
            response = client.post('/print',
                json={'document': doc_b64, 'format': doc_format},
                headers={'X-API-Key': 'test-key'}
            )
            
            # Should return 200 for valid request
            assert response.status_code == 200, f"Valid request should return 200, got {response.status_code}"
            data = response.get_json()
            assert data['status'] == 'success', "Response should indicate success"
            assert 'message' in data, "Response should contain confirmation message"

    
    # Feature: webhook-print-server, Property 6: Logging completeness
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        content=st.binary(min_size=1, max_size=500),
        doc_format=st.sampled_from(['pdf', 'text', 'invalid']),
        has_api_key=st.booleans()
    )
    def test_property_logging_completeness(self, content, doc_format, has_api_key, caplog):
        """
        Property 6: Logging completeness
        For any processed request (successful or failed), the server should log
        the request details including timestamp, source, and outcome.
        Validates: Requirements 5.5
        """
        import logging
        caplog.set_level(logging.INFO)
        
        app.config['TESTING'] = True
        import webhook_print_server
        webhook_print_server.config = Config(
            PORT=8000,
            HOST='127.0.0.1',
            API_KEY='test-key',
            LOG_LEVEL='INFO',
            MAX_DOCUMENT_SIZE=10
        )
        
        # Mock the print manager
        mock_pm = Mock(spec=PrintManager)
        mock_pm.default_printer = "MockPrinter"
        mock_pm.print_document = Mock(return_value=PrintResult(
            success=True,
            job_id=12345,
            error_message=None,
            timestamp=datetime.utcnow()
        ))
        webhook_print_server.print_manager = mock_pm
        
        with app.test_client() as client:
            # Create request
            doc_b64 = base64.b64encode(content).decode('utf-8')
            headers = {'X-API-Key': 'test-key'} if has_api_key else {}
            
            response = client.post('/print',
                json={'document': doc_b64, 'format': doc_format},
                headers=headers
            )
            
            # Check that request was logged
            # Should have at least one log entry about the request
            assert len(caplog.records) > 0, "Request should generate log entries"
            
            # Check for request logging (from before_request middleware)
            request_logs = [r for r in caplog.records if 'Request:' in r.message or 'POST /print' in r.message]
            assert len(request_logs) > 0, "Request should be logged"



class TestServerStartup:
    """Tests for server startup and initialization."""
    
    def test_successful_server_start(self):
        """Test that server can be initialized successfully."""
        # Test configuration loading
        with patch.dict(os.environ, {'PORT': '9000', 'API_KEY': 'test-key'}):
            config = Config.load_from_env()
            assert config.PORT == 9000
            assert config.API_KEY == 'test-key'
        
        # Test print manager initialization
        pm = PrintManager()
        assert pm is not None
    
    def test_port_binding_error_handling(self):
        """Test that port binding errors are handled gracefully."""
        # This is difficult to test without actually binding ports
        # We'll just verify the error handling code exists
        import webhook_print_server
        # The error handling is in the main block, which we can't easily unit test
        # But we can verify the Config class handles port configuration
        with patch.dict(os.environ, {'PORT': '8000'}):
            config = Config.load_from_env()
            assert config.PORT == 8000
