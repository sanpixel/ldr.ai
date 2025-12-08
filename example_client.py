#!/usr/bin/env python3
"""
Example client for Webhook Print Server

This script demonstrates how to send documents to the webhook print server.
"""

import requests
import base64
import sys
import os

# Configuration
SERVER_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Replace with your actual API key


def print_pdf(file_path):
    """Send a PDF file to the print server."""
    print(f"Reading PDF file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    
    # Encode to base64
    pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')
    
    # Prepare request
    payload = {
        'document': pdf_b64,
        'format': 'pdf',
        'filename': os.path.basename(file_path),
        'metadata': {
            'source': 'example_client',
            'original_path': file_path
        }
    }
    
    headers = {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"Sending to print server: {SERVER_URL}/print")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/print",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        
        if response.status_code == 200:
            print("✓ Document sent successfully!")
            return True
        else:
            print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to server at {SERVER_URL}")
        print("  Make sure the server is running.")
        return False
    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def print_text(text_content, filename="document.txt"):
    """Send text content to the print server."""
    print(f"Preparing text document: {filename}")
    
    # Encode to base64
    text_b64 = base64.b64encode(text_content.encode('utf-8')).decode('utf-8')
    
    # Prepare request
    payload = {
        'document': text_b64,
        'format': 'text',
        'filename': filename,
        'metadata': {
            'source': 'example_client'
        }
    }
    
    headers = {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    print(f"Sending to print server: {SERVER_URL}/print")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/print",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        
        if response.status_code == 200:
            print("✓ Document sent successfully!")
            return True
        else:
            print(f"✗ Error: {response.json().get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to server at {SERVER_URL}")
        print("  Make sure the server is running.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_health():
    """Check if the server is running."""
    print(f"Checking server health: {SERVER_URL}/health")
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✓ Server is healthy!")
            return True
        else:
            print("✗ Server returned unexpected status")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to server at {SERVER_URL}")
        print("  Make sure the server is running.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("  Webhook Print Server - Example Client")
    print("=" * 60)
    print()
    
    # Check server health
    if not check_health():
        print("\nPlease start the server first:")
        print("  python webhook_print_server.py")
        return
    
    print()
    
    # Example 1: Print text
    print("Example 1: Printing text")
    print("-" * 60)
    text_content = """Hello from Webhook Print Server!

This is a test document sent via the API.

Features:
- API key authentication
- PDF and text support
- Easy integration

Visit the README for more information.
"""
    print_text(text_content, "test_document.txt")
    
    print()
    
    # Example 2: Print PDF (if file exists)
    print("Example 2: Printing PDF")
    print("-" * 60)
    
    # Check if there's a PDF file to print
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print_pdf(pdf_path)
    else:
        print("No PDF file specified.")
        print("Usage: python example_client.py <path_to_pdf>")
        print("Example: python example_client.py document.pdf")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
