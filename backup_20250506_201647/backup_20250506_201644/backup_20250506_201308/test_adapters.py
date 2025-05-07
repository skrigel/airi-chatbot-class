#!/usr/bin/env python
import requests
import sys
import os
import time
import json

def print_colored(text, color):
    """Print colored text."""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def test_endpoint(url, data=None, method='GET'):
    """Test an endpoint and return response."""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=5)
        else:  # POST
            response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def run_tests(base_url):
    """Run tests against the adapter."""
    print_colored(f"Testing adapter at {base_url}", 'blue')
    print("-" * 50)
    
    # Test health endpoint
    print_colored("Testing health endpoint...", 'blue')
    success, result = test_endpoint(f"{base_url}/api/health")
    if success:
        print_colored("✓ Health endpoint working", 'green')
        print(f"Status: {result.get('status')}")
        if 'components' in result:
            for component, status in result.get('components', {}).items():
                color = 'green' if status == 'ok' else 'yellow' if status == 'disabled' else 'red'
                print_colored(f"  - {component}: {status}", color)
    else:
        print_colored(f"✗ Health endpoint failed: {result}", 'red')
    print()
    
    # Test message endpoint with a simple query
    print_colored("Testing message endpoint...", 'blue')
    success, result = test_endpoint(
        f"{base_url}/api/v1/sendMessage", 
        data={"message": "What is AI risk?", "conversationId": "test-123"},
        method='POST'
    )
    if success:
        print_colored("✓ Message endpoint working", 'green')
        response = result.get('response', '')
        preview = response[:100] + "..." if len(response) > 100 else response
        print(f"Response preview: {preview}")
    else:
        print_colored(f"✗ Message endpoint failed: {result}", 'red')
    print()
    
    # Test reset endpoint
    print_colored("Testing reset endpoint...", 'blue')
    success, result = test_endpoint(f"{base_url}/api/v1/reset", method='POST')
    if success:
        print_colored("✓ Reset endpoint working", 'green')
        print(f"Result: {result}")
    else:
        print_colored(f"✗ Reset endpoint failed: {result}", 'red')
    print()
    
    # Final status
    if success:
        print_colored("All tests completed successfully! Your adapter is working.", 'green')
        print_colored("You can access the chatbot at: " + base_url, 'blue')
    else:
        print_colored("Some tests failed. Check the logs above for details.", 'yellow')

if __name__ == "__main__":
    # Default URL
    base_url = "http://localhost:8090"
    
    # Check if a URL was provided as an argument
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    run_tests(base_url)