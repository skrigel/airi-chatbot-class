#!/usr/bin/env python
import requests
import json
import time
import sys

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/api/health")
    if response.status_code == 200:
        print("✅ Health endpoint working")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Health endpoint failed: {response.status_code}")
        print(f"Response: {response.text}")
    print()

def test_message_endpoint(base_url, message="What are the main risks of AI?"):
    """Test the message endpoint"""
    print(f"Testing message endpoint with: '{message}'")
    
    payload = {
        "message": message,
        "conversationId": "test-conversation"
    }
    
    response = requests.post(
        f"{base_url}/api/v1/sendMessage",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("✅ Message endpoint working")
        # Print just the beginning of the response for brevity
        response_json = response.json()
        if "response" in response_json:
            response_text = response_json["response"]
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print(f"Response preview: {preview}")
        else:
            print(f"Response: {response_json}")
    else:
        print(f"❌ Message endpoint failed: {response.status_code}")
        print(f"Response: {response.text}")
    print()

def test_stream_endpoint(base_url, message="Tell me about employment risks from AI"):
    """Test the streaming endpoint"""
    print(f"Testing stream endpoint with: '{message}'")
    
    payload = {
        "message": message,
        "conversationId": "test-conversation"
    }
    
    try:
        with requests.post(
            f"{base_url}/api/v1/stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        ) as response:
            if response.status_code == 200:
                print("✅ Stream endpoint connected")
                print("Streaming response (first few chunks):")
                
                # Just show the first few chunks for brevity
                chunks_shown = 0
                max_chunks = 3
                
                for line in response.iter_lines():
                    if line:
                        chunks_shown += 1
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            if "delta" in chunk_data:
                                preview = chunk_data["delta"][:50] + "..." if len(chunk_data["delta"]) > 50 else chunk_data["delta"]
                                print(f"Chunk {chunks_shown}: {preview}")
                            else:
                                print(f"Chunk {chunks_shown}: {chunk_data}")
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line.decode('utf-8')}")
                        
                        if chunks_shown >= max_chunks:
                            print("... more chunks available ...")
                            break
                
                print("✅ Stream endpoint working")
            else:
                print(f"❌ Stream endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Stream endpoint error: {str(e)}")
    print()

def test_reset_endpoint(base_url):
    """Test the reset endpoint"""
    print("Testing reset endpoint...")
    response = requests.post(f"{base_url}/api/v1/reset")
    if response.status_code == 200:
        print("✅ Reset endpoint working")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Reset endpoint failed: {response.status_code}")
        print(f"Response: {response.text}")
    print()

def run_tests(base_url):
    """Run all API tests"""
    print(f"🧪 Testing adapter API at {base_url} 🧪")
    print("=" * 60)
    print()
    
    test_health_endpoint(base_url)
    test_message_endpoint(base_url)
    test_stream_endpoint(base_url)
    test_reset_endpoint(base_url)
    
    print("✨ Testing complete ✨")

if __name__ == "__main__":
    base_url = "http://localhost:8090"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    run_tests(base_url)