#!/usr/bin/env python
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import os
import logging
import json
import time
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app with CORS
app = Flask(__name__)
CORS(app)

# Get directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, 'github-frontend-build')

# Check if frontend directory exists
if not os.path.exists(frontend_path):
    logger.error(f"Frontend directory not found: {frontend_path}")
    logger.error("Please run integration.sh first to set up the frontend")
    sys.exit(1)

# Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received message: {data}")
    
    # Basic response for now
    return jsonify({
        "id": "test-id",
        "response": "This is a test response from the simplified adapter",
        "status": "complete"
    })

@app.route('/api/v1/stream', methods=['POST'])
def stream():
    """Streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received stream request: {data}")
    
    # Define a simple streaming response generator
    def generate():
        chunks = [
            "This ", "is ", "a ", "test ", "streaming ", "response ", 
            "from ", "the ", "simplified ", "adapter."
        ]
        
        for chunk in chunks:
            yield json.dumps({
                "id": "test-id",
                "delta": chunk,
                "status": "processing" if chunk != chunks[-1] else "complete"
            }) + '\n'
            time.sleep(0.2)  # Simulate delay between chunks
    
    return Response(generate(), mimetype='text/event-stream')

# Serve frontend files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the React app (GitHub frontend)"""
    logger.info(f"Serving path: {path}")
    
    try:
        if path and os.path.exists(os.path.join(frontend_path, path)):
            logger.info(f"Serving file: {path}")
            return send_from_directory(frontend_path, path)
        elif path and '.' in path:  # File with extension that doesn't exist
            logger.warning(f"File not found: {path}")
            return f"File not found: {path}", 404
        else:
            # Try to serve index.html for all other routes (SPA routing)
            logger.info(f"Serving index.html for path: {path}")
            return send_from_directory(frontend_path, 'index.html')
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print(f"Starting server at http://localhost:8090")
    print(f"Serving frontend from: {frontend_path}")
    app.run(host='0.0.0.0', port=8090, debug=True)