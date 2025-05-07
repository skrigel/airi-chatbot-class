#!/usr/bin/env python
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import os
import logging
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# STEP 1: Import the vector_store
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create Flask app with CORS
app = Flask(__name__)
CORS(app)

# Get directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, 'github-frontend-build')
repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
persist_directory = os.path.join(current_dir, 'chroma_db')

# Check if frontend directory exists
if not os.path.exists(frontend_path):
    logger.error(f"Frontend directory not found: {frontend_path}")
    logger.info("This is not critical, the backend API will still function without the frontend.")

# STEP 1: Initialize vector store
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')
vector_store = None

try:
    if os.path.exists(repository_path):
        logger.info(f"Initializing vector store with repository: {repository_path}")
        vector_store = VectorStore(
            embedding_provider="google",
            api_key=GEMINI_API_KEY,
            repository_path=repository_path,
            persist_directory=persist_directory
        )
        
        # Ingest documents if not already done
        if not os.path.exists(persist_directory):
            logger.info("Ingesting documents into vector store...")
            vector_store.ingest_documents()
        
        logger.info("Vector store initialized successfully")
    else:
        logger.warning(f"Repository path {repository_path} not found")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    vector_store = None

# Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    components_status = {
        "vector_store": "ok" if vector_store else "disabled",
    }
    
    overall_status = "degraded" if "error" in components_status.values() else "ok"
    
    return jsonify({
        "status": overall_status,
        "components": components_status
    })

@app.route('/api/v1/reset', methods=['POST'])
def reset():
    """Reset conversation endpoint - required by frontend"""
    return jsonify({
        "success": True, 
        "message": "Conversation reset successfully"
    })

@app.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received message: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    # STEP 1: Use vector store to get relevant documents
    response_text = "This is a test response"
    
    if vector_store and message:
        try:
            # Get relevant documents
            docs = vector_store.get_relevant_documents(message, k=5)
            
            if docs:
                # Format the context
                context = vector_store.format_context_from_docs(docs)
                # Create a more user-friendly response
                response_text = f"I found {len(docs)} relevant documents in the AI Risk Repository. Here's what I found:\n\n{docs[0].page_content[:200]}..."
                
                # Add citation for better UX
                if 'citation' in docs[0].metadata:
                    response_text += f"\n\n[Source: {docs[0].metadata['citation']}]"
            else:
                response_text = "I couldn't find any relevant documents in the repository for your query. Could you try rephrasing your question?"
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            response_text = f"I'm sorry, I encountered an error while searching for information: {str(e)}"
    else:
        response_text = "I'm sorry, but I can't access the AI Risk Repository at the moment. Please try again later."
    
    return jsonify({
        "id": conversation_id,
        "response": response_text,
        "status": "complete"
    })

@app.route('/api/v1/stream', methods=['POST'])
def stream():
    """Streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received stream request: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    # Define a streaming response generator
    def generate():
        # Initial stream setup - important to make frontend happy
        # Send a single event with status: "processing"
        yield json.dumps({
            "id": conversation_id, 
            "delta": "", 
            "status": "processing"
        }) + '\n'
        
        # Simulate processing steps with actual content
        yield json.dumps({
            "id": conversation_id,
            "delta": "Processing your request...\n",
            "status": "processing"
        }) + '\n'
        time.sleep(0.3)
        
        # STEP 1: Use vector store to get relevant documents
        if vector_store and message:
            try:
                # Send status update
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Searching repository...\n",
                    "status": "processing"
                }) + '\n'
                time.sleep(0.3)
                
                # Get relevant documents
                docs = vector_store.get_relevant_documents(message, k=5)
                
                if docs:
                    # Format the context - break into small chunks for streaming
                    context = vector_store.format_context_from_docs(docs)
                    response_text = f"I found {len(docs)} relevant documents in the AI Risk Repository. "
                    
                    # Stream the response in small chunks to better simulate typing
                    yield json.dumps({
                        "id": conversation_id,
                        "delta": response_text,
                        "status": "processing"
                    }) + '\n'
                    time.sleep(0.2)
                    
                    # Add first document summary
                    doc_preview = f"Here's what I found:\n\n{docs[0].page_content[:150]}..."
                    words = doc_preview.split()
                    
                    # Stream word by word for a more natural feel
                    for i in range(0, len(words), 3):
                        chunk = ' '.join(words[i:i+3]) + ' '
                        yield json.dumps({
                            "id": conversation_id,
                            "delta": chunk,
                            "status": "processing" if i < len(words) - 3 else "complete"
                        }) + '\n'
                        time.sleep(0.1)
                else:
                    yield json.dumps({
                        "id": conversation_id,
                        "delta": "I couldn't find any relevant documents in the repository for your query. Could you try rephrasing your question?",
                        "status": "complete"
                    }) + '\n'
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                yield json.dumps({
                    "id": conversation_id,
                    "delta": f"I'm sorry, I encountered an error while searching for information: {str(e)}",
                    "status": "error"
                }) + '\n'
        else:
            # Fallback if vector store is not available
            yield json.dumps({
                "id": conversation_id,
                "delta": "I'm sorry, but I can't access the AI Risk Repository at the moment. Please try again later.",
                "status": "complete"
            }) + '\n'
    
    # Set proper headers for SSE
    return Response(
        stream_with_context(generate()), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream'  # Make sure content type is correctly set
        }
    )

# Route to manually trigger document ingestion
@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Manually trigger document ingestion"""
    if not vector_store:
        return jsonify({'error': 'Vector store not initialized'}), 500
    
    success = vector_store.ingest_documents()
    
    if success:
        return jsonify({'status': 'Documents ingested successfully'})
    else:
        return jsonify({'error': 'Failed to ingest documents'}), 500

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
            if os.path.exists(os.path.join(frontend_path, 'index.html')):
                return send_from_directory(frontend_path, 'index.html')
            else:
                return "Frontend not found. Please run integration.sh first.", 404
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print(f"Starting server at http://localhost:8090")
    print(f"Serving frontend from: {frontend_path}")
    print(f"Using repository path: {repository_path}")
    app.run(host='0.0.0.0', port=8090, debug=True)