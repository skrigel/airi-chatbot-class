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

# Import the vector_store and monitor
from vector_store import VectorStore
from monitor import Monitor  # STEP 2: Add monitor

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
    logger.error("Please run integration.sh first to set up the frontend")
    sys.exit(1)

# Initialize components
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
vector_store = None
monitor = None  # STEP 2: Add monitor

try:
    # Initialize vector store
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
    
    # STEP 2: Initialize monitor
    logger.info("Initializing monitor...")
    monitor = Monitor(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    logger.info("Monitor initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    vector_store = None
    monitor = None

# Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    components_status = {
        "vector_store": "ok" if vector_store else "disabled",
        "monitor": "ok" if monitor else "error",  # STEP 2: Add monitor status
    }
    
    overall_status = "degraded" if "error" in components_status.values() else "ok"
    
    return jsonify({
        "status": overall_status,
        "components": components_status
    })

@app.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received message: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    # STEP 2: Use monitor to analyze the query
    inquiry_type = "GENERAL"
    primary_domain = "OTHER"
    
    if monitor and message:
        try:
            result = monitor.determine_inquiry_type(message)
            inquiry_type = result.get("inquiry_type", "GENERAL")
            override_attempt = result.get("override_attempt", False)
            primary_domain = result.get("primary_domain", "OTHER")
            
            # Reject inappropriate requests
            if override_attempt:
                return jsonify({
                    "id": conversation_id,
                    "response": "I'm sorry, but I'm unable to process that request as it appears to be outside my operational guidelines.",
                    "status": "error"
                })
                
            logger.info(f"Monitor classified message as: {inquiry_type}, domain: {primary_domain}")
        except Exception as e:
            logger.error(f"Error in monitor: {str(e)}")
    
    # Get relevant documents based on domain
    if vector_store and message:
        try:
            # Number of results to retrieve based on domain relevance
            k = 5  # Default
            if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
                k = 8  # More results for employment questions
            
            # Get relevant documents
            docs = vector_store.get_relevant_documents(message, k=k)
            
            if docs:
                # Format the context based on inquiry type
                if inquiry_type == "EMPLOYMENT_RISK":
                    # Use specialized formatting for employment-related queries
                    context = vector_store.format_context_for_domain(docs, "employment")
                else:
                    context = vector_store.format_context_from_docs(docs)
                
                # Create a user-friendly response
                intro = f"I found {len(docs)} relevant documents about "
                
                if inquiry_type == "EMPLOYMENT_RISK":
                    intro += "AI's impact on employment and labor markets. "
                elif inquiry_type == "SPECIFIC_RISK":
                    intro += "specific AI risks you asked about. "
                else:
                    intro += "AI risks in the repository. "
                
                # Prepare document content
                doc_content = f"Here's what I found:\n\n{docs[0].page_content[:200]}..."
                if 'citation' in docs[0].metadata:
                    doc_content += f"\n\n[Source: {docs[0].metadata['citation']}]"
                
                response_text = intro + doc_content
            else:
                # No documents found response
                response_text = "I couldn't find any relevant documents for your query about "
                if inquiry_type == "EMPLOYMENT_RISK":
                    response_text += "AI's impact on employment. Could you try rephrasing your question?"
                elif inquiry_type == "SPECIFIC_RISK":
                    response_text += "specific AI risks. Could you try asking about a different risk area?"
                else:
                    response_text += "AI risks. Could you try asking in a different way?"
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
        yield json.dumps({
            "id": conversation_id, 
            "delta": "", 
            "status": "processing"
        }) + '\n'
        
        # Simulate processing steps
        yield json.dumps({
            "id": conversation_id,
            "delta": "Processing your request...\n",
            "status": "processing"
        }) + '\n'
        time.sleep(0.3)
        
        # STEP 2: Use monitor to analyze the query
        inquiry_type = "GENERAL"
        primary_domain = "OTHER"
        
        if monitor and message:
            try:
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Analyzing your question...\n",
                    "status": "processing"
                }) + '\n'
                time.sleep(0.3)
                
                result = monitor.determine_inquiry_type(message)
                inquiry_type = result.get("inquiry_type", "GENERAL")
                override_attempt = result.get("override_attempt", False)
                primary_domain = result.get("primary_domain", "OTHER")
                
                # Reject inappropriate requests
                if override_attempt:
                    yield json.dumps({
                        "id": conversation_id,
                        "delta": "I'm sorry, but I'm unable to process that request as it appears to be outside my operational guidelines.",
                        "status": "error"
                    }) + '\n'
                    return
                
                # Send status update based on inquiry type
                status_message = ""
                if inquiry_type == "SPECIFIC_RISK":
                    status_message = "Searching risk repository for specific risks...\n"
                elif inquiry_type == "EMPLOYMENT_RISK":
                    status_message = "Searching for employment-related AI risks...\n"
                elif inquiry_type == "RECOMMENDATION":
                    status_message = "Generating recommendations...\n"
                else:
                    status_message = "Processing your question...\n"
                
                yield json.dumps({
                    "id": conversation_id,
                    "delta": status_message,
                    "status": "processing"
                }) + '\n'
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error in monitor: {str(e)}")
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Processing your question...\n",
                    "status": "processing"
                }) + '\n'
        
        # Use vector store to get relevant documents
        if vector_store and message:
            try:
                # Send status update
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Searching repository...\n",
                    "status": "processing"
                }) + '\n'
                time.sleep(0.2)
                
                # Number of results to retrieve based on domain relevance
                k = 5  # Default
                if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
                    k = 8  # More results for employment questions
                
                # Get relevant documents
                docs = vector_store.get_relevant_documents(message, k=k)
                
                if docs:
                    # Format the context based on inquiry type
                    if inquiry_type == "EMPLOYMENT_RISK":
                        # Use specialized formatting for employment-related queries
                        context = vector_store.format_context_for_domain(docs, "employment")
                    else:
                        context = vector_store.format_context_from_docs(docs)
                    
                    # Stream response in a more natural way
                    intro = f"I found {len(docs)} relevant documents about "
                    
                    if inquiry_type == "EMPLOYMENT_RISK":
                        intro += "AI's impact on employment and labor markets. "
                    elif inquiry_type == "SPECIFIC_RISK":
                        intro += "specific AI risks you asked about. "
                    else:
                        intro += "AI risks in the repository. "
                        
                    # Send the introduction
                    yield json.dumps({
                        "id": conversation_id,
                        "delta": intro,
                        "status": "processing"
                    }) + '\n'
                    time.sleep(0.2)
                    
                    # Prepare document content
                    doc_content = f"Here's what I found:\n\n{docs[0].page_content[:200]}..."
                    if 'citation' in docs[0].metadata:
                        doc_content += f"\n\n[Source: {docs[0].metadata['citation']}]"
                    
                    # Stream content word by word
                    words = doc_content.split()
                    for i in range(0, len(words), 3):
                        chunk = ' '.join(words[i:i+3]) + ' '
                        yield json.dumps({
                            "id": conversation_id,
                            "delta": chunk,
                            "status": "processing" if i < len(words) - 3 else "complete"
                        }) + '\n'
                        time.sleep(0.1)
                else:
                    # No documents found response
                    response = "I couldn't find any relevant documents for your query about "
                    if inquiry_type == "EMPLOYMENT_RISK":
                        response += "AI's impact on employment. Could you try rephrasing your question?"
                    elif inquiry_type == "SPECIFIC_RISK":
                        response += "specific AI risks. Could you try asking about a different risk area?"
                    else:
                        response += "AI risks. Could you try asking in a different way?"
                        
                    yield json.dumps({
                        "id": conversation_id,
                        "delta": response,
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
        generate(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
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
            return send_from_directory(frontend_path, 'index.html')
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print(f"Starting server at http://localhost:8090")
    print(f"Serving frontend from: {frontend_path}")
    print(f"Using repository path: {repository_path}")
    app.run(host='0.0.0.0', port=8090, debug=True)