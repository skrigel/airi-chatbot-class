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

# Import all backend components
from vector_store import VectorStore
from monitor import Monitor
from gemini_model import GeminiModel  # STEP 3: Add Gemini model

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
monitor = None
model = None  # STEP 3: Add model

try:
    # STEP 3: Initialize model
    logger.info(f"Initializing Gemini model: {GEMINI_MODEL}")
    model = GeminiModel(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL, repository_path=repository_path)
    logger.info("Model initialized successfully")
    
    # Initialize monitor
    logger.info("Initializing monitor...")
    monitor = Monitor(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    logger.info("Monitor initialized successfully")
    
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
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    model = None
    monitor = None
    vector_store = None

# Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    components_status = {
        "model": "ok" if model else "error",
        "monitor": "ok" if monitor else "error",
        "vector_store": "ok" if vector_store else "disabled",
    }
    
    overall_status = "degraded" if "error" in components_status.values() else "ok"
    
    return jsonify({
        "status": overall_status,
        "components": components_status,
        "model_name": GEMINI_MODEL
    })

@app.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received message: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    if not message:
        return jsonify({
            "id": conversation_id,
            "response": "Please provide a message.",
            "status": "error"
        }), 400
    
    try:
        # Use monitor to analyze the query
        inquiry_type = "GENERAL"
        override_attempt = False
        primary_domain = "OTHER"
        
        if monitor:
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
            except Exception as e:
                logger.error(f"Error in monitor: {str(e)}")
        
        # Get relevant documents
        context = None
        docs = []
        
        # Number of results to retrieve based on domain relevance
        k = 5  # Default
        if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
            k = 8  # More results for employment questions
        
        if vector_store:
            docs = vector_store.get_relevant_documents(message, k=k)
            if docs:
                # Format context based on inquiry type
                if inquiry_type == "EMPLOYMENT_RISK":
                    context = vector_store.format_context_for_domain(docs, "employment")
                else:
                    context = vector_store.format_context_from_docs(docs)
        
        # Generate response
        response_text = "Sorry, the model is not available."
        
        if model:
            # STEP 3: Generate response with model
            if context:
                # Different prompts for different inquiry types
                if inquiry_type == "EMPLOYMENT_RISK":
                    enhanced_message = f"""Looking for info on AI and jobs/employment.

{context}

Question: {message}

Give a detailed answer about how AI affects jobs and work, using the info above. Mention specific impacts and concerns from the repository. If you quote or reference something specific, say which entry it came from."""
                elif inquiry_type == "SPECIFIC_RISK":
                    enhanced_message = f"""The user wants to know about specific AI risks.

{context}

Question: {message}

Analyze the relevant risks from the info above. Be specific about what kinds of risks apply to their question and mention where the info comes from."""
                else:
                    enhanced_message = f"""Here's some relevant info from the MIT AI Risk Repository:

{context}

Question: {message}

Answer using the info above. If you use specific facts or quotes, mention which section or entry they came from."""
                
                response_text = model.generate_response(enhanced_message)
            else:
                # No context found but still try to help
                if inquiry_type == "EMPLOYMENT_RISK":
                    enhanced_message = f"Question about AI and jobs: {message} - Answer with what you know about AI's effects on employment and labor markets."
                    response_text = model.generate_response(enhanced_message)
                else:
                    # Just pass the question directly
                    response_text = model.generate_response(message)
        
        # Log conversation
        logger.info(f"Conversation {conversation_id} - Type: {inquiry_type}, Domain: {primary_domain}, Has context: {bool(context)}")
        
        return jsonify({
            "id": conversation_id,
            "response": response_text,
            "status": "complete"
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "id": conversation_id,
            "response": f"Sorry, there was an error processing your request: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/v1/stream', methods=['POST'])
def stream():
    """Streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received stream request: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    if not message:
        return Response(json.dumps({
            "id": conversation_id,
            "delta": "Please provide a message.",
            "status": "error"
        }) + '\n', mimetype='text/event-stream')
    
    # Define a streaming response generator
    def generate():
        # Initial stream setup - important to make frontend happy
        yield json.dumps({
            "id": conversation_id, 
            "delta": "", 
            "status": "processing"
        }) + '\n'
        
        # Initial status
        yield json.dumps({
            "id": conversation_id,
            "delta": "Processing your request...\n",
            "status": "processing"
        }) + '\n'
        time.sleep(0.3)
        
        try:
            # Use monitor to analyze the query
            inquiry_type = "GENERAL"
            primary_domain = "OTHER"
            
            if monitor:
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
            
            # Retrieve context
            context = None
            docs = []
            
            # Number of results to retrieve based on domain relevance
            k = 5  # Default
            if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
                k = 8  # More results for employment questions
            
            if vector_store:
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Retrieving relevant information...\n",
                    "status": "processing"
                }) + '\n'
                time.sleep(0.2)
                
                docs = vector_store.get_relevant_documents(message, k=k)
                if docs:
                    if inquiry_type == "EMPLOYMENT_RISK":
                        context = vector_store.format_context_for_domain(docs, "employment")
                    else:
                        context = vector_store.format_context_from_docs(docs)
            
            # Generate response
            yield json.dumps({
                "id": conversation_id,
                "delta": "Generating response...\n",
                "status": "processing"
            }) + '\n'
            time.sleep(0.2)
            
            if model:
                # STEP 3: Generate streaming response with model
                if context:
                    # Different prompts for different inquiry types
                    if inquiry_type == "EMPLOYMENT_RISK":
                        enhanced_message = f"""Looking for info on AI and jobs/employment.

{context}

Question: {message}

Give a detailed answer about how AI affects jobs and work, using the info above. Mention specific impacts and concerns from the repository. If you quote or reference something specific, say which entry it came from."""
                    elif inquiry_type == "SPECIFIC_RISK":
                        enhanced_message = f"""The user wants to know about specific AI risks.

{context}

Question: {message}

Analyze the relevant risks from the info above. Be specific about what kinds of risks apply to their question and mention where the info comes from."""
                    else:
                        enhanced_message = f"""Here's some relevant info from the MIT AI Risk Repository:

{context}

Question: {message}

Answer using the info above. If you use specific facts or quotes, mention which section or entry they came from."""
                    
                    # Get the complete response from the model
                    response_text = model.generate_response(enhanced_message, stream=True)
                    
                    # Break response into smaller chunks for streaming
                    # This provides a more natural typing effect for the frontend
                    words = response_text.split()
                    chunk_size = 5  # Send 5 words at a time
                    
                    for i in range(0, len(words), chunk_size):
                        end_idx = min(i + chunk_size, len(words))
                        chunk = ' '.join(words[i:end_idx])
                        
                        if i == 0:
                            # First chunk should have a space after
                            chunk += ' '
                        elif i > 0 and i < len(words) - chunk_size:
                            # Middle chunks should have a space after
                            chunk += ' '
                        
                        yield json.dumps({
                            "id": conversation_id,
                            "delta": chunk,
                            "status": "processing" if i < len(words) - chunk_size else "complete"
                        }) + '\n'
                        
                        # Slight delay to simulate typing
                        time.sleep(0.1)
                else:
                    # No context found but still try to help
                    if inquiry_type == "EMPLOYMENT_RISK":
                        enhanced_message = f"Question about AI and jobs: {message} - Answer with what you know about AI's effects on employment and labor markets."
                        response_text = model.generate_response(enhanced_message, stream=True)
                    else:
                        # Just pass the question directly
                        response_text = model.generate_response(message, stream=True)
                    
                    # Break response into smaller chunks for streaming
                    words = response_text.split()
                    chunk_size = 5  # Send 5 words at a time
                    
                    for i in range(0, len(words), chunk_size):
                        end_idx = min(i + chunk_size, len(words))
                        chunk = ' '.join(words[i:end_idx])
                        
                        if i == 0:
                            # First chunk should have a space after
                            chunk += ' '
                        elif i > 0 and i < len(words) - chunk_size:
                            # Middle chunks should have a space after
                            chunk += ' '
                        
                        yield json.dumps({
                            "id": conversation_id,
                            "delta": chunk,
                            "status": "processing" if i < len(words) - chunk_size else "complete"
                        }) + '\n'
                        
                        # Slight delay to simulate typing
                        time.sleep(0.1)
            else:
                yield json.dumps({
                    "id": conversation_id,
                    "delta": "Sorry, the model is not available.",
                    "status": "error"
                }) + '\n'
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield json.dumps({
                "id": conversation_id,
                "delta": f"Sorry, there was an error processing your request: {str(e)}",
                "status": "error"
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

# Reset conversation
@app.route('/api/v1/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    if not model:
        return jsonify({'error': 'Model not initialized'}), 500
        
    model.reset_conversation()
    return jsonify({'status': 'Conversation reset successfully'})

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