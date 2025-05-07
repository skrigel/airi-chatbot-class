from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
import logging
import time
from dotenv import load_dotenv
from pathlib import Path

from gemini_model import GeminiModel
from vector_store import VectorStore
from monitor import Monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load API key from environment or use the default for testing
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.0-flash')

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
persist_directory = os.path.join(current_dir, 'chroma_db')

# Initialize components
try:
    # Initialize model
    model = GeminiModel(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL, repository_path=repository_path)
    logger.info(f"Initialized Gemini model: {GEMINI_MODEL}")
    
    # Initialize monitor
    monitor = Monitor(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    logger.info("Initialized Monitor")
    
    # Initialize vector store if path exists
    vector_store = None
    if os.path.exists(repository_path):
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
    else:
        logger.warning(f"Repository path {repository_path} not found")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    model = None
    monitor = None
    vector_store = None

# Routes for static files
@app.route('/')
def index():
    """Serve the index.html file"""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error serving index.html: {e}", 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# API routes
@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint for chat interactions"""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    user_message = data['message']
    conversation_id = data.get('conversation_id', 'default')
    stream_enabled = data.get('stream', False)
    
    if stream_enabled:
        return Response(stream_with_context(generate_streaming_response(user_message, conversation_id)),
                      content_type='text/event-stream')
    else:
        response_data = generate_response(user_message, conversation_id)
        return jsonify(response_data)

def generate_streaming_response(user_message, conversation_id):
    """Generate a streaming response with status updates."""
    # Step 1: Send initial status
    yield json_to_stream({"status": "processing", "message": "Initializing..."})
    time.sleep(0.5)  # Small delay for UX
    
    # Step 2: Run monitor
    yield json_to_stream({"status": "processing", "message": "Analyzing your question..."})
    
    inquiry_type = "GENERAL"
    primary_domain = "OTHER"
    
    try:
        if monitor:
            result = monitor.determine_inquiry_type(user_message)
            inquiry_type = result.get("inquiry_type", "GENERAL")
            override_attempt = result.get("override_attempt", False)
            primary_domain = result.get("primary_domain", "OTHER")
            
            # Reject inappropriate requests
            if override_attempt:
                yield json_to_stream({
                    "status": "error", 
                    "message": "I'm sorry, but I'm unable to process that request as it appears to be outside my operational guidelines."
                })
                return
                
            # Send status update based on inquiry type
            if inquiry_type == "SPECIFIC_RISK":
                yield json_to_stream({"status": "processing", "message": "Searching risk repository..."})
            elif inquiry_type == "EMPLOYMENT_RISK":
                yield json_to_stream({"status": "processing", "message": "Searching for employment-related AI risks..."})
            elif inquiry_type == "RECOMMENDATION":
                yield json_to_stream({"status": "processing", "message": "Generating recommendations..."})
            else:
                yield json_to_stream({"status": "processing", "message": "Processing your question..."})
        else:
            # Skip monitor if not available
            yield json_to_stream({"status": "processing", "message": "Processing your question..."})
    except Exception as e:
        logger.error(f"Error in monitor: {str(e)}")
        yield json_to_stream({"status": "processing", "message": "Processing your question..."})
    
    time.sleep(0.5)  # Small delay for UX
    
    # Step 3: Retrieve context
    context = None
    docs = []
    
    # Number of results to retrieve based on domain relevance
    k = 5  # Default
    if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
        k = 8  # More results for employment questions
    
    if vector_store:
        yield json_to_stream({"status": "processing", "message": "Retrieving relevant information..."})
        docs = vector_store.get_relevant_documents(user_message, k=k)
        if docs:
            context = vector_store.format_context_from_docs(docs)
    
    time.sleep(0.5)  # Small delay for UX
    
    # Step 4: Generate response
    yield json_to_stream({"status": "processing", "message": "Generating response..."})
    
    try:
        # Enhance the prompt based on the inquiry type
        if context:
            # Different prompts for different inquiry types
            if inquiry_type == "EMPLOYMENT_RISK":
                # Use specialized formatting for employment-related queries
                employment_context = vector_store.format_context_for_domain(docs, "employment")
                
                enhanced_message = f"""Looking for info on AI and jobs/employment.

{employment_context}

Question: {user_message}

Give a detailed answer about how AI affects jobs and work, using the info above. Mention specific impacts and concerns from the repository. If you quote or reference something specific, say which entry it came from."""
            elif inquiry_type == "SPECIFIC_RISK":
                enhanced_message = f"""The user wants to know about specific AI risks.

{context}

Question: {user_message}

Analyze the relevant risks from the info above. Be specific about what kinds of risks apply to their question and mention where the info comes from."""
            else:
                enhanced_message = f"""Here's some relevant info from the MIT AI Risk Repository:

{context}

Question: {user_message}

Answer using the info above. If you use specific facts or quotes, mention which section or entry they came from."""
            
            response_text = model.generate_response(enhanced_message, stream=True)
        else:
            # No context found but still try to help
            if inquiry_type == "EMPLOYMENT_RISK":
                enhanced_message = f"Question about AI and jobs: {user_message} - Answer with what you know about AI's effects on employment and labor markets."
                response_text = model.generate_response(enhanced_message, stream=True)
            else:
                # Just pass the question directly
                response_text = model.generate_response(user_message, stream=True)
        
        # Final response
        yield json_to_stream({
            "status": "complete",
            "response": response_text,
            "conversation_id": conversation_id
        })
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield json_to_stream({
            "status": "error",
            "message": "Sorry, there was an error processing your request. Please try again."
        })

def json_to_stream(data):
    """Convert data to SSE format."""
    return f"data: {json.dumps(data)}\n\n"

def generate_response(user_message, conversation_id):
    """Generate a non-streaming response."""
    try:
        # Step 1: Run monitor
        inquiry_type = "GENERAL"
        override_attempt = False
        primary_domain = "OTHER"
        
        if monitor:
            try:
                result = monitor.determine_inquiry_type(user_message)
                inquiry_type = result.get("inquiry_type", "GENERAL")
                override_attempt = result.get("override_attempt", False)
                primary_domain = result.get("primary_domain", "OTHER")
                
                # Reject inappropriate requests
                if override_attempt:
                    return {
                        "response": "I'm sorry, but I'm unable to process that request as it appears to be outside my operational guidelines.",
                        "conversation_id": conversation_id
                    }
            except Exception as e:
                logger.error(f"Error in monitor: {str(e)}")
        
        # Step 2: Retrieve context
        context = None
        
        # Number of results to retrieve based on domain relevance
        k = 5  # Default
        if inquiry_type == "EMPLOYMENT_RISK" or primary_domain == "SOCIOECONOMIC":
            k = 8  # More results for employment questions
        
        if vector_store:
            docs = vector_store.get_relevant_documents(user_message, k=k)
            if docs:
                context = vector_store.format_context_from_docs(docs)
        
        # Step 3: Generate response
        if context:
            # Different prompts for different inquiry types
            if inquiry_type == "EMPLOYMENT_RISK":
                # Use specialized formatting for employment-related queries
                employment_context = vector_store.format_context_for_domain(docs, "employment")
                
                enhanced_message = f"""Looking for info on AI and jobs/employment.

{employment_context}

Question: {user_message}

Give a detailed answer about how AI affects jobs and work, using the info above. Mention specific impacts and concerns from the repository. If you quote or reference something specific, say which entry it came from."""
            elif inquiry_type == "SPECIFIC_RISK":
                enhanced_message = f"""The user wants to know about specific AI risks.

{context}

Question: {user_message}

Analyze the relevant risks from the info above. Be specific about what kinds of risks apply to their question and mention where the info comes from."""
            else:
                enhanced_message = f"""Here's some relevant info from the MIT AI Risk Repository:

{context}

Question: {user_message}

Answer using the info above. If you use specific facts or quotes, mention which section or entry they came from."""
            
            response_text = model.generate_response(enhanced_message)
        else:
            # No context but try to be helpful anyway
            if inquiry_type == "EMPLOYMENT_RISK":
                enhanced_message = f"Question about AI and jobs: {user_message} - Answer with what you know about AI's effects on employment and labor markets."
                response_text = model.generate_response(enhanced_message)
            else:
                # Pass question directly
                response_text = model.generate_response(user_message)
        
        # Log conversation
        logger.info(f"Conversation {conversation_id} - Type: {inquiry_type}, Domain: {primary_domain}, Has context: {bool(context)}")
        
        return {
            "response": response_text,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            "response": "Sorry, there was an error processing your request. Please try again.",
            "conversation_id": conversation_id
        }

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Endpoint to reset conversation history"""
    if not model:
        return jsonify({'error': 'Model not initialized'}), 500
        
    model.reset_conversation()
    return jsonify({'status': 'Conversation reset successfully'})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    components_status = {
        "model": "ok" if model else "error",
        "monitor": "ok" if monitor else "error",
        "vector_store": "ok" if vector_store else "disabled",
    }
    
    overall_status = "degraded" if "error" in components_status.values() else "ok"
    
    return jsonify({
        'status': overall_status,
        'components': components_status,
        'model_name': GEMINI_MODEL
    })

# Optional route to manually trigger document ingestion
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

# Debug route
@app.route('/debug', methods=['GET'])
def debug():
    """Debug route to check API status"""
    result = {
        'status': 'API is running',
        'configured_endpoints': [
            '/',
            '/static/<path>',
            '/api/chat',
            '/api/reset',
            '/api/health',
            '/api/ingest',
            '/debug',
            '/test'
        ],
        'components': {
            'model': 'ok' if model else 'error',
            'monitor': 'ok' if monitor else 'error', 
            'vector_store': 'ok' if vector_store else 'error'
        }
    }
    return jsonify(result)

# Simple test endpoint for frontend debugging
@app.route('/test', methods=['GET', 'POST'])
def test():
    """Simple test endpoint that always returns success"""
    if request.method == 'GET':
        return jsonify({'status': 'success', 'message': 'Test endpoint is working'})
    else:
        data = request.get_json(silent=True) or {}
        return jsonify({
            'status': 'success',
            'message': 'Test endpoint received POST request',
            'data_received': data,
            'test_response': 'This is a test response from the server'
        })

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8090))  # Changed to use port 8090
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # Run the app
    app.run(host=host, port=port, debug=debug)