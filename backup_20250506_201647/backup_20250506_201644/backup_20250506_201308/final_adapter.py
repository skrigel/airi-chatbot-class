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

# Import all backend components with proper error handling
from vector_store import VectorStore
try:
    from monitor import Monitor as QueryMonitor
except ImportError:
    try:
        from monitor import QueryMonitor
    except ImportError:
        QueryMonitor = None
from gemini_model import GeminiModel

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

# Initialize backend components
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')

# 1. Initialize Vector Store (RAG system)
vector_store = None
try:
    if os.path.exists(repository_path):
        logger.info(f"Initializing vector store with repository: {repository_path}")
        vector_store = VectorStore(
            embedding_provider="google",
            api_key=GEMINI_API_KEY,
            repository_path=repository_path,
            persist_directory=persist_directory,
            use_hybrid_search=True  # Enable hybrid search (vector + keyword)
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

# 2. Initialize Query Monitor
query_monitor = None
try:
    if QueryMonitor is not None:
        logger.info("Initializing query monitor")
        query_monitor = QueryMonitor(api_key=GEMINI_API_KEY)
        logger.info("Query monitor initialized successfully")
    else:
        logger.warning("QueryMonitor class not found. Query type detection will be disabled.")
except Exception as e:
    logger.error(f"Error initializing query monitor: {str(e)}")
    query_monitor = None

# 3. Initialize Gemini Model
gemini_model = None
try:
    logger.info("Initializing Gemini model")
    gemini_model = GeminiModel(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash"  # Use more capable model
    )
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    gemini_model = None

# Conversation history (simple in-memory store)
conversations = {}

# Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    components_status = {
        "vector_store": "ok" if vector_store else "disabled",
        "query_monitor": "ok" if query_monitor else "disabled",
        "gemini_model": "ok" if gemini_model else "disabled"
    }
    
    if "disabled" in components_status.values():
        overall_status = "degraded"
    else:
        overall_status = "ok"
    
    return jsonify({
        "status": overall_status,
        "components": components_status
    })

@app.route('/api/v1/reset', methods=['POST'])
def reset():
    """Reset conversation endpoint - required by frontend"""
    # Get conversation ID from request
    data = request.json or {}
    conversation_id = data.get('conversationId', 'default')
    
    # Reset the conversation history
    if conversation_id in conversations:
        conversations[conversation_id] = []
    
    # Also reset the Gemini model if available
    if gemini_model:
        gemini_model.reset_conversation()
    
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
    
    # Initialize conversation history if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    # Full processing with all backend components
    response_text = process_query(message, conversation_id)
    
    # Add response to conversation history
    conversations[conversation_id].append({"role": "assistant", "content": response_text})
    
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
    
    # Initialize conversation history if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    # Define a streaming response generator
    def generate():
        # Initial status update
        yield json.dumps("Processing your query...") + '\n'
        time.sleep(0.3)
        
        # 1. Analyze the query with the monitor
        query_type = "general"
        domain = None
        
        if query_monitor:
            try:
                yield json.dumps("Analyzing your question...") + '\n'
                
                # Use the correct method based on the implementation
                if hasattr(query_monitor, 'analyze_query'):
                    query_analysis = query_monitor.analyze_query(message)
                    query_type = query_analysis.get('query_type', 'general')
                elif hasattr(query_monitor, 'determine_inquiry_type'):
                    inquiry_result = query_monitor.determine_inquiry_type(message)
                    query_type = inquiry_result.get('inquiry_type', 'GENERAL').lower()
                    domain = inquiry_result.get('primary_domain', 'OTHER').lower()
                    
                    # Map inquiry types to our internal types
                    if query_type == "employment_risk":
                        query_type = "employment"
                
                logger.info(f"Query type detected: {query_type}")
                if domain:
                    logger.info(f"Domain detected: {domain}")
            except Exception as e:
                logger.error(f"Error analyzing query: {str(e)}")
        
        # Fall back to basic detection for employment questions
        if not query_type or query_type == "general":
            if "job" in message.lower() or "employ" in message.lower() or "work" in message.lower():
                query_type = "employment"
                logger.info("Basic detection found employment related query")
        
        # 2. Get relevant documents using the vector store
        docs = []
        if vector_store:
            try:
                yield json.dumps("Searching repository for relevant information...") + '\n'
                docs = vector_store.get_relevant_documents(message, k=5)
                logger.info(f"Retrieved {len(docs)} relevant documents")
                
                if docs:
                    yield json.dumps(f"Found {len(docs)} relevant documents in the repository.") + '\n'
                else:
                    yield json.dumps("No specific documents found. Using general knowledge.") + '\n'
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
        
        # 3. Format context from documents
        context = ""
        if docs:
            try:
                # Use special formatting for employment-related queries
                if query_type == "employment" or domain == "socioeconomic":
                    if hasattr(vector_store, 'format_context_for_domain'):
                        context = vector_store.format_context_for_domain(docs, "employment")
                    else:
                        context = vector_store.format_context_from_docs(docs)
                else:
                    context = vector_store.format_context_from_docs(docs)
                logger.info(f"Formatted context with {len(context)} characters")
            except Exception as e:
                logger.error(f"Error formatting context: {str(e)}")
        
        # 4. Generate response using Gemini with retrieved context
        complete_response = ""
        
        if gemini_model:
            try:
                yield json.dumps("Generating response...") + '\n'
                
                # Prepare history for the model (if needed)
                history = conversations.get(conversation_id, [])[-5:]  # Last 5 messages
                
                # Convert our history format to the model's expected format if needed
                model_history = []
                for msg in history:
                    if msg['role'] == 'user':
                        model_history.append({"role": "user", "parts": [{"text": msg['content']}]})
                    elif msg['role'] == 'assistant':
                        model_history.append({"role": "model", "parts": [{"text": msg['content']}]})
                
                # Prepare the prompt with context
                if context:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on information from the MIT AI Risk Repository. 
Answer based on the retrieved context when possible. If the context doesn't contain relevant information, say so honestly.

Context: {context}

Question: {message}"""
                else:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
Answer the following question about AI risk: {message}"""
                
                # Use the correct streaming method based on implementation
                if hasattr(gemini_model, 'generate_stream'):
                    # Use generate_stream if available
                    for chunk in gemini_model.generate_stream(prompt, model_history):
                        yield json.dumps(chunk) + '\n'
                        complete_response += chunk
                        time.sleep(0.05)
                elif hasattr(gemini_model, 'generate_response'):
                    # Use generate_response with stream=True
                    response_generator = gemini_model.generate_response(prompt, stream=True)
                    
                    # Check if it returned a string (non-streaming response) or an iterable (streaming)
                    if isinstance(response_generator, str):
                        # Single response returned
                        yield json.dumps(response_generator) + '\n'
                        complete_response = response_generator
                    else:
                        try:
                            # Try to iterate through the response
                            for chunk in response_generator:
                                text_chunk = chunk if isinstance(chunk, str) else chunk.text if hasattr(chunk, 'text') else str(chunk)
                                yield json.dumps(text_chunk) + '\n'
                                complete_response += text_chunk
                                time.sleep(0.05)
                        except Exception as stream_err:
                            logger.error(f"Error streaming response: {str(stream_err)}")
                            # Fall back to non-streaming response
                            response = gemini_model.generate_response(prompt, stream=False)
                            yield json.dumps(response) + '\n'
                            complete_response = response
                else:
                    # No streaming method available, use regular generate
                    response = gemini_model.generate_response(prompt)
                    
                    # Break up the response into chunks for smoother display
                    words = response.split()
                    for i in range(0, len(words), 5):  # Send 5 words at a time
                        chunk = ' '.join(words[i:i+5]) + ' '
                        yield json.dumps(chunk) + '\n'
                        time.sleep(0.1)
                    
                    complete_response = response
                
                # If no content was generated, provide a fallback
                if not complete_response:
                    fallback = "I'm sorry, I couldn't generate a detailed response. The AI Risk Repository covers various risk domains including discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
                    yield json.dumps(fallback) + '\n'
                    complete_response = fallback
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                error_message = f"I encountered an error while generating a response: {str(e)}"
                yield json.dumps(error_message) + '\n'
                complete_response = error_message
        else:
            # Fallback if Gemini model is not available - use raw document content
            if docs:
                # More detailed response with multiple documents
                yield json.dumps(f"I found {len(docs)} relevant documents in the AI Risk Repository.") + '\n'
                time.sleep(0.3)
                
                for i, doc in enumerate(docs[:3]):  # Show up to 3 documents
                    # Format the document content
                    source = f"\n\n[Source: {doc.metadata.get('citation', 'MIT AI Risk Repository')}]"
                    content = f"\nDocument {i+1}:\n{doc.page_content[:800]}..." + source
                    
                    # Split content into chunks for a more natural streaming experience
                    words = content.split()
                    for j in range(0, len(words), 5):  # Send 5 words at a time
                        chunk = ' '.join(words[j:j+5]) + ' '
                        yield json.dumps(chunk) + '\n'
                        time.sleep(0.05)
                    
                    if i < len(docs[:3]) - 1:
                        yield json.dumps("\n\n") + '\n'
                        time.sleep(0.3)
                    
                    # Add to complete response
                    complete_response += content
            else:
                fallback = "I'm sorry, but I couldn't find specific information in the AI Risk Repository for your query. The repository covers risks related to discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
                yield json.dumps(fallback) + '\n'
                complete_response = fallback
        
        # Add the complete response to conversation history
        conversations[conversation_id].append({"role": "assistant", "content": complete_response})
    
    # Set proper headers for SSE
    return Response(
        stream_with_context(generate()), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream'
        }
    )

def process_query(message, conversation_id):
    """Process a query using all backend components (non-streaming)"""
    # 1. Analyze the query with the monitor
    query_type = "general"
    domain = None
    
    if query_monitor:
        try:
            # Use the correct method based on the implementation
            if hasattr(query_monitor, 'analyze_query'):
                query_analysis = query_monitor.analyze_query(message)
                query_type = query_analysis.get('query_type', 'general')
            elif hasattr(query_monitor, 'determine_inquiry_type'):
                inquiry_result = query_monitor.determine_inquiry_type(message)
                query_type = inquiry_result.get('inquiry_type', 'GENERAL').lower()
                domain = inquiry_result.get('primary_domain', 'OTHER').lower()
                
                # Map inquiry types to our internal types
                if query_type == "employment_risk":
                    query_type = "employment"
            
            logger.info(f"Query type detected: {query_type}")
            if domain:
                logger.info(f"Domain detected: {domain}")
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
    
    # Fall back to basic detection for employment questions
    if not query_type or query_type == "general":
        if "job" in message.lower() or "employ" in message.lower() or "work" in message.lower():
            query_type = "employment"
            logger.info("Basic detection found employment related query")
    
    # 2. Get relevant documents using the vector store
    docs = []
    if vector_store:
        try:
            docs = vector_store.get_relevant_documents(message, k=5)
            logger.info(f"Retrieved {len(docs)} relevant documents")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
    
    # 3. Format context from documents
    context = ""
    if docs:
        try:
            # Use special formatting for employment-related queries
            if query_type == "employment" or domain == "socioeconomic":
                if hasattr(vector_store, 'format_context_for_domain'):
                    context = vector_store.format_context_for_domain(docs, "employment")
                else:
                    context = vector_store.format_context_from_docs(docs)
            else:
                context = vector_store.format_context_from_docs(docs)
            logger.info(f"Formatted context with {len(context)} characters")
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
    
    # 4. Generate response using Gemini with retrieved context
    if gemini_model:
        try:
            # Prepare history for the model (if needed)
            history = conversations.get(conversation_id, [])[-5:]  # Last 5 messages
            
            # Convert our history format to the model's expected format if needed
            model_history = []
            for msg in history:
                if msg['role'] == 'user':
                    model_history.append({"role": "user", "parts": [{"text": msg['content']}]})
                elif msg['role'] == 'assistant':
                    model_history.append({"role": "model", "parts": [{"text": msg['content']}]})
            
            # Prepare the prompt with context
            if context:
                prompt = f"""You are an AI assistant that helps users understand AI risks based on information from the MIT AI Risk Repository. 
Answer based on the retrieved context when possible. If the context doesn't contain relevant information, say so honestly.

Context: {context}

Question: {message}"""
            else:
                prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
Answer the following question about AI risk: {message}"""
            
            # Use the correct method based on implementation
            if hasattr(gemini_model, 'generate'):
                response = gemini_model.generate(prompt, model_history)
            else:
                response = gemini_model.generate_response(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    else:
        # Fallback if Gemini model is not available - use raw document content
        if docs:
            # More detailed response with multiple documents
            response = f"I found {len(docs)} relevant documents in the AI Risk Repository.\n\n"
            
            for i, doc in enumerate(docs[:3]):  # Show up to 3 documents
                # Format the document content
                source = f"\n\n[Source: {doc.metadata.get('citation', 'MIT AI Risk Repository')}]"
                response += f"Document {i+1}:\n{doc.page_content[:800]}..." + source
                
                if i < len(docs[:3]) - 1:
                    response += "\n\n"
        else:
            response = "I'm sorry, but I couldn't find specific information in the AI Risk Repository for your query. The repository covers risks related to discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
        
        return response

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
    # Get port from environment variables or use default port 5000
    # (matching the frontend's hardcoded port)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server at http://localhost:{port}")
    print(f"Serving frontend from: {frontend_path}")
    print(f"Using repository path: {repository_path}")
    print("\nBackend components status:")
    print(f"- Vector Store: {'Active' if vector_store else 'Disabled'}")
    print(f"- Query Monitor: {'Active' if query_monitor else 'Disabled'}")
    print(f"- Gemini Model: {'Active' if gemini_model else 'Disabled'}")
    
    # Run the Flask application with the specified port
    app.run(host='0.0.0.0', port=port, debug=True)