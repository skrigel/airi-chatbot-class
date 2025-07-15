"""
Chat routes for the AIRI chatbot API.
"""
import json
import time
from flask import Blueprint, request, jsonify, Response, stream_with_context

from ...core.services.chat_service import ChatService
from ...config.logging import get_logger

logger = get_logger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# This will be injected by the app factory
chat_service: ChatService = None

def init_chat_routes(chat_service_instance: ChatService):
    """Initialize chat routes with service dependency."""
    global chat_service
    chat_service = chat_service_instance

@chat_bp.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint."""
    try:
        data = request.json
        
        # Log the incoming request
        logger.info(f"Received message: {data}")
        message = data.get('message', '')
        conversation_id = data.get('conversationId', 'default')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Process the query
        response_text, docs = chat_service.process_query(message, conversation_id)
        
        return jsonify({
            "id": conversation_id,
            "response": response_text,
            "status": "complete"
        })
        
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@chat_bp.route('/api/v1/stream', methods=['POST'])
def stream_message():
    """Streaming chat endpoint."""
    try:
        data = request.json
        
        # Log the incoming request
        logger.info(f"Received stream request: {data}")
        message = data.get('message', '')
        conversation_id = data.get('conversationId', 'default')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Define a streaming response generator
        def generate():
            try:
                # Initialize variables
                query_type = "general"
                domain = None
                docs = []
                
                # Initial status update with system info
                yield json.dumps("Processing your query...") + '\n'
                
                # Add system version info to logs
                from ...config.settings import settings
                logger.info(f"🔧 Processing with Field-Aware Hybrid: {settings.USE_FIELD_AWARE_HYBRID}")
                if chat_service.vector_store and hasattr(chat_service.vector_store, 'hybrid_retriever'):
                    if chat_service.vector_store.hybrid_retriever:
                        retriever_class = chat_service.vector_store.hybrid_retriever.__class__.__name__
                        logger.info(f"🎯 Using retriever: {retriever_class}")
                
                time.sleep(0.3)
                
                # 1. Analyze the query
                yield json.dumps("Analyzing your question...") + '\n'
                query_type, domain = chat_service.query_processor.analyze_query(message)
                
                # 2. Retrieve documents
                yield json.dumps("Searching repository for relevant information...") + '\n'
                docs = chat_service._retrieve_documents(message, query_type)
                
                if docs:
                    if query_type in ["employment", "socioeconomic"]:
                        employment_count = len([doc for doc in docs if any(keyword in doc.metadata.get('domain', '').lower() + doc.metadata.get('subdomain', '').lower() + doc.metadata.get('specific_domain', '').lower() 
                                                                          for keyword in ['employ', 'job', 'work', 'labor', 'socioeconomic', 'economic', 'inequality'])])
                        if employment_count > 0:
                            yield json.dumps(f"Found {employment_count} employment-specific documents and {len(docs) - employment_count} additional relevant documents.") + '\n'
                        else:
                            yield json.dumps(f"Found {len(docs)} relevant documents in the repository.") + '\n'
                    else:
                        yield json.dumps(f"Found {len(docs)} relevant documents in the repository.") + '\n'
                else:
                    yield json.dumps("No specific documents found. Using general knowledge.") + '\n'
                
                # 3. Format context
                context = chat_service._format_context(docs, query_type)
                
                # 4. Generate response
                yield json.dumps("Generating response...") + '\n'
                
                if chat_service.gemini_model:
                    try:
                        # Prepare history
                        history = chat_service._get_conversation_history(conversation_id)
                        
                        # Generate enhanced prompt
                        prompt = chat_service.query_processor.generate_prompt(message, query_type, context)
                        
                        # Generate complete response first
                        if hasattr(chat_service.gemini_model, 'generate_stream'):
                            # For now, collect all chunks then enhance citations
                            complete_response = ""
                            for chunk in chat_service.gemini_model.generate_stream(prompt, history):
                                complete_response += chunk
                        else:
                            # Non-streaming generation
                            complete_response = chat_service.gemini_model.generate(prompt, history)
                        
                        # Apply citations to complete response
                        enhanced_response = chat_service.citation_service.enhance_response_with_citations(complete_response, docs)
                        
                        # Now stream the enhanced response with proper citations
                        words = enhanced_response.split()
                        for i in range(0, len(words), 5):  # Send 5 words at a time
                            chunk = ' '.join(words[i:i+5]) + ' '
                            yield json.dumps(chunk) + '\n'
                            time.sleep(0.1)
                        
                        # Send the related documents
                        if docs:
                            related_docs = [{"title": doc.metadata.get("title", "Unknown Title"), "url": doc.metadata.get("url", "#")} for doc in docs]
                            yield json.dumps({"related_documents": related_docs}) + '\n'

                        # Update conversation history
                        chat_service._update_conversation_history(conversation_id, message, enhanced_response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        error_message = f"I encountered an error while generating a response: {str(e)}"
                        yield json.dumps(error_message) + '\n'
                else:
                    # Fallback if Gemini model is not available
                    fallback_response = chat_service._create_fallback_response(context, message)
                    
                    # Stream the fallback response
                    words = fallback_response.split()
                    for i in range(0, len(words), 5):
                        chunk = ' '.join(words[i:i+5]) + ' '
                        yield json.dumps(chunk) + '\n'
                        time.sleep(0.1)
                    
                    # Update history
                    enhanced_response = chat_service.citation_service.enhance_response_with_citations(fallback_response, docs)
                    chat_service._update_conversation_history(conversation_id, message, enhanced_response)
                
            except Exception as e:
                logger.error(f"Error in streaming generator: {str(e)}")
                yield json.dumps(f"An error occurred: {str(e)}") + '\n'
        
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
        
    except Exception as e:
        logger.error(f"Error in stream_message: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@chat_bp.route('/api/v1/use_cases', methods=['POST'])
def get_use_cases():
    """Endpoint to generate use cases for a given domain."""
    try:
        data = request.json
        domain = data.get('domain', '')

        if not domain:
            return jsonify({"error": "Domain is required"}), 400

        use_cases = chat_service.generate_use_cases(domain)

        return jsonify({
            "use_cases": use_cases
        })

    except Exception as e:
        logger.error(f"Error in get_use_cases: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@chat_bp.route('/api/v1/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation endpoint."""
    try:
        data = request.json or {}
        conversation_id = data.get('conversationId', 'default')
        
        # Reset the conversation
        chat_service.reset_conversation(conversation_id)
        
        return jsonify({
            "success": True, 
            "message": "Conversation reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in reset_conversation: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500