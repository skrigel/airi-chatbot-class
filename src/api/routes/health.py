"""
Health and status routes for the AIRI chatbot API.
"""
from flask import Blueprint, jsonify

from ...config.logging import get_logger

logger = get_logger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__)

# This will be injected by the app factory
chat_service = None

def init_health_routes(chat_service_instance):
    """Initialize health routes with service dependency."""
    global chat_service
    chat_service = chat_service_instance

@health_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        components_status = {
            "vector_store": "ok" if chat_service and chat_service.vector_store else "disabled",
            "query_monitor": "ok" if chat_service and chat_service.query_processor.query_monitor else "disabled",
            "gemini_model": "ok" if chat_service and chat_service.gemini_model else "disabled"
        }
        
        if "disabled" in components_status.values():
            overall_status = "degraded"
        else:
            overall_status = "ok"
        
        return jsonify({
            "status": overall_status,
            "components": components_status,
            "service": "AIRI Chatbot API"
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "service": "AIRI Chatbot API"
        }), 500

@health_bp.route('/api/status', methods=['GET'])
def status():
    """Detailed status endpoint."""
    try:
        status_info = {
            "service": "AIRI Chatbot API",
            "version": "2.0.0",
            "components": {}
        }
        
        if chat_service:
            # Vector store status
            if chat_service.vector_store:
                status_info["components"]["vector_store"] = {
                    "status": "active",
                    "type": "ChromaDB with Google Embeddings",
                    "hybrid_search": chat_service.vector_store.use_hybrid_search
                }
            else:
                status_info["components"]["vector_store"] = {"status": "disabled"}
            
            # Model status
            if chat_service.gemini_model:
                status_info["components"]["gemini_model"] = {
                    "status": "active",
                    "model": chat_service.gemini_model.model_name
                }
            else:
                status_info["components"]["gemini_model"] = {"status": "disabled"}
            
            # Query processor status
            status_info["components"]["query_processor"] = {
                "status": "active",
                "monitor": "enabled" if chat_service.query_processor.query_monitor else "disabled"
            }
            
            # Citation service status
            status_info["components"]["citation_service"] = {
                "status": "active",
                "snippets_available": True
            }
        else:
            status_info["components"]["chat_service"] = {"status": "not_initialized"}
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Error in status check: {str(e)}")
        return jsonify({
            "service": "AIRI Chatbot API",
            "status": "error",
            "error": str(e)
        }), 500