"""
Document snippet routes for the AIRI chatbot API.
"""
from flask import Blueprint, jsonify

from ...config.logging import get_logger

logger = get_logger(__name__)

# Create blueprint
snippets_bp = Blueprint('snippets', __name__)

# This will be injected by the app factory
chat_service = None

def init_snippet_routes(chat_service_instance):
    """Initialize snippet routes with service dependency."""
    global chat_service
    chat_service = chat_service_instance

@snippets_bp.route('/api/snippet/<doc_id>', methods=['GET'])
def get_snippet(doc_id):
    """Retrieve a document snippet by its ID."""
    try:
        if not chat_service or not chat_service.citation_service:
            return jsonify({"error": "Citation service not available"}), 503
        
        content = chat_service.citation_service.get_snippet_content(doc_id)
        
        if content == "Snippet not found":
            return jsonify({"error": "Snippet not found"}), 404
        elif content.startswith("Error"):
            return jsonify({"error": content}), 500
        else:
            return jsonify({"content": content})
        
    except Exception as e:
        logger.error(f"Error retrieving snippet {doc_id}: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500