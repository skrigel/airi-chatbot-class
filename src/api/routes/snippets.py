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
    """Retrieve a document snippet by its ID (supports both legacy and RID formats)."""
    try:
        if not chat_service or not chat_service.citation_service:
            return jsonify({"error": "Citation service not available"}), 503
        
        # Check if this is a RID format (RID-#####)
        if doc_id.startswith('RID-') and len(doc_id) == 9:
            content = chat_service.citation_service.get_snippet_by_rid(doc_id)
        else:
            # Legacy format support
            content = chat_service.citation_service.get_snippet_content(doc_id)
        
        if content == "Snippet not found" or "not found" in content.lower():
            return jsonify({"error": "Snippet not found"}), 404
        elif content.startswith("Error"):
            return jsonify({"error": content}), 500
        else:
            return jsonify({
                "content": content,
                "snippet_id": doc_id,
                "type": "RID" if doc_id.startswith('RID-') else "legacy"
            })
        
    except Exception as e:
        logger.error(f"Error retrieving snippet {doc_id}: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@snippets_bp.route('/api/snippet/<rid>/raw', methods=['GET'])
def get_snippet_raw(rid):
    """Get raw snippet content for RID (plain text)."""
    try:
        if not chat_service or not chat_service.citation_service:
            return "Citation service not available", 503
            
        if not rid.startswith('RID-'):
            return "Invalid RID format", 400
            
        content = chat_service.citation_service.get_snippet_by_rid(rid)
        
        if "not found" in content.lower():
            return "Snippet not found", 404
        
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except Exception as e:
        logger.error(f"Error retrieving raw snippet {rid}: {str(e)}")
        return "Internal server error", 500