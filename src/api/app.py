"""
Flask application factory for the AIRI chatbot API.
"""
import os
import socket
from typing import Dict
from flask import Flask, send_from_directory
from flask_cors import CORS

from .routes.chat import chat_bp, init_chat_routes
from .routes.health import health_bp, init_health_routes  
from .routes.snippets import snippets_bp, init_snippet_routes
from ..core.services.chat_service import ChatService
from ..core.models.gemini import GeminiModel
from ..core.storage.vector_store import VectorStore
from ..config.logging import setup_logging, get_logger
from ..config.settings import settings

def create_app(config=None):
    """
    Application factory for creating Flask app instances.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Flask application instance
    """
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Create Flask app
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Initialize services
    chat_service = _initialize_services(logger)
    
    # Initialize route blueprints with dependencies
    init_chat_routes(chat_service)
    init_health_routes(chat_service)
    init_snippet_routes(chat_service)
    
    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(snippets_bp)
    
    # Add frontend routes
    _add_frontend_routes(app, logger)
    
    # Add error handlers
    _add_error_handlers(app, logger)
    
    logger.info("Flask application created successfully")
    return app

def _initialize_services(logger):
    """Initialize all services and components."""
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            embedding_provider=settings.EMBEDDING_PROVIDER,
            api_key=settings.GEMINI_API_KEY,
            repository_path=settings.get_repository_path(),
            persist_directory=str(settings.CHROMA_DB_DIR),
            use_hybrid_search=settings.USE_HYBRID_SEARCH
        )
        
        # Initialize vector store - unified approach
        logger.info("Starting vector store initialization...")
        try:
            success = vector_store.initialize()
            if not success:
                logger.error("Vector store initialization failed completely")
                vector_store = None
            else:
                logger.info("Vector store initialization successful")
        except Exception as e:
            logger.error(f"Vector store initialization threw exception: {str(e)}")
            vector_store = None
        
        # Initialize query monitor (optional)
        query_monitor = None
        try:
            from ..core.query.monitor import Monitor as QueryMonitor
            query_monitor = QueryMonitor(api_key=settings.GEMINI_API_KEY)
            logger.info("Query monitor initialized")
        except ImportError:
            logger.warning("Query monitor not available")
        except Exception as e:
            logger.warning(f"Query monitor initialization failed: {str(e)}")
        
        # Initialize Gemini model
        logger.info("Initializing Gemini model...")
        gemini_model = GeminiModel(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME
        )
        
        # Initialize chat service
        logger.info("Initializing chat service...")
        chat_service = ChatService(
            gemini_model=gemini_model,
            vector_store=vector_store,
            query_monitor=query_monitor
        )
        
        # Validate system readiness
        readiness_status = _validate_system_readiness(chat_service)
        logger.info(f"System readiness: {readiness_status}")
        
        logger.info("All services initialized successfully")
        return chat_service
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        # Return a minimal chat service even if initialization fails
        return ChatService()

def _validate_system_readiness(chat_service) -> Dict[str, str]:
    """Validate system component readiness and log status."""
    status = {
        "vector_store": "✗ Unavailable",
        "gemini_model": "✗ Unavailable", 
        "query_monitor": "✗ Unavailable",
        "overall": "Degraded"
    }
    
    # Check vector store
    if chat_service.vector_store:
        try:
            # Test with a simple query
            test_docs = chat_service.vector_store.get_relevant_documents("test", k=1)
            status["vector_store"] = f"✓ Ready ({len(test_docs)} test docs)"
        except Exception as e:
            status["vector_store"] = f"✗ Error: {str(e)[:50]}"
    
    # Check Gemini model
    if chat_service.gemini_model:
        try:
            # Test generation
            test_response = chat_service.gemini_model.generate("Test")
            status["gemini_model"] = "✓ Ready" if test_response else "✗ No response"
        except Exception as e:
            status["gemini_model"] = f"✗ Error: {str(e)[:50]}"
    
    # Check query monitor
    if chat_service.query_processor.query_monitor:
        try:
            test_analysis = chat_service.query_processor.query_monitor.determine_inquiry_type("test")
            status["query_monitor"] = "✓ Ready" if test_analysis else "✗ No response"
        except Exception as e:
            status["query_monitor"] = f"✗ Error: {str(e)[:50]}"
    
    # Determine overall status
    ready_count = sum(1 for s in status.values() if s.startswith("✓"))
    total_components = 3
    
    if ready_count == total_components:
        status["overall"] = "Fully Operational"
    elif ready_count >= 1:
        status["overall"] = f"Partially Operational ({ready_count}/{total_components})"
    else:
        status["overall"] = "Degraded Mode"
    
    return status

def _add_frontend_routes(app, logger):
    """Add routes for serving the frontend."""
    
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_frontend(path):
        """Serve the React app (GitHub frontend)."""
        logger.debug(f"Serving path: {path}")
        
        # Special route for source snippets
        if path.startswith('snippet/'):
            snippet_id = path.split('/')[-1]
            return _get_snippet_page(snippet_id)
        
        try:
            frontend_path = settings.FRONTEND_DIR
            
            if path and (frontend_path / path).exists():
                logger.debug(f"Serving file: {path}")
                return send_from_directory(str(frontend_path), path)
            elif path and '.' in path:  # File with extension that doesn't exist
                logger.warning(f"File not found: {path}")
                return f"File not found: {path}", 404
            else:
                # Try to serve index.html for all other routes (SPA routing)
                logger.debug(f"Serving index.html for path: {path}")
                index_path = frontend_path / 'index.html'
                if index_path.exists():
                    return send_from_directory(str(frontend_path), 'index.html')
                else:
                    return "Frontend not found. Please build the frontend first.", 404
        except Exception as e:
            logger.error(f"Error serving file: {str(e)}")
            return f"Error: {str(e)}", 500

def _get_snippet_page(snippet_id):
    """Generate an HTML page for viewing a document snippet."""
    # This could be enhanced to use the citation service
    snippet_path = settings.DOC_SNIPPETS_DIR / f"doc_{snippet_id}.txt"
    
    if snippet_path.exists():
        try:
            with open(snippet_path, 'r') as f:
                content = f.read()
                
            # Simple HTML formatting for the snippet content
            formatted_content = content.replace('\\n', '<br>')
            
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Source</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .snippet {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .metadata {{ font-weight: bold; color: #555; }}
        .content {{ margin-top: 20px; white-space: pre-wrap; }}
        .back-button {{ display: inline-block; margin-top: 20px; padding: 10px 15px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Document Source Reference</h1>
    <div class="snippet">
        <div class="content">{formatted_content}</div>
    </div>
    <a href="javascript:history.back()" class="back-button">Go Back</a>
</body>
</html>"""
            return html
        except Exception as e:
            return f"Error reading snippet: {str(e)}", 500
    else:
        return "Snippet not found", 404

def _add_error_handlers(app, logger):
    """Add error handlers to the app."""
    
    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 error: {error}")
        return jsonify({"error": "Not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        return jsonify({"error": "Internal server error"}), 500

def get_available_port(preferred_port=None):
    """Find an available port to run the server on."""
    preferred_port = preferred_port or settings.get_port()
    
    def is_port_available(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        available = False
        try:
            sock.bind(('0.0.0.0', port))
            available = True
        except:
            pass
        finally:
            sock.close()
        return available
    
    # Try preferred port first
    if is_port_available(preferred_port):
        return preferred_port
    
    # Try alternative ports
    for alt_port in settings.ALLOWED_PORTS:
        if is_port_available(alt_port):
            return alt_port
    
    # Return preferred port anyway and let it fail if needed
    return preferred_port