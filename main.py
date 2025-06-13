#!/usr/bin/env python3
"""
Main entry point for the AIRI Chatbot application.

The application provides a clean, modular architecture with:
- Proper separation of concerns
- Configuration management
- Service dependency injection
- Error handling and logging
"""
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.api.app import create_app, get_available_port
from src.config.logging import setup_logging, get_logger
from src.config.settings import settings

def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Ensure data directories exist
    settings.ensure_directories()
    
    # Create the Flask application
    app = create_app()
    
    # Get an available port
    port = get_available_port()
    
    # Print startup information
    print("\\n" + "="*50)
    print(" AIRI CHATBOT - Modular Architecture")
    print("="*50)
    print(f"Starting server at http://localhost:{port}")
    print(f"Frontend directory: {settings.FRONTEND_DIR}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Repository path: {settings.get_repository_path()}")
    print("="*50 + "\\n")
    
    # Log component status
    logger.info("Application startup complete")
    logger.info(f"Server starting on port {port}")
    
    try:
        # Run the Flask application
        app.run(
            host='0.0.0.0',
            port=port,
            debug=settings.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()