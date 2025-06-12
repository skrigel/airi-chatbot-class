#!/usr/bin/env python3
"""
Setup script for the AIRI Chatbot application.
"""
import sys
import subprocess
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.config.logging import setup_logging, get_logger
from src.config.settings import settings

def main():
    """Set up the application."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Setting up AIRI Chatbot application...")
    
    # Ensure directories exist
    logger.info("Creating data directories...")
    settings.ensure_directories()
    
    # Install dependencies (if requirements.txt exists)
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if requirements_file.exists():
        logger.info("Installing Python dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                         check=True, capture_output=True, text=True)
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            print("❌ Failed to install dependencies")
            return False
    
    # Check if we need to rebuild the database
    if not settings.CHROMA_DB_DIR.exists():
        logger.info("Vector database not found. Run 'python scripts/rebuild_database.py' to create it.")
        print("⚠️  Vector database not found. Run 'python scripts/rebuild_database.py' to create it.")
    
    # Check for required environment variables
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI':
        logger.warning("Using default Gemini API key. Set GEMINI_API_KEY environment variable for production.")
        print("⚠️  Using default Gemini API key. Set GEMINI_API_KEY environment variable for production.")
    
    print("\\n✅ Setup completed successfully!")
    print("\\nNext steps:")
    print("1. Set your GEMINI_API_KEY environment variable (if not already set)")
    print("2. Run 'python scripts/rebuild_database.py' to create the vector database")
    print("3. Run 'python main.py' to start the application")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)