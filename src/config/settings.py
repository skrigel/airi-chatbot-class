"""
Configuration settings for the AIRI chatbot application.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration settings."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    FRONTEND_DIR = BASE_DIR / "frontend"
    
    # Data directories
    INFO_FILES_DIR = DATA_DIR / "info_files"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    DOC_SNIPPETS_DIR = DATA_DIR / "doc_snippets"
    SIMPLE_STORE_PATH = DATA_DIR / "simple_store.pkl"
    
    # API Configuration
    DEFAULT_PORT = 8090
    ALLOWED_PORTS = [8090, 8080, 8000, 3000]
    DEBUG = True
    
    # Model Configuration
    GEMINI_API_KEY: str = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "models/embedding-001"
    
    # Vector Store Configuration
    EMBEDDING_PROVIDER = "google"
    USE_HYBRID_SEARCH = os.environ.get('USE_HYBRID_SEARCH', 'true').lower() == 'true'
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    RISK_ENTRY_CHUNK_SIZE = 2000
    RISK_ENTRY_CHUNK_OVERLAP = 300
    
    # Query Configuration
    DEFAULT_DOCS_RETRIEVED = 5
    EMPLOYMENT_DOCS_RETRIEVED = 8
    MAX_EMPLOYMENT_DOCS = 6
    MAX_OTHER_DOCS = 2
    
    # Cache Configuration
    QUERY_CACHE_EXPIRY = 60 * 15  # 15 minutes
    
    # Conversation Configuration
    MAX_CONVERSATION_HISTORY = 5
    
    @classmethod
    def get_port(cls) -> int:
        """Get the port from environment or use default."""
        return int(os.environ.get('PORT', cls.DEFAULT_PORT))
    
    @classmethod
    def get_repository_path(cls) -> str:
        """Get the repository path."""
        return os.environ.get('REPOSITORY_PATH', str(cls.INFO_FILES_DIR))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.INFO_FILES_DIR,
            cls.DOC_SNIPPETS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()