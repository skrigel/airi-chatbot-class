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
    GEMINI_API_KEY: str = os.environ.get('GEMINI_API_KEY', 'AIzaSyAVrH9JPSqSivrbfUWMS3XZA52zZ5DEkhk')
    
    # Multi-model fallback configuration
    GEMINI_MODEL_CHAIN = [
        "gemini-2.0-flash",
        "gemini-2.5-flash", 
        "gemini-2.5-flash-lite-06-17"
    ]
    
    # Primary model (for backward compatibility)
    GEMINI_MODEL_NAME = GEMINI_MODEL_CHAIN[0]
    
    # Model retry configuration
    MODEL_RETRY_DELAY = 5.0  # seconds between retries
    MODEL_COOLDOWN_TIME = 300  # 5 minutes cooldown for failed models
    MAX_RETRIES_PER_MODEL = 2
    
    # Model-specific settings
    MODEL_SETTINGS = {
        "gemini-2.0-flash": {
            "supports_thinking": True,
            "max_tokens": 8192,
            "temperature": 0.1
        },
        "gemini-2.5-flash": {
            "supports_thinking": True,
            "max_tokens": 8192,
            "temperature": 0.1
        },
        "gemini-2.5-flash-lite-06-17": {
            "supports_thinking": False,
            "max_tokens": 4096,
            "temperature": 0.1
        }
    }
    
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"
    
    # Vector Store Configuration
    EMBEDDING_PROVIDER = "google"
    USE_HYBRID_SEARCH = os.environ.get('USE_HYBRID_SEARCH', 'true').lower() == 'true'
    USE_FIELD_AWARE_HYBRID = os.environ.get('USE_FIELD_AWARE_HYBRID', 'true').lower() == 'true'
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    RISK_ENTRY_CHUNK_SIZE = 2000
    RISK_ENTRY_CHUNK_OVERLAP = 300
    
    # Query Configuration
    DEFAULT_DOCS_RETRIEVED = 5
    # Generic domain search configuration (replaces employment-specific)
    DOMAIN_DOCS_RETRIEVED = 8  # Default docs retrieved for domain searches
    MAX_DOMAIN_DOCS = 6       # Max docs to include from domain searches
    MAX_OTHER_DOCS = 2
    
    # Relevance Threshold Configuration (Phase 1.2)
    MINIMUM_RELEVANCE_THRESHOLD = 0.25  # Cosine similarity floor
    DOMAIN_RELEVANCE_THRESHOLDS = {
        'socioeconomic': 0.20,  # Lower threshold for employment queries
        'safety': 0.30,         # Higher threshold for safety (more specific)
        'privacy': 0.28,        # Medium threshold for privacy
        'bias': 0.25,           # Standard threshold for bias
        'governance': 0.32,     # Higher threshold for governance (specific)
        'technical': 0.35,      # Highest threshold for technical (very specific)
        'general': 0.25         # Standard threshold for general queries
    }
    MAX_DOCS_ABOVE_THRESHOLD = 3  # Maximum docs to return if above threshold
    
    # Cache Configuration
    QUERY_CACHE_EXPIRY = 60 * 15  # 15 minutes
    
    # Conversation Configuration
    MAX_CONVERSATION_HISTORY = 5
    
    # Monitor Configuration
    MONITOR_MODEL_NAME = "gemini-2.0-flash"
    MONITOR_TIMEOUT = 30  # seconds
    MONITOR_MAX_RETRIES = 3
    MONITOR_ENABLE_RULE_BASED = True
    MONITOR_ENABLE_MODEL_BASED = True
    
    # Vector Store Configuration (additional)
    VECTOR_WEIGHT = 0.7  # Weight for vector search in hybrid retrieval
    KEYWORD_WEIGHT = 0.3  # Weight for keyword search in hybrid retrieval
    HYBRID_RERANK_TOP_K = 10  # Top K results for hybrid reranking
    BM25_TOP_K = 10  # Top K results for BM25 retriever
    
    # Field-Aware Search Configuration
    FIELD_AWARE_SEARCH_ENABLED = True  # Enable field-aware metadata boosting
    HIGH_PRIORITY_FIELD_BOOST = 3.0    # Boost factor for high-priority fields (title, domain, category)
    MEDIUM_PRIORITY_FIELD_BOOST = 2.0  # Boost factor for medium-priority fields (subdomain, specific_domain)
    METADATA_BOOST_FACTOR = 1.5        # General boost for metadata field matches
    AI_RISK_ENTRY_BOOST = 0.2          # Boost for primary AI risk entries
    DOMAIN_SUMMARY_BOOST = 0.15        # Boost for domain summary documents
    DOMAIN_SPECIFIC_BOOST = 0.1        # Boost for documents with specific domain metadata
    
    # Context Formatting
    CONTEXT_TEMPLATE_HEADER = "INFORMATION FROM THE AI RISK REPOSITORY:\\n\\n"
    CONTEXT_SECTION_TEMPLATE = "SECTION {section_number}"
    CONTEXT_DOMAIN_TEMPLATE = " (Domain: {domain})"
    CONTEXT_SECTION_SEPARATOR = ":\\n{content}\\n\\n"
    
    # Generic Domain Configuration (replaces employment-specific)
    DOMAIN_ENHANCED_SEARCH_ENABLED = True  # Enable enhanced search for all domains
    DEFAULT_DOMAIN_DOCS_LIMIT = 2          # Default document limit per domain
    
    # Backward compatibility (deprecated - use domain system instead)
    EMPLOYMENT_KEYWORDS = ['employ', 'job', 'work', 'labor']  # DEPRECATED
    EMPLOYMENT_SEARCH_ENABLED = True                          # DEPRECATED  
    EMPLOYMENT_DOCS_LIMIT = 2                                # DEPRECATED
    
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