#!/usr/bin/env python3
"""
Script to rebuild the vector database with improved processing.
"""
import sys
import shutil
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.storage.vector_store import VectorStore
from src.config.logging import setup_logging, get_logger
from src.config.settings import settings

def main():
    """Rebuild the vector database."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting vector database rebuild...")
    
    # Remove existing database
    if settings.CHROMA_DB_DIR.exists():
        logger.info(f"Removing existing database at {settings.CHROMA_DB_DIR}")
        shutil.rmtree(settings.CHROMA_DB_DIR)
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        embedding_provider=settings.EMBEDDING_PROVIDER,
        api_key=settings.GEMINI_API_KEY,
        repository_path=settings.get_repository_path(),
        persist_directory=str(settings.CHROMA_DB_DIR),
        use_hybrid_search=settings.USE_HYBRID_SEARCH
    )
    
    # Ingest documents
    logger.info("Starting document ingestion...")
    success = vector_store.ingest_documents()
    
    if success:
        logger.info("Database rebuild completed successfully!")
        
        # Test retrieval
        logger.info("Testing document retrieval...")
        test_query = "What does the AI Risk Repository say about job loss risk?"
        docs = vector_store.get_relevant_documents(test_query, k=5)
        logger.info(f"Test query retrieved {len(docs)} documents")
        
        for i, doc in enumerate(docs[:3]):
            domain = doc.metadata.get('domain', 'Unknown')
            file_type = doc.metadata.get('file_type', 'Unknown')
            logger.info(f"  Doc {i+1}: Domain={domain}, Type={file_type}")
        
        print("\\n✅ Database rebuild completed successfully!")
    else:
        logger.error("Database rebuild failed!")
        print("\\n❌ Database rebuild failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()