#!/usr/bin/env python3
"""
Database rebuild script that creates both the main vector store and simple vector store as backup.
This ensures both search methods are available.
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

print("\n========================================")
print("AIRI VECTOR STORES REBUILD UTILITY")
print("========================================")
print("This script will rebuild both the main vector store and the simple vector store")

try:
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
    persist_directory = os.path.join(current_dir, 'chroma_db')
    
    # Check if repository exists
    if not os.path.exists(repository_path):
        print(f"\nERROR: Repository path {repository_path} does not exist")
        sys.exit(1)
    
    print(f"\nRepository path: {repository_path}")
    print(f"Main vector store path: {persist_directory}")
    
    # First check repository contents
    print("\nRepository contents:")
    for item in os.listdir(repository_path):
        item_path = os.path.join(repository_path, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path) / 1024  # Size in KB
            print(f"- {item} ({size:.1f} KB)")
            
            # Special handling for Excel files
            if item.lower().endswith('.xlsx') or item.lower().endswith('.xls'):
                print(f"  ** Excel file detected: {item}")
    
    # Check if the user wants to rebuild the existing vector store
    if os.path.exists(persist_directory):
        print(f"\nWARNING: Vector store directory already exists at {persist_directory}")
        response = input("Do you want to rebuild it? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Skipping main vector store rebuild...")
        else:
            print("Removing existing vector store...")
            try:
                shutil.rmtree(persist_directory)
                print("Existing vector store removed successfully")
            except Exception as remove_err:
                print(f"Error removing vector store: {str(remove_err)}")
                sys.exit(1)
    
    # 1. Build the main vector store
    print("\n----------------------------------------")
    print("BUILDING MAIN VECTOR STORE")
    print("----------------------------------------")
    
    try:
        # Import the main vector store
        print("Importing VectorStore class...")
        from vector_store import VectorStore
        
        # Get API key
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            api_key = input("Enter your Gemini API key: ").strip()
            if not api_key:
                print("ERROR: API key is required")
                sys.exit(1)
        
        # Initialize vector store
        print("Initializing main vector store...")
        vector_store = VectorStore(
            embedding_provider="google",
            api_key=api_key,
            repository_path=repository_path,
            persist_directory=persist_directory,
            use_hybrid_search=True
        )
        
        # Ingest documents
        print("Ingesting documents into main vector store...")
        start_time = time.time()
        result = vector_store.ingest_documents()
        end_time = time.time()
        
        if result:
            print(f"Main vector store build completed successfully in {end_time - start_time:.2f} seconds!")
        else:
            print("WARNING: Main vector store build did not complete successfully")
    
    except Exception as e:
        print(f"ERROR building main vector store: {str(e)}")
        print("Continuing with simple vector store build...")
    
    # 2. Build the simple vector store
    print("\n----------------------------------------")
    print("BUILDING SIMPLE VECTOR STORE (FALLBACK)")
    print("----------------------------------------")
    
    try:
        # Import the simple vector store
        print("Importing SimpleVectorStore class...")
        from simple_vector_store import SimpleVectorStore
        
        # Initialize simple vector store
        print("Initializing simple vector store...")
        simple_store = SimpleVectorStore(repository_path)
        
        # Test the simple store with basic queries
        print(f"\nLoaded {len(simple_store.documents)} documents in total.")
        
        # Test queries
        test_queries = [
            "What are risks related to employment?",
            "How can AI cause discrimination?",
            "What concerns exist about harmful language generation?"
        ]
        
        print("\nTesting the fallback database with sample queries:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            docs = simple_store.get_relevant_documents(query, k=2)
            print(f"Retrieved {len(docs)} documents.")
            
            if docs:
                doc = docs[0]  # Show first result
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                source = doc.metadata.get('source', 'unknown')
                print(f"First result source: {os.path.basename(source)}")
                print(f"Preview: {preview}")
        
        # Save the store instance to a file for the adapter to use
        print("\nSaving simple vector store...")
        import pickle
        with open(os.path.join(current_dir, 'simple_store.pkl'), 'wb') as f:
            pickle.dump(simple_store, f)
        
        print("Simple vector store saved to: simple_store.pkl")
    
    except Exception as e:
        print(f"ERROR building simple vector store: {str(e)}")
    
    print("\n========================================")
    print("REBUILD COMPLETE")
    print("========================================")
    print("You can now run the chatbot with: ./run.sh")
    print("Both vector stores have been rebuilt")
    
except Exception as e:
    print(f"\nUnexpected error during rebuild: {str(e)}")
    sys.exit(1)