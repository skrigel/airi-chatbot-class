#!/usr/bin/env python3
"""
Simple database rebuild script using keyword-based search instead of embeddings.
This avoids the embedding model errors.
"""

import os
import sys
import time
from pathlib import Path

print("Starting simple vector database rebuild...")

try:
    from simple_vector_store import SimpleVectorStore
    
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
    
    print(f"\nInitializing simple vector store with repository path: {repository_path}")
    
    # Verify repository contents
    if os.path.exists(repository_path):
        print("\nRepository contents:")
        for item in os.listdir(repository_path):
            item_path = os.path.join(repository_path, item)
            size = os.path.getsize(item_path) / 1024  # Size in KB
            print(f"- {item} ({size:.1f} KB)")
            
            # Special handling for Excel files
            if item.lower().endswith('.xlsx') or item.lower().endswith('.xls'):
                print(f"  ** Excel file detected: {item}")
    
    # Initialize simple vector store
    print("\nInitalizing simple vector store (keyword-based)...")
    store = SimpleVectorStore(repository_path)
    
    # Save the store data
    output_file = os.path.join(current_dir, 'simple_store_data.json')
    
    print(f"\nLoaded {len(store.documents)} documents in total.")
    
    # Test the store with a few sample queries
    test_queries = [
        "What are risks related to employment?",
        "How can AI cause discrimination?",
        "What concerns exist about harmful language generation?"
    ]
    
    print("\nTesting the database with sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        docs = store.get_relevant_documents(query, k=2)
        print(f"Retrieved {len(docs)} documents.")
        
        if docs:
            doc = docs[0]  # Show first result
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            source = doc.metadata.get('source', 'unknown')
            print(f"First result source: {os.path.basename(source)}")
            print(f"Preview: {preview}")
    
    print("\nSimple vector database build completed successfully.")
    
    # Save the store instance to a file for the adapter to use
    import pickle
    with open(os.path.join(current_dir, 'simple_store.pkl'), 'wb') as f:
        pickle.dump(store, f)
    
    print(f"Simple vector store saved to: simple_store.pkl")
    print("\nYou can now run the chatbot with: ./run.sh")
    
except Exception as e:
    print(f"\nError during database rebuild: {str(e)}")
    sys.exit(1)