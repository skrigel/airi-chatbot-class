#!/bin/bash

echo "========================================="
echo "   REBUILDING VECTOR DATABASE"
echo "========================================="
echo "This will clear and rebuild the vector database with fresh data."

# Find the Python command to use
if command -v python3 &> /dev/null; then
  PYTHON="python3"
elif command -v python &> /dev/null; then
  PYTHON="python"
else
  echo "Error: Python not found. Please install Python 3."
  exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
  PYTHON="python"
fi

# Check for required modules
echo "Checking for required modules..."
if ! $PYTHON -c "import langchain_google_genai" &> /dev/null || ! $PYTHON -c "import openpyxl" &> /dev/null; then
  echo "Required modules not found. Running dependency installation..."
  chmod +x install_deps.sh
  ./install_deps.sh
  
  # Make sure openpyxl is installed (explicitly needed for Excel files)
  echo "Making sure openpyxl is installed for Excel support..."
  $PYTHON -m pip install --user openpyxl
  
  # Verify openpyxl installation
  if ! $PYTHON -c "import openpyxl" &> /dev/null; then
    echo "WARNING: openpyxl still not installed. Excel files may not load correctly."
    echo "Please install manually with: pip install openpyxl"
  else
    echo "openpyxl installed successfully for Excel support."
  fi
fi

# Clear existing database
if [ -d "chroma_db" ]; then
  echo "Removing existing vector database..."
  rm -rf chroma_db
  echo "Existing database removed."
fi

# Create ingestion script
echo "Creating ingestion script..."
cat > rebuild_db.py << 'EOF'
import os
import sys
import time
print("Starting vector database rebuild...")

# Verify all required modules are available
try:
    import openpyxl
    print("✓ openpyxl is available (needed for Excel files)")
except ImportError:
    print("✗ ERROR: openpyxl is not installed! Excel files will not be processed correctly.")
    print("  Please install it with: pip install openpyxl")
    
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("✓ langchain_google_genai is available")
except ImportError:
    print("✗ ERROR: langchain_google_genai is not installed!")
    print("  Please install it with: pip install langchain-google-genai==0.0.7")

try:
    import pandas as pd
    print("✓ pandas is available (needed for data processing)")
except ImportError:
    print("✗ ERROR: pandas is not installed! Data processing will fail.")
    print("  Please install it with: pip install pandas")

try:
    from vector_store import VectorStore
    
    # Get API key from environment or use default
    api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')
    
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
    persist_directory = os.path.join(current_dir, 'chroma_db')
    
    print(f"\nInitializing vector store with repository path: {repository_path}")
    print(f"Database will be stored in: {persist_directory}")
    
    # Verify repository contents
    if os.path.exists(repository_path):
        print("\nRepository contents:")
        main_excel_file = None
        for item in os.listdir(repository_path):
            item_path = os.path.join(repository_path, item)
            size = os.path.getsize(item_path) / 1024  # Size in KB
            print(f"- {item} ({size:.1f} KB)")
            
            # Special handling for main AI Risk Repository Excel file
            if item.lower().endswith(".xlsx") and ("risk" in item.lower() or "repository" in item.lower()):
                main_excel_file = item
                print(f"  ** Found main AI Risk Repository Excel file: {item}")
        
        # If we found an Excel file, prioritize it
        if main_excel_file:
            print(f"\nPrioritizing Excel file: {main_excel_file}")
            # Create a special flag file to prioritize this Excel file
            with open(os.path.join(repository_path, ".repository_config"), "w") as f:
                f.write(f"excel_priority={main_excel_file}")
    
    # Initialize vector store
    vector_store = VectorStore(
        embedding_provider="google",
        api_key=api_key,
        repository_path=repository_path,
        persist_directory=persist_directory,
        use_hybrid_search=True
    )
    
    print("\nIngesting documents into vector store...")
    start_time = time.time()
    success = vector_store.ingest_documents()
    end_time = time.time()
    
    if success:
        print(f"\nDocuments ingested successfully in {end_time - start_time:.1f} seconds!")
        
        # Try a test query to verify the database works
        print("\nTesting database with a sample query...")
        try:
            docs = vector_store.get_relevant_documents("What are AI risks related to employment?", k=2)
            print(f"Retrieved {len(docs)} documents successfully.")
            
            if docs:
                print("\nSample document content:")
                for i, doc in enumerate(docs[:1]):  # Show just the first doc
                    print(f"\nDocument {i+1}:")
                    print("-" * 40)
                    print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    print("-" * 40)
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
        except Exception as query_err:
            print(f"Test query failed: {str(query_err)}")
        
        sys.exit(0)
    else:
        print("\nFailed to ingest documents.")
        sys.exit(1)
except Exception as e:
    print(f"\nError during database rebuild: {str(e)}")
    sys.exit(1)
EOF

# Run the ingestion script
echo "Rebuilding vector database... This may take a few minutes."
$PYTHON rebuild_db.py

# Check if rebuild was successful
if [ $? -eq 0 ]; then
  echo "Vector database rebuilt successfully!"
  
  # Remove the temporary script
  rm rebuild_db.py
  
  echo "========================================="
  echo "Your AI Risk Repository is now searchable!"
  echo "Run the chatbot with: ./run.sh"
  echo "========================================="
else
  echo "Error: Failed to rebuild the vector database."
  echo "Please check the error messages above."
  # Keep the script for debugging
  echo "The rebuild_db.py script has been kept for debugging."
  exit 1
fi