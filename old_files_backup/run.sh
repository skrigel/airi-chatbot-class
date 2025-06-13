#!/bin/bash

PORT=${1:-8090}
export PORT=$PORT

# Find the Python command to use
if command -v python3 &> /dev/null; then
  PYTHON="python3"
elif command -v python &> /dev/null; then
  # Check if python is actually python3
  PYTHON_VERSION=$(python --version 2>&1)
  if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
    PYTHON="python"
  else
    echo "Error: Python 3 is required but not found. Please install Python 3.6 or newer."
    exit 1
  fi
else
  echo "Error: No Python installation found. Please install Python 3.6 or newer."
  exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
  PYTHON="python"  # Use the venv's python
fi

# Check if required modules are installed
MISSING_MODULES=false

# Create a script to check for all critical dependencies
cat > check_deps.py << 'EOF'
import sys

missing_modules = []
modules_to_check = [
    'flask',
    'langchain_google_genai',
    'google.generativeai',
    'langchain',
    'langchain_community'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f'✓ {module} is installed')
    except ImportError as e:
        print(f'✗ {module} is MISSING: {e}')
        missing_modules.append(module)

if missing_modules:
    print(f'\nMissing modules: {", ".join(missing_modules)}')
    sys.exit(1)
else:
    print('\nAll critical modules are installed')
    sys.exit(0)
EOF

# Check if dependencies are missing
if ! $PYTHON check_deps.py; then
  echo "Missing dependencies detected. Attempting to install now..."
  
  # Make the install_deps.sh script executable
  chmod +x install_deps.sh
  
  # Run the installation script
  ./install_deps.sh
  
  # Check again if critical modules are now installed
  if ! $PYTHON check_deps.py; then
    echo "ERROR: Critical dependencies installation failed."
    echo ""
    echo "Please try manually installing the specific packages:"
    echo "  pip install --user flask flask-cors langchain-google-genai==0.0.7 google-generativeai"
    echo "  or:"
    echo "  pip3 install --user flask flask-cors langchain-google-genai==0.0.7 google-generativeai"
    echo ""
    echo "Alternatively, install all requirements with:"
    echo "  pip install --user -r requirements.txt"
    echo ""
    exit 1
  fi
  
  echo "Dependencies installed successfully!"
fi

# Check if vector database exists, rebuild if not
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ] || [ ! -f "simple_store.pkl" ]; then
  echo "Vector database missing or incomplete. Rebuilding..."
  
  # Make rebuild script executable
  chmod +x rebuild_stores.py
  
  # Run our improved rebuild script that creates both vector stores
  $PYTHON rebuild_stores.py
  
  if [ $? -ne 0 ]; then
    echo "WARNING: Vector store rebuild failed, trying simple vector store only..."
    
    # Try to at least build the simple vector store as fallback
    $PYTHON simple_rebuild.py
    
    if [ $? -ne 0 ]; then
      echo "ERROR: Failed to build even the simple vector store."
      echo "The chatbot may still run but search functionality will be limited."
    else
      echo "Simple vector store built successfully as fallback."
    fi
  else
    echo "Vector stores rebuilt successfully!"
  fi
fi

# Clean up the check script
rm -f check_deps.py

echo "Starting AIRI Chatbot at http://localhost:$PORT..."
$PYTHON airi_adapter.py
