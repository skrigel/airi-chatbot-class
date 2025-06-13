#!/bin/bash

echo "========================================="
echo "   INSTALLING CRITICAL DEPENDENCIES"
echo "========================================="

# Identify the correct Python and pip commands
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
  PIP_CMD="python3 -m pip"
elif command -v python &> /dev/null; then
  PYTHON_CMD="python"
  PIP_CMD="python -m pip"
else
  echo "ERROR: Python not found. Please install Python 3."
  exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Using pip: $PIP_CMD"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
  PYTHON_CMD="python"
  PIP_CMD="python -m pip"
fi

# Ensure pip is up to date
echo "Updating pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Try multiple installation methods to ensure all dependencies are installed
echo "Installing all dependencies from requirements.txt (Method 1)..."
$PIP_CMD install -r requirements.txt || echo "Installing from requirements.txt failed, trying alternative methods..."

echo "Installing critical packages (Method 2 - direct install)..."
$PIP_CMD install flask flask-cors python-dotenv google-generativeai langchain langchain-community langchain-google-genai chromadb tiktoken pydantic==2.6.1 rank-bm25 pandas openpyxl

echo "Installing critical packages (Method 3 - user install)..."
$PIP_CMD install --user flask flask-cors python-dotenv google-generativeai langchain langchain-community langchain-google-genai

# Make sure the langchain-google-genai package is installed (this is a critical one that was failing)
echo "Making sure langchain-google-genai is installed..."
$PIP_CMD install --user langchain-google-genai==0.0.7

# Make sure essential pydantic and langchain packages are installed
echo "Installing additional critical packages..."
$PIP_CMD install --user pydantic==2.6.1 langchain==0.1.12 tiktoken==0.5.2

# Create a test script to verify critical packages are installed
echo "
import sys

missing_modules = []

# Check critical modules
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
        print(f'{module} is installed successfully!')
    except ImportError as e:
        print(f'{module} is MISSING: {e}')
        missing_modules.append(module)

if missing_modules:
    print(f'FAILED: Missing modules: {missing_modules}')
    sys.exit(1)
else:
    print('All critical modules verified successfully!')
    sys.exit(0)
" > test_deps.py

# Test if critical packages were installed
echo "Verifying critical package installation..."
if $PYTHON_CMD test_deps.py; then
    echo "All critical packages verified successfully."
    rm test_deps.py
else
    echo "Flask verification failed! Trying stronger methods..."
    
    # Try multiple alternate methods
    echo "Attempting direct installation with pip (Method 4)..."
    pip install flask flask-cors langchain-google-genai==0.0.7 google-generativeai langchain-community langchain==0.1.12 tiktoken==0.5.2 pydantic==2.6.1
    
    echo "Attempting user installation with pip (Method 5)..."
    pip install --user flask flask-cors langchain-google-genai==0.0.7 google-generativeai langchain-community
    
    echo "Attempting with pip3 (Method 6)..."
    pip3 install --user flask flask-cors langchain-google-genai==0.0.7 google-generativeai langchain-community
    
    echo "Installing essential modules with explicit versions (Method 7)..."
    pip3 install --user langchain-google-genai==0.0.7 langchain==0.1.12 pydantic==2.6.1 tiktoken==0.5.2
    
    echo "Last resort: Installing from requirements.txt with pip3 (Method 8)..."
    pip3 install --user -r requirements.txt
    
    # Final verification
    if $PYTHON_CMD test_deps.py; then
        echo "All critical modules installed with alternative method."
        rm test_deps.py
    else
        echo "FINAL ATTEMPT: Installing individual critical modules..."
        $PIP_CMD install --user flask==2.3.3
        $PIP_CMD install --user langchain-google-genai
        $PIP_CMD install --user google-generativeai
        $PIP_CMD install --user langchain
        $PIP_CMD install --user langchain-community
        
        if $PYTHON_CMD test_deps.py; then
            echo "Basic critical modules installation successful."
            rm test_deps.py
        else
            echo "ERROR: Could not install required modules."
            echo "Please try manually with: pip install -r requirements.txt"
            exit 1
        fi
    fi
fi

# Ensure run.sh is executable
chmod +x run.sh

echo "========================================="
echo "Critical dependencies installed."
echo "You can now run the chatbot with: ./run.sh"
echo "========================================="