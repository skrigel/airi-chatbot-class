#!/bin/bash

echo "========================================="
echo "   SIMPLE VECTOR DATABASE REBUILD"
echo "========================================="
echo "This script uses a simple keyword-based search instead of embeddings"
echo "to avoid model name format errors."

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

# Make sure pandas is installed for Excel support
echo "Checking for pandas and Excel support..."
$PYTHON -m pip install --user pandas openpyxl

# Run the simple rebuild script
echo "Starting simple vector database rebuild..."
$PYTHON simple_rebuild.py

# Check if rebuild was successful
if [ $? -eq 0 ]; then
  echo "==========================================" 
  echo "Simple vector database built successfully!"
  echo "Run the chatbot with: ./run.sh"
  echo "=========================================="
else
  echo "Error: Failed to build the simple vector database."
  echo "Please check the error messages above."
  exit 1
fi