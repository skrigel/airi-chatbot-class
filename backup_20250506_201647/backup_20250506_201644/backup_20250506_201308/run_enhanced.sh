#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    # Use python3/pip3 if not in a virtual environment
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check if requirements are installed
echo "Installing/updating dependencies..."
$PIP_CMD install -r requirements.txt

# Get the port to use for the adapter
PORT=${1:-5000}  # Default to port 5000 if not specified
export PORT=$PORT

# Function to check if port is available
is_port_available() {
    # Check if the port is in use
    if command -v nc >/dev/null 2>&1; then
        nc -z localhost $1 >/dev/null 2>&1
        return $?
    elif command -v lsof >/dev/null 2>&1; then
        lsof -i:$1 >/dev/null 2>&1
        return $?
    fi
    # If we can't check, assume it's available
    return 1
}

# Check if the port is available
if is_port_available $PORT; then
    echo "Warning: Port $PORT is already in use."
    
    # Try some alternative ports
    for alt_port in 8090 8080 8000 3000; do
        if ! is_port_available $alt_port; then
            echo "Trying alternative port: $alt_port"
            PORT=$alt_port
            export PORT=$PORT
            break
        fi
    done
    
    if is_port_available $PORT; then
        echo "Could not find an available port. Please try specifying a different port manually."
        exit 1
    fi
fi

# Make sure the info_files directory exists
echo "Checking info_files directory..."
mkdir -p info_files

if [ -z "$(ls -A info_files)" ]; then
    echo "Warning: info_files directory is empty. The chatbot needs documents to work properly."
    echo "Please add some documents to the info_files directory for the chatbot to analyze."
fi

# Create snippets directory
echo "Creating directory for document snippets..."
mkdir -p doc_snippets

# Run the application
echo "=========================================="
echo "Starting the ENHANCED AI Risk Repository Chatbot"
echo "This version includes:"
echo "- Clickable document citations"
echo "- Source file viewers"
echo "- Excel-specific references with sheet/row info"
echo "=========================================="
echo "Access the application at: http://localhost:$PORT"

$PYTHON_CMD enhanced_adapter.py