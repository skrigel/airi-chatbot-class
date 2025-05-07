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

# Run the application
echo "Starting the AI Risk Repository Chatbot on port $PORT..."
echo "Access the application at: http://localhost:$PORT"

# If the port is not 5000, show warning about frontend compatibility
if [ "$PORT" != "5000" ]; then
    echo ""
    echo "====================== WARNING ======================"
    echo "Running on port $PORT instead of port 5000."
    echo "This may cause issues with the frontend."
    echo "Consider running on port 5000 for full compatibility."
    echo "====================================================="
    echo ""
fi

$PYTHON_CMD flexible_adapter.py