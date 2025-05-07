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

# Set default port
PORT=8090

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

# Try ports 8090, 8080, 8000, 3000 and 5000
for p in 8090 8080 8000 3000 5000; do
    if ! is_port_available $p; then
        PORT=$p
        break
    fi
done

# Set environment variable for the port
export PORT=$PORT

# Make sure the info_files directory exists
echo "Checking info_files directory..."
mkdir -p info_files

# Check which adapter to run
ADAPTER=${1:-"fixed"}

if [ "$ADAPTER" = "step1" ]; then
    echo "Starting adapter_step1.py on port $PORT..."
    $PYTHON_CMD adapter_step1.py
elif [ "$ADAPTER" = "step2" ]; then
    echo "Starting adapter_step2.py on port $PORT..."
    $PYTHON_CMD adapter_step2.py
elif [ "$ADAPTER" = "step3" ]; then
    echo "Starting adapter_step3.py on port $PORT..."
    $PYTHON_CMD adapter_step3.py
else
    echo "Starting fixed adapter on port $PORT..."
    $PYTHON_CMD adapter_step1_fixed.py
fi

echo "Access the application at: http://localhost:$PORT"