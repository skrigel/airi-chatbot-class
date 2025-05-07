#!/bin/bash
set -e  # Exit on any error

echo "==============================================="
echo "🚀 Deploying the MIT AI Risk Repository Chatbot"
echo "==============================================="

# Check if the python environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "It's recommended to activate your virtual environment first."
    echo "Example: source venv/bin/activate"
    
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 1
    fi
fi

# Check for required files
echo "✅ Checking for required files..."
required_files=("vector_store.py" "monitor.py" "gemini_model.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: $file not found!"
        exit 1
    fi
done

# Check if GitHub frontend is built
frontend_dir="github-frontend-build"
if [ ! -d "$frontend_dir" ] || [ ! -f "$frontend_dir/index.html" ]; then
    echo "⚠️  GitHub frontend not built yet."
    echo "Running integration script first..."
    
    # Make integration script executable if it isn't already
    if [ ! -x "integration.sh" ]; then
        chmod +x integration.sh
    fi
    
    # Run the integration script
    ./integration.sh
fi

# Install required packages
echo "✅ Installing required packages..."
pip install flask flask-cors python-dotenv

# Check which adapter step to use
step=${1:-3}  # Default to step 3 (full integration)

if [ "$step" -eq 1 ]; then
    adapter_file="adapter_step1.py"
    echo "🚀 Deploying with vector store only (Step 1)..."
elif [ "$step" -eq 2 ]; then
    adapter_file="adapter_step2.py"
    echo "🚀 Deploying with vector store and monitor (Step 2)..."
else
    adapter_file="adapter_step3.py"
    echo "🚀 Deploying with full functionality (Step 3)..."
fi

# Check if adapter file exists
if [ ! -f "$adapter_file" ]; then
    echo "❌ ERROR: $adapter_file not found!"
    exit 1
fi

# Create a symbolic link to the selected adapter
echo "✅ Setting up adapter..."
if [ -f "adapter.py" ]; then
    rm adapter.py
fi
ln -s "$adapter_file" adapter.py
echo "✅ Linked $adapter_file to adapter.py"

# Test the adapter API
echo "✅ Running API tests..."
python test_adapter.py

echo "==============================================="
echo "✅ Deployment complete!"
echo "🌐 The application is available at http://localhost:8090"
echo "🚀 Run: python adapter.py"
echo "==============================================="