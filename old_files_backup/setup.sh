#!/bin/bash
set -e  # Exit on any error

echo "========================================="
echo "   AIRI CHATBOT SETUP"
echo "========================================="

WORK_DIR="$(pwd)"
GITHUB_REPO="https://github.com/skrigel/airi-chatbot-class.git"
FRONTEND_DIR="${WORK_DIR}/frontend"
GITHUB_TEMP_DIR="${WORK_DIR}/github-temp"

# Check Python environment
echo "Checking Python environment..."

# Find the Python command to use
if command -v python3 &> /dev/null; then
  PYTHON="python3"
  PIP="pip3"
elif command -v python &> /dev/null; then
  # Check if python is actually python3
  PYTHON_VERSION=$(python --version 2>&1)
  if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
    PYTHON="python"
    PIP="pip"
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
  PYTHON="python"
  PIP="pip"
fi

# Install required Python packages
echo "Installing Python requirements..."
echo "Using Python command: $PYTHON"
echo "Using pip command: $PIP"

# Make sure pip is available
if ! command -v $PIP &> /dev/null; then
  echo "Error: $PIP not found. Trying to install it..."
  $PYTHON -m ensurepip --upgrade || {
    echo "Failed to install pip. Please install pip manually."
    exit 1
  }
fi

# Verify that the dependencies get installed properly
echo "Installing dependencies from requirements.txt..."
$PIP install -r requirements.txt || {
  echo "Failed to install dependencies from requirements.txt"
  echo "Trying to install critical dependencies directly..."
  $PIP install flask flask-cors python-dotenv google-generativeai langchain langchain-community langchain-google-genai chromadb
}

# Verify that Flask was installed
if ! $PYTHON -c "import flask" &> /dev/null; then
  echo "ERROR: Flask installation failed. Please install Flask manually with:"
  echo "  pip install flask flask-cors"
  exit 1
else
  echo "Flask is successfully installed!"
fi

# Create directories
echo "Setting up directories..."
mkdir -p info_files
mkdir -p doc_snippets
mkdir -p chroma_db

# Check if info_files is empty and add sample file if needed
if [ -z "$(ls -A info_files 2>/dev/null)" ]; then
  echo "Creating sample info file..."
  cat > info_files/sample_ai_risks.txt << EOF
MIT AI Risk Repository Information

The repository contains information about various AI risks across different domains including:

1. Discrimination & Fairness
2. Privacy & Security
3. Misinformation & Truth
4. Malicious Use
5. Human-Computer Interaction
6. Socioeconomic Impacts
7. System Safety

Each risk includes detailed descriptions, categories, and potential impacts.
EOF
  echo "Created sample info file in info_files directory"
fi

# Set up frontend from GitHub repo
echo "Setting up frontend..."
mkdir -p "${GITHUB_TEMP_DIR}"

if command -v git &> /dev/null; then
  echo "Cloning GitHub repository..."
  git clone --depth 1 $GITHUB_REPO "${GITHUB_TEMP_DIR}" || {
    echo "Warning: Git clone failed. Will try to use cached version if available."
  }
else
  echo "Warning: Git not found. Will try to use cached version if available."
fi

# Check if we have a valid repository
if [ -d "${GITHUB_TEMP_DIR}/chatbot" ] && [ -f "${GITHUB_TEMP_DIR}/chatbot/package.json" ]; then
  echo "Building frontend..."
  cd "${GITHUB_TEMP_DIR}/chatbot"
  
  if command -v npm &> /dev/null; then
    npm install
    npm run build
    
    echo "Copying frontend files..."
    mkdir -p "${FRONTEND_DIR}"
    if [ -d "dist" ]; then
      cp -r dist/* "${FRONTEND_DIR}/"
      echo "Frontend files copied successfully"
    else
      echo "Error: Build directory not found!"
      # Create a minimal frontend as fallback
      mkdir -p "${FRONTEND_DIR}"
      cat > "${FRONTEND_DIR}/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
  <title>AIRI Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    #chatbox { height: 400px; border: 1px solid #ccc; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
    #input { width: 80%; padding: 8px; }
    button { padding: 8px 16px; background: #4285f4; color: white; border: none; }
  </style>
</head>
<body>
  <h1>AIRI Chatbot</h1>
  <div id="chatbox"></div>
  <input id="input" type="text" placeholder="Type your message here...">
  <button onclick="sendMessage()">Send</button>
  
  <script>
    const chatbox = document.getElementById('chatbox');
    const input = document.getElementById('input');
    
    function addMessage(text, isUser) {
      const msg = document.createElement('div');
      msg.style.textAlign = isUser ? 'right' : 'left';
      msg.style.margin = '10px 0';
      msg.style.color = isUser ? '#4285f4' : '#000';
      msg.textContent = text;
      chatbox.appendChild(msg);
      chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      addMessage(text, true);
      input.value = '';
      
      try {
        const response = await fetch('/api/v1/sendMessage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, conversationId: 'default' })
        });
        
        const data = await response.json();
        addMessage(data.response, false);
      } catch (err) {
        addMessage('Error connecting to server', false);
      }
    }
    
    input.addEventListener('keypress', event => {
      if (event.key === 'Enter') sendMessage();
    });
    
    // Welcome message
    addMessage('Welcome to the AIRI Chatbot! Ask me about AI risks.', false);
  </script>
</body>
</html>
EOF
      echo "Created fallback frontend"
    fi
  else
    echo "Error: npm not found. Creating minimal frontend..."
    # Create a minimal frontend
    mkdir -p "${FRONTEND_DIR}"
    cat > "${FRONTEND_DIR}/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
  <title>AIRI Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    #chatbox { height: 400px; border: 1px solid #ccc; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
    #input { width: 80%; padding: 8px; }
    button { padding: 8px 16px; background: #4285f4; color: white; border: none; }
  </style>
</head>
<body>
  <h1>AIRI Chatbot</h1>
  <div id="chatbox"></div>
  <input id="input" type="text" placeholder="Type your message here...">
  <button onclick="sendMessage()">Send</button>
  
  <script>
    const chatbox = document.getElementById('chatbox');
    const input = document.getElementById('input');
    
    function addMessage(text, isUser) {
      const msg = document.createElement('div');
      msg.style.textAlign = isUser ? 'right' : 'left';
      msg.style.margin = '10px 0';
      msg.style.color = isUser ? '#4285f4' : '#000';
      msg.textContent = text;
      chatbox.appendChild(msg);
      chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      addMessage(text, true);
      input.value = '';
      
      try {
        const response = await fetch('/api/v1/sendMessage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, conversationId: 'default' })
        });
        
        const data = await response.json();
        addMessage(data.response, false);
      } catch (err) {
        addMessage('Error connecting to server', false);
      }
    }
    
    input.addEventListener('keypress', event => {
      if (event.key === 'Enter') sendMessage();
    });
    
    // Welcome message
    addMessage('Welcome to the AIRI Chatbot! Ask me about AI risks.', false);
  </script>
</body>
</html>
EOF
    echo "Created minimal frontend"
  fi
else
  echo "Error: Frontend source files not found or invalid GitHub repository!"
  # Create a minimal frontend
  mkdir -p "${FRONTEND_DIR}"
  cat > "${FRONTEND_DIR}/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
  <title>AIRI Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    #chatbox { height: 400px; border: 1px solid #ccc; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
    #input { width: 80%; padding: 8px; }
    button { padding: 8px 16px; background: #4285f4; color: white; border: none; }
  </style>
</head>
<body>
  <h1>AIRI Chatbot</h1>
  <div id="chatbox"></div>
  <input id="input" type="text" placeholder="Type your message here...">
  <button onclick="sendMessage()">Send</button>
  
  <script>
    const chatbox = document.getElementById('chatbox');
    const input = document.getElementById('input');
    
    function addMessage(text, isUser) {
      const msg = document.createElement('div');
      msg.style.textAlign = isUser ? 'right' : 'left';
      msg.style.margin = '10px 0';
      msg.style.color = isUser ? '#4285f4' : '#000';
      msg.textContent = text;
      chatbox.appendChild(msg);
      chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      addMessage(text, true);
      input.value = '';
      
      try {
        const response = await fetch('/api/v1/sendMessage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, conversationId: 'default' })
        });
        
        const data = await response.json();
        addMessage(data.response, false);
      } catch (err) {
        addMessage('Error connecting to server', false);
      }
    }
    
    input.addEventListener('keypress', event => {
      if (event.key === 'Enter') sendMessage();
    });
    
    // Welcome message
    addMessage('Welcome to the AIRI Chatbot! Ask me about AI risks.', false);
  </script>
</body>
</html>
EOF
  echo "Created minimal frontend"
fi

cd "${WORK_DIR}"

# Create frontend patch script to fix API URL
echo "Creating frontend patch..."
cat > "${FRONTEND_DIR}/api-patch.js" << 'EOF'
// API URL patch script
(function() {
  console.log("API URL patch applied");
  
  // Override fetch to redirect API calls
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {
    if (typeof url === 'string' && url.includes('localhost:5000')) {
      const newUrl = url.replace('http://localhost:5000/', window.location.origin + '/');
      console.log('Redirecting API call from', url, 'to', newUrl);
      url = newUrl;
    }
    return originalFetch.call(this, url, options);
  };
})();
EOF

# Add patch script to index.html
if [ -f "${FRONTEND_DIR}/index.html" ]; then
  if ! grep -q "api-patch.js" "${FRONTEND_DIR}/index.html"; then
    sed -i.bak 's/<\/head>/<script src="\/api-patch.js"><\/script>\n  <\/head>/' "${FRONTEND_DIR}/index.html"
    rm -f "${FRONTEND_DIR}/index.html.bak"
    echo "Patched index.html to fix API URL"
  else
    echo "index.html already patched"
  fi
fi

# Create simple run script if it doesn't exist
if [ ! -f "run.sh" ] || [ ! -x "run.sh" ]; then
  echo "Creating run script..."
  cat > run.sh << 'EOF'
#!/bin/bash

PORT=${1:-5000}
export PORT=$PORT

echo "Starting AIRI Chatbot at http://localhost:$PORT..."
python3 airi_adapter.py
EOF

  chmod +x run.sh
  echo "Created run.sh script"
fi

# Clean up temp files
echo "Cleaning up temporary files..."
if [ -d "${GITHUB_TEMP_DIR}" ]; then
  rm -rf "${GITHUB_TEMP_DIR}"
fi

echo "========================================="
echo "Setup complete!"
echo "You can now run the AIRI Chatbot with:"
echo "  ./run.sh"
echo ""
echo "The chatbot will be available at:"
echo "  http://localhost:5000"
echo "========================================="

# Run a data ingestion step to properly build the vector database
if [ -d "info_files" ] && [ "$(ls -A info_files 2>/dev/null)" ]; then
  echo "========================================="
  echo "   BUILDING VECTOR DATABASE"
  echo "========================================="
  echo "This will index the AI Risk Repository data for search..."
  
  # Make the rebuild scripts executable
  chmod +x rebuild_database.sh run_simple_rebuild.sh
  
  # First try the standard rebuild (with updated embedding model name)
  echo "Running standard vector database build..."
  ./rebuild_database.sh
  
  # Check if it succeeded
  if [ $? -ne 0 ]; then
    echo "Standard vector database build failed."
    echo "Trying alternative keyword-based build..."
    
    # Try the simple rebuild as fallback
    ./run_simple_rebuild.sh
    
    if [ $? -ne 0 ]; then
      echo "WARNING: Both database builds failed."
      echo "You may need to run './rebuild_database.sh' or './run_simple_rebuild.sh' manually after setup."
    else
      echo "Alternative keyword-based database built successfully!"
    fi
  else
    echo "Vector database built successfully!"
  fi
  
  echo "Vector database setup completed."
fi

# Offer to start the chatbot web interface immediately
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
read -p "Do you want to start the chatbot web interface now? (y/n) " RUN_NOW
if [[ "$RUN_NOW" == "y" || "$RUN_NOW" == "Y" ]]; then
  echo "Starting AIRI Chatbot web interface..."
  ./run.sh
else
  echo "You can start the chatbot later with: ./run.sh"
fi