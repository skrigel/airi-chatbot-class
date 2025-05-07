#!/bin/bash
set -e  # Exit on any error

# Create directory for GitHub repository
WORK_DIR="$(pwd)"
GITHUB_DIR="${WORK_DIR}/github-repo"
FRONTEND_BUILD="${WORK_DIR}/github-frontend-build"

echo "========================================="
echo "Full Backend + GitHub Frontend Integration"
echo "========================================="

# Step 1: Clone the GitHub repository if not already cloned
echo "Step 1: Clone the GitHub repository"
if [ ! -d "${GITHUB_DIR}" ]; then
  git clone https://github.com/skrigel/airi-chatbot-class.git "${GITHUB_DIR}"
  echo "Repository cloned to ${GITHUB_DIR}"
else
  echo "Repository directory already exists at ${GITHUB_DIR}"
  
  # Optional update - uncomment to enable
  # echo "Updating repository..."
  # cd "${GITHUB_DIR}"
  # git pull
  # cd "${WORK_DIR}"
fi

# Step 2: Check if Node.js is installed
echo "Step 2: Check Node.js requirements"
if ! command -v node &> /dev/null; then
  echo "Error: Node.js is not installed. Please install Node.js to build the frontend."
  echo "You can download it from: https://nodejs.org/"
  exit 1
fi

if ! command -v npm &> /dev/null; then
  echo "Error: npm is not installed. Please install npm to build the frontend."
  exit 1
fi

# Step 3: Build the GitHub Frontend
echo "Step 3: Build the GitHub Frontend"
cd "${GITHUB_DIR}/chatbot"
if [ -f "package.json" ]; then
  echo "Installing npm dependencies..."
  npm install
  
  echo "Building frontend..."
  npm run build
  
  if [ $? -eq 0 ]; then
    echo "Frontend built successfully"
  else
    echo "ERROR: Frontend build failed"
    cd "${WORK_DIR}"
    exit 1
  fi
else
  echo "ERROR: package.json not found in ${GITHUB_DIR}/chatbot"
  cd "${WORK_DIR}"
  exit 1
fi

# Step 4: Create directory for the Frontend Build
echo "Step 4: Create directory for the Frontend Build"
cd "${WORK_DIR}"
mkdir -p "${FRONTEND_BUILD}"
echo "Copying frontend build files..."
cp -r "${GITHUB_DIR}/chatbot/dist/"* "${FRONTEND_BUILD}/"
echo "Frontend build copied to ${FRONTEND_BUILD}"

# Step 5: Install Python dependencies
echo "Step 5: Install Python dependencies"
cd "${WORK_DIR}"

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

echo "Installing/updating Python dependencies..."
$PIP_CMD install -r requirements.txt

# Step 6: Ensure required files exist
echo "Step 6: Checking for required files"
REQUIRED_FILES=("vector_store.py" "app.py" "monitor.py" "gemini_model.py" "adapter_step1_fixed.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    MISSING_FILES+=("$file")
  fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
  echo "ERROR: The following required files are missing:"
  for file in "${MISSING_FILES[@]}"; do
    echo "  - $file"
  done
  exit 1
fi

echo "All required files are present."

# Step 7: Make sure info_files directory exists
echo "Step 7: Ensure info_files directory exists"
mkdir -p info_files

# Check if info_files directory is empty
if [ -z "$(ls -A info_files)" ]; then
  echo "Warning: info_files directory is empty. The chatbot needs documents to work properly."
  
  # If sample data exists in the GitHub repo, copy it
  if [ -d "${GITHUB_DIR}/info_files" ] && [ ! -z "$(ls -A ${GITHUB_DIR}/info_files)" ]; then
    echo "Copying sample data from GitHub repository..."
    cp -r "${GITHUB_DIR}/info_files/"* "info_files/"
    echo "Sample data copied to info_files/"
  else
    echo "Please add documents to the info_files directory before running the chatbot."
  fi
fi

# Step 8: Set up the complete adapter
echo "Step 8: Creating final adapter.py"
cp adapter_step1_fixed.py adapter.py
echo "Created adapter.py from adapter_step1_fixed.py"

# Make scripts executable
chmod +x run_adapter.sh

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

echo "========================================="
echo "Integration complete!"
echo "To run the integrated chatbot, execute:"
echo "  ./run_adapter.sh"
echo ""
echo "The chatbot will be available at:"
echo "  http://localhost:$PORT"
echo "========================================="

# Optional: run the adapter immediately
read -p "Do you want to run the adapter now? (y/n) " RUN_NOW
if [[ "$RUN_NOW" == "y" || "$RUN_NOW" == "Y" ]]; then
  echo "Starting adapter on port $PORT..."
  $PYTHON_CMD adapter.py
fi