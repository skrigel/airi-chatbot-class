#!/bin/bash
set -e  # Exit on any error

# Create directory for GitHub repository
WORK_DIR="$(pwd)"
GITHUB_DIR="${WORK_DIR}/github-repo"
FRONTEND_BUILD="${WORK_DIR}/github-frontend-build"

echo "========================================="
echo "Simple Backend + GitHub Frontend Integration"
echo "========================================="

# Step 1: Clone the GitHub repository if not already cloned
echo "Step 1: Clone the GitHub repository"
if [ ! -d "${GITHUB_DIR}" ]; then
  git clone https://github.com/skrigel/airi-chatbot-class.git "${GITHUB_DIR}"
  echo "Repository cloned to ${GITHUB_DIR}"
else
  echo "Repository directory already exists at ${GITHUB_DIR}"
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

# Step 6: Make sure info_files directory exists
echo "Step 6: Ensure info_files directory exists"
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

# Make the run_flexible.sh executable
chmod +x run_flexible.sh

echo "========================================="
echo "Integration complete!"
echo "To run the integrated chatbot on port 5000 (which is what the frontend expects):"
echo "  ./run_flexible.sh"
echo ""
echo "The chatbot will be available at:"
echo "  http://localhost:5000"
echo "========================================="

# Optional: run the adapter immediately
read -p "Do you want to run the adapter now? (y/n) " RUN_NOW
if [[ "$RUN_NOW" == "y" || "$RUN_NOW" == "Y" ]]; then
  echo "Starting adapter on port 5000..."
  $PYTHON_CMD flexible_adapter.py
fi