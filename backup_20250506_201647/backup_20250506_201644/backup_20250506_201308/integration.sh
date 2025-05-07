#!/bin/bash
set -e  # Exit on any error

# Create directory for GitHub repository
WORK_DIR="$(pwd)"
GITHUB_DIR="${WORK_DIR}/github-repo"
FRONTEND_BUILD="${WORK_DIR}/github-frontend-build"

echo "Step 1: Clone the GitHub repository"
if [ ! -d "${GITHUB_DIR}" ]; then
  git clone https://github.com/skrigel/airi-chatbot-class.git "${GITHUB_DIR}"
  echo "Repository cloned to ${GITHUB_DIR}"
else
  echo "Repository directory already exists, skipping clone"
fi

echo "Step 2: Build the GitHub Frontend"
cd "${GITHUB_DIR}/chatbot"
if [ -f "package.json" ]; then
  npm install
  npm run build
  echo "Frontend built successfully"
else
  echo "ERROR: package.json not found in ${GITHUB_DIR}/chatbot"
  exit 1
fi

echo "Step 3: Create directory for the Frontend Build"
mkdir -p "${FRONTEND_BUILD}"
cp -r "${GITHUB_DIR}/chatbot/dist/"* "${FRONTEND_BUILD}/"
echo "Frontend build copied to ${FRONTEND_BUILD}"

echo "Step 4: Copy your Enhanced Backend Files"
# These files should already be in your project root, so we don't need to copy them

echo "Step 5: Install Requirements"
cd "${WORK_DIR}"
pip install -r requirements.txt

echo "Step 6: Ready to run the Adapter Server"
echo "You can now run the adapter server with: python adapter.py"
echo "Then access the application at http://localhost:8090"

echo "Integration setup complete!"