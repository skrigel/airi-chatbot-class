#!/bin/bash

echo "========================================="
echo "   SETUP AND RUN AIRI CHATBOT"
echo "========================================="

# Make scripts executable
chmod +x install_deps.sh
chmod +x run.sh

# Run the installation script
echo "Running dependency installation script..."
./install_deps.sh

# Check if installation was successful
if [ $? -eq 0 ]; then
  echo "Dependencies installed successfully!"
  
  # Run the chatbot
  echo "Starting the AIRI chatbot..."
  ./run.sh
else
  echo "Error: Dependency installation failed!"
  exit 1
fi