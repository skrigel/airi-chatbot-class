#!/usr/bin/env python3
"""
Simple script to install Flask and other required dependencies
"""
import subprocess
import sys
import os

print("=== Flask Installation Script ===")

def run_pip_command(cmd):
    print(f"Running: {cmd}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, text=True,
                                capture_output=True)
        print(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False

# Try different installation methods
methods = [
    "pip install flask flask-cors python-dotenv",
    "pip3 install flask flask-cors python-dotenv",
    "pip install --user flask flask-cors python-dotenv",
    "pip3 install --user flask flask-cors python-dotenv",
    "python -m pip install flask flask-cors python-dotenv",
    "python3 -m pip install flask flask-cors python-dotenv",
    "python -m pip install --user flask flask-cors python-dotenv",
    "python3 -m pip install --user flask flask-cors python-dotenv"
]

# Try each method until one succeeds
success = False
for cmd in methods:
    print(f"\nTrying: {cmd}")
    if run_pip_command(cmd):
        success = True
        print(f"Successfully installed with: {cmd}")
        break
    else:
        print(f"Failed with: {cmd}")

# Verify Flask is installed
print("\nVerifying Flask installation...")
try:
    import flask
    print(f"✅ Flask is installed (version: {flask.__version__})")
    success = True
except ImportError:
    print("❌ Flask is not installed or not accessible in the current Python environment.")
    success = False

if success:
    print("\n=== Flask installation successful! ===")
    print("You can now run the AIRI Chatbot with: ./run.sh")
else:
    print("\n=== Flask installation failed! ===")
    print("Please try manually installing Flask with one of these commands:")
    print("  pip install flask flask-cors")
    print("  pip install --user flask flask-cors")
    print("Or check your Python environment setup.")