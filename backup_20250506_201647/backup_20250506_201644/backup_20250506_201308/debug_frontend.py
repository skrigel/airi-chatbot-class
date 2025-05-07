#!/usr/bin/env python
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, 'github-frontend-build')

print(f"Checking frontend directory: {frontend_path}")
print(f"Directory exists: {os.path.exists(frontend_path)}")

if not os.path.exists(frontend_path):
    print("ERROR: Frontend directory does not exist!")
    sys.exit(1)

# Check index.html
index_path = os.path.join(frontend_path, 'index.html')
print(f"Checking for index.html: {index_path}")
print(f"Index file exists: {os.path.exists(index_path)}")

if not os.path.exists(index_path):
    print("ERROR: index.html not found in the frontend directory!")
    sys.exit(1)

# List files in the frontend directory
print("\nFiles in the frontend directory:")
for root, dirs, files in os.walk(frontend_path):
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, frontend_path)
        print(f"- {rel_path}")

print("\nFrontend looks good! Try running: python adapter.py")
print("Then visit: http://localhost:8090")