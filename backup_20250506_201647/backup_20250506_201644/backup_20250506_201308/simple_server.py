#!/usr/bin/env python
from flask import Flask, send_from_directory
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, 'github-frontend-build')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    print(f"Request for path: {path}")
    
    if path and os.path.exists(os.path.join(frontend_path, path)):
        return send_from_directory(frontend_path, path)
    else:
        return send_from_directory(frontend_path, 'index.html')

if __name__ == '__main__':
    print(f"Starting server at http://localhost:5000")
    print(f"Serving files from: {frontend_path}")
    app.run(debug=True, port=5000)