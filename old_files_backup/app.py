#!/usr/bin/env python
"""
AIRI Chatbot Application Entry Point

This file serves as the entry point for the AIRI chatbot application.
It imports the Flask app from airi_adapter.py and starts the server.
"""

import os
from airi_adapter import app

if __name__ == '__main__':
    # Get port from environment variables or use default port 8090
    port = int(os.environ.get('PORT', 8090))
    
    # Function to check if port is available
    def is_port_available(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        available = False
        try:
            sock.bind(('0.0.0.0', port))
            available = True
        except:
            pass
        finally:
            sock.close()
        return available
    
    # Try to use specified port, fall back to alternatives if needed
    if not is_port_available(port):
        print(f"Warning: Port {port} is already in use")
        
        # Try alternative ports
        for alt_port in [8090, 8080, 8000, 3000]:
            if is_port_available(alt_port):
                print(f"Using alternative port: {alt_port}")
                port = alt_port
                break
        
        if not is_port_available(port):
            print(f"Warning: Port {port} is still unavailable. The server may fail to start.")
    
    print("\n=========================================")
    print(" AIRI CHATBOT")
    print("=========================================")
    print(f"Starting server at http://localhost:{port}")
    print("=========================================\n")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)