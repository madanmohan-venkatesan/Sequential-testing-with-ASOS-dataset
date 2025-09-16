#!/usr/bin/env python3
"""
Simple script to run the FastAPI server.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import uvicorn
    from api import app
    
    print("Starting FastAPI server...")
    print("API will be available at: http://127.0.0.1:8000")
    print("API documentation: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("pip install fastapi uvicorn")
except Exception as e:
    print(f"Error starting server: {e}")
