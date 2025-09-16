#!/usr/bin/env python3
"""
Simple script to run the Streamlit frontend.
"""

import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Starting Streamlit frontend...")
    print("Streamlit will be available at: http://localhost:8501")
    print("Make sure the FastAPI backend is running on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    
    # Start Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        os.path.join("src", "streamlit_app.py"),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("uv add --active streamlit")
except Exception as e:
    print(f"Error starting Streamlit: {e}")
