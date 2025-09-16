#!/usr/bin/env python3
"""
Startup script for Streamlit frontend.
"""

import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Start Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        os.path.join("src", "streamlit_app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
