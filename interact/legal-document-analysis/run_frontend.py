#!/usr/bin/env python3
"""
Streamlit Frontend Runner
Run this script to start the Streamlit frontend
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("🎨 Starting Streamlit Frontend...")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the frontend")
    print("-" * 50)
    
    # Run streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_frontend.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ])
