#!/usr/bin/env python3
"""
Simplified startup script that starts Streamlit directly.
Validates basic setup before starting.
"""
import os
import sys
import subprocess

def main():
    """Start Streamlit directly."""
    port = os.environ.get("PORT", "8501")
    
    print("=" * 50)
    print("Starting Streamlit Application")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Quick validation
    if not os.path.exists("streamlit_app.py"):
        print("ERROR: streamlit_app.py not found!")
        sys.exit(1)
    
    print("Starting Streamlit...")
    print()
    
    # Start Streamlit using subprocess (more reliable)
    try:
        import subprocess
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.allowRunOnSave", "false",
            "--browser.gatherUsageStats", "false",
            "--server.runOnSave", "false",
            "--server.fileWatcherType", "none",
            "--server.maxUploadSize", "200",
            "--server.maxMessageSize", "200",
            "--logger.level", "error",
        ]
        # Use exec to replace this process with Streamlit
        os.execvpe(sys.executable, cmd, os.environ)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

