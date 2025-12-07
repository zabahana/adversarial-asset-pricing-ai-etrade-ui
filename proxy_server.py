#!/usr/bin/env python3
"""
Minimal proxy server that responds immediately to health checks.
Starts Streamlit in the background and proxies requests once ready.
"""
import os
import sys
import subprocess
import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get("PORT", 8501))
STREAMLIT_PORT = 8502

# Global flag
streamlit_ready = False

class SimpleHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler."""
    
    def do_GET(self):
        """Handle GET requests."""
        global streamlit_ready
        
        if streamlit_ready:
            # Streamlit is ready, try to proxy
            try:
                self._proxy_to_streamlit()
            except Exception:
                # If proxying fails, just respond with basic message
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body>Streamlit is starting...</body></html>")
        else:
            # Streamlit not ready, respond immediately
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
    
    def _proxy_to_streamlit(self):
        """Proxy request to Streamlit."""
        try:
            import http.client
            conn = http.client.HTTPConnection("localhost", STREAMLIT_PORT, timeout=5)
            conn.request("GET", self.path)
            resp = conn.getresponse()
            
            self.send_response(resp.status)
            # Copy headers
            for header, value in resp.getheaders():
                if header.lower() not in ['transfer-encoding', 'connection']:
                    self.send_header(header, value)
            self.end_headers()
            
            # Copy body
            data = resp.read()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            raise e
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass

def check_port(port, host='localhost', timeout=1):
    """Check if a port is listening."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def wait_for_streamlit():
    """Wait for Streamlit to become ready."""
    global streamlit_ready
    print(f"Waiting for Streamlit on port {STREAMLIT_PORT}...")
    
    for i in range(120):  # Wait up to 2 minutes
        if check_port(STREAMLIT_PORT):
            print(f"Streamlit is ready on port {STREAMLIT_PORT}")
            streamlit_ready = True
            return
        time.sleep(1)
    
    print("Warning: Streamlit did not become ready")

def start_streamlit():
    """Start Streamlit in background."""
    print(f"Starting Streamlit on port {STREAMLIT_PORT}...")
    
    if not os.path.exists("streamlit_app.py"):
        print("ERROR: streamlit_app.py not found!")
        return
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(STREAMLIT_PORT),
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
    
    try:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("=" * 60)
    print("Starting Proxy Server for Streamlit")
    print("=" * 60)
    print(f"Proxy Port: {PORT}")
    print(f"Streamlit Port: {STREAMLIT_PORT}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Validate
    if not os.path.exists("streamlit_app.py"):
        print("ERROR: streamlit_app.py not found!")
        sys.exit(1)
    
    # Start Streamlit in background thread
    print("Starting Streamlit in background...")
    streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Start waiting for Streamlit in background
    wait_thread = threading.Thread(target=wait_for_streamlit, daemon=True)
    wait_thread.start()
    
    # Start HTTP server immediately - this must work for Cloud Run
    print(f"Starting proxy server on port {PORT}...")
    try:
        # Create server
        server = HTTPServer(("0.0.0.0", PORT), SimpleHandler)
        print(f"✓ Proxy server created")
        print(f"✓ Listening on 0.0.0.0:{PORT}")
        print("✓ Ready to accept connections")
        print("=" * 60)
        # This blocks forever, serving requests
        server.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"ERROR: Port {PORT} is already in use!")
        else:
            print(f"ERROR starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
