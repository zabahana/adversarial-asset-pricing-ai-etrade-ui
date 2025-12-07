#!/usr/bin/env python3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
PORT = int(os.environ.get("PORT", "8501"))
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, *args):
        pass
server = HTTPServer(("0.0.0.0", PORT), H)
server.serve_forever()
