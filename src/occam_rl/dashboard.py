"""Local read-only dashboard; bind loopback and expose only intentional result data."""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .io import ROOT
from .reporting import snapshot


def serve(port: int = 8765) -> None:
    """Reuse static UI with a fresh metrics endpoint, without exposing filesystem roots."""
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/live.json":
                payload = json.dumps(snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            return super().do_GET()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "site")))
    print(f"Dashboard: http://127.0.0.1:{port}/?live=1", flush=True)
    server.serve_forever()
