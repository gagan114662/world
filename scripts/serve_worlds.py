#!/usr/bin/env python3
"""
Simple HTTP server for serving pre-generated worlds during development.

Usage:
    python serve_worlds.py --port 8010 --dir ./world_library

This serves the world library at http://localhost:8010/worlds/{world_id}/{segment}.mp4
"""

import os
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import functools


class CORSHandler(SimpleHTTPRequestHandler):
    """Handler with CORS support for development."""

    def __init__(self, *args, directory=None, **kwargs):
        self.base_directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path):
        """Translate URL path to filesystem path, handling /worlds prefix."""
        # Remove /worlds prefix if present
        if path.startswith('/worlds'):
            path = path[7:]  # Remove '/worlds'

        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser(description="Serve world library for development")
    parser.add_argument("--port", type=int, default=8010, help="Port to serve on")
    parser.add_argument("--dir", type=str, default="./world_library",
                        help="Directory containing world files")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")

    args = parser.parse_args()

    directory = Path(args.dir).absolute()

    if not directory.exists():
        print(f"Creating directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    # Create handler with directory
    handler = functools.partial(CORSHandler, directory=str(directory))

    server = HTTPServer((args.host, args.port), handler)

    print(f"\n{'='*60}")
    print(f"World Library Development Server")
    print(f"{'='*60}")
    print(f"\nServing: {directory}")
    print(f"URL: http://{args.host}:{args.port}/worlds/")
    print(f"\nExample URLs:")
    print(f"  http://{args.host}:{args.port}/worlds/solar_system/start.mp4")
    print(f"  http://{args.host}:{args.port}/worlds/solar_system/metadata.json")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
