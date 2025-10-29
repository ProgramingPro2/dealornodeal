#!/usr/bin/env python3
"""
Quick start script for API server.

This script provides an easy way to run the API server
with sensible defaults.
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.api_server import run_server


def main():
    """Main entry point for server mode."""
    parser = argparse.ArgumentParser(description='Deal or No Deal Tracker - API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--config', default='config.yaml', help='Config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEAL OR NO DEAL TRACKER - API SERVER")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print("=" * 60)
    print()
    print("API Endpoints:")
    print(f"- Health check: http://{args.host}:{args.port}/health")
    print(f"- API docs: http://{args.host}:{args.port}/docs")
    print(f"- Stats: http://{args.host}:{args.port}/stats")
    print()
    print("Starting server...")
    print()
    
    try:
        run_server(args.host, args.port, args.config)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
