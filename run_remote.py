#!/usr/bin/env python3
"""
Quick start script for remote mode.

This script provides an easy way to run the tracker in remote mode
with sensible defaults for Raspberry Pi.
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.tracker_client import TrackerClient


def main():
    """Main entry point for remote mode."""
    parser = argparse.ArgumentParser(description='Deal or No Deal Tracker - Remote Mode')
    parser.add_argument('--server', default='http://localhost:8000', 
                       help='Server URL (default: http://localhost:8000)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--config', default='config.yaml', help='Config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEAL OR NO DEAL CASE TRACKER - REMOTE MODE")
    print("=" * 60)
    print(f"Server: {args.server}")
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Config: {args.config}")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Make sure the API server is running")
    print("2. Position camera to view the arcade screen")
    print("3. Wait for game to start (looks for 200/100 values)")
    print("4. Tracker will send frames to server for processing")
    print("5. Game ends when it sees case numbers 1 and 16")
    print("6. Video will be uploaded for verification")
    print("7. Final ranking will be displayed")
    print()
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset")
    print("- Press 's' to save screenshot")
    print()
    print("Starting tracker...")
    print()
    
    try:
        # Create client with custom settings
        client = TrackerClient(args.config)
        
        # Override settings from command line
        if hasattr(client, 'config'):
            client.config['server']['url'] = args.server
            client.config['camera']['device_id'] = args.camera
            client.config['camera']['width'] = args.width
            client.config['camera']['height'] = args.height
            client.config['camera']['fps'] = args.fps
        
        # Run in remote mode
        client.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
