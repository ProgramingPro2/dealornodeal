"""
Main Tracker Client Application

Handles the main client application with webcam capture, visualization,
and mode selection (local/remote processing).
"""

import cv2
import numpy as np
import yaml
import time
import threading
import requests
import base64
import json
from typing import Dict, Any, Optional
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import CaseTracker
from .webcam_capture import WebcamCapture
from .visualizer import Visualizer


class TrackerClient:
    """Main tracker client application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize tracker client.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.tracker = None
        self.webcam = None
        self.visualizer = None
        self.is_running = False
        self.mode = "local"  # "local" or "remote"
        self.session_id = None
        self.server_url = self.config.get('server', {}).get('url', 'http://localhost:8000')
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Video recording for remote mode
        self.video_writer = None
        self.recorded_frames = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'camera': {'width': 640, 'height': 480, 'fps': 30, 'device_id': 0},
            'processing': {'resize_width': 640, 'resize_height': 480, 'skip_frames': 2},
            'visualization': {'show_tracking': True, 'show_confidence': True, 'highlight_best': True},
            'server': {'url': 'http://localhost:8000'},
            'values': {'expected_values': [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400]}
        }
    
    def initialize(self, mode: str = "local") -> bool:
        """
        Initialize the tracker client.
        
        Args:
            mode: Processing mode ("local" or "remote")
            
        Returns:
            True if successful, False otherwise
        """
        self.mode = mode
        
        # Initialize webcam
        camera_config = self.config.get('camera', {})
        self.webcam = WebcamCapture(
            device_id=camera_config.get('device_id', 0),
            width=camera_config.get('width', 640),
            height=camera_config.get('height', 480),
            fps=camera_config.get('fps', 30)
        )
        
        if not self.webcam.initialize():
            print("Failed to initialize camera")
            return False
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.config)
        
        # Initialize tracker based on mode
        if mode == "local":
            self.tracker = CaseTracker(self.config)
        elif mode == "remote":
            if not self._test_server_connection():
                print("Failed to connect to server")
                return False
        else:
            print(f"Invalid mode: {mode}")
            return False
        
        print(f"Tracker client initialized in {mode} mode")
        return True
    
    def _test_server_connection(self) -> bool:
        """Test connection to remote server."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _init_remote_session(self) -> bool:
        """Initialize remote tracking session."""
        try:
            response = requests.post(f"{self.server_url}/api/track/init", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('session_id')
                return True
        except Exception as e:
            print(f"Failed to initialize remote session: {e}")
        return False
    
    def _process_frame_local(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame locally."""
        if self.tracker is None:
            return {}
        
        start_time = time.time()
        results = self.tracker.process_frame(frame)
        processing_time = (time.time() - start_time) * 1000
        
        # Add debug info
        results['processing_time'] = processing_time
        results['case_count'] = len(results.get('cases', {}))
        
        return results
    
    def _process_frame_remote(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame remotely."""
        if self.session_id is None:
            return {}
        
        try:
            # Encode frame as base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to server
            payload = {
                'session_id': self.session_id,
                'frame_base64': frame_base64,
                'timestamp': time.time()
            }
            
            response = requests.post(f"{self.server_url}/api/track/frame", 
                                   json=payload, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server error: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Remote processing error: {e}")
            return {}
    
    def _start_video_recording(self, frame: np.ndarray):
        """Start video recording for remote mode."""
        if self.video_writer is not None:
            return
        
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f"tracking_session_{int(time.time())}.mp4"
        self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        self.recorded_frames = []
    
    def _stop_video_recording(self):
        """Stop video recording and upload to server."""
        if self.video_writer is None:
            return
        
        self.video_writer.release()
        self.video_writer = None
        
        # Upload video to server
        if self.session_id:
            self._upload_video_to_server()
    
    def _upload_video_to_server(self):
        """Upload recorded video to server for verification."""
        try:
            # Find the most recent video file
            video_files = [f for f in os.listdir('.') if f.startswith('tracking_session_') and f.endswith('.mp4')]
            if not video_files:
                return
            
            latest_video = max(video_files, key=os.path.getctime)
            
            with open(latest_video, 'rb') as f:
                files = {'video_file': f}
                data = {'session_id': self.session_id}
                
                response = requests.post(f"{self.server_url}/api/track/upload_video", 
                                       files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    print("Video uploaded for verification")
                else:
                    print(f"Failed to upload video: {response.status_code}")
                    
        except Exception as e:
            print(f"Error uploading video: {e}")
    
    def _frame_callback(self, frame: np.ndarray):
        """Callback for each captured frame."""
        if not self.is_running:
            return
        
        # Process frame based on mode
        if self.mode == "local":
            results = self._process_frame_local(frame)
        else:  # remote
            if self.session_id is None:
                if not self._init_remote_session():
                    return
            results = self._process_frame_remote(frame)
            
            # Record video for verification
            if self.video_writer is None:
                self._start_video_recording(frame)
            if self.video_writer is not None:
                self.video_writer.write(frame)
                self.recorded_frames.append(frame.copy())
        
        # Update FPS counter
        self._update_fps()
        results['fps'] = self.current_fps
        
        # Draw visualization
        if self.visualizer:
            display_frame = self.visualizer.draw_tracking_overlay(frame, results)
        else:
            display_frame = frame
        
        # Show frame
        cv2.imshow('Deal or No Deal Tracker', display_frame)
        
        # Check for game completion
        if results.get('tracking_complete'):
            self._handle_game_completion(results)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _handle_game_completion(self, results: Dict[str, Any]):
        """Handle game completion."""
        print("\n" + "="*50)
        print("GAME COMPLETED!")
        print("="*50)
        
        final_results = results.get('final_results', [])
        if final_results:
            print("\nFINAL RANKING (Highest to Lowest):")
            print("-" * 30)
            for i, (case_num, value) in enumerate(final_results):
                print(f"{i+1:2d}. Case {case_num:2d}: ${value:3d}")
        
        # Stop video recording in remote mode
        if self.mode == "remote":
            self._stop_video_recording()
        
        # Show results window
        if self.visualizer and final_results:
            self.visualizer.create_results_window(final_results)
    
    def run(self):
        """Run the tracker client."""
        if not self.initialize():
            return
        
        print("Starting Deal or No Deal Tracker...")
        print("Press 'q' to quit, 'r' to reset, 's' to save screenshot")
        
        self.is_running = True
        
        try:
            # Start webcam capture
            self.webcam.start_capture(self._frame_callback)
            
            # Main loop
            while self.is_running:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_tracker()
                elif key == ord('s'):
                    self._save_screenshot()
                
                time.sleep(0.01)  # Small delay to prevent high CPU usage
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def _reset_tracker(self):
        """Reset the tracker."""
        if self.tracker:
            self.tracker.reset()
        if self.mode == "remote":
            self.session_id = None
        print("Tracker reset")
    
    def _save_screenshot(self):
        """Save a screenshot."""
        if self.webcam:
            frame = self.webcam.capture_single_frame()
            if frame is not None:
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        if self.webcam:
            self.webcam.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Deal or No Deal Case Tracker')
    parser.add_argument('--mode', choices=['local', 'remote'], default='local',
                       help='Processing mode (default: local)')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Create and run client
    client = TrackerClient(args.config)
    client.run()


if __name__ == "__main__":
    main()
