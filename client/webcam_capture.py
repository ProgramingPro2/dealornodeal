"""
Webcam Capture Module

Handles webcam capture and frame processing for the tracker.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import threading
import time


class WebcamCapture:
    """Webcam capture and frame processing."""
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize webcam capture.
        
        Args:
            device_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.capture_thread = None
        self.frame_interval = 1.0 / fps
        
    def initialize(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def start_capture(self, frame_callback: Callable[[np.ndarray], None]):
        """
        Start capturing frames in a separate thread.
        
        Args:
            frame_callback: Function to call with each captured frame
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
        
        if self.is_running:
            raise RuntimeError("Capture already running")
        
        self.frame_callback = frame_callback
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop capturing frames."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        last_frame_time = 0
        
        while self.is_running and self.cap.isOpened():
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Call the frame callback
            if self.frame_callback:
                try:
                    self.frame_callback(frame)
                except Exception as e:
                    print(f"Error in frame callback: {e}")
            
            last_frame_time = current_time
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.
        
        Returns:
            Captured frame or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera properties
        """
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """
        Set camera property.
        
        Args:
            property_id: OpenCV property ID
            value: Property value
            
        Returns:
            True if successful
        """
        if self.cap is None or not self.cap.isOpened():
            return False
        
        return self.cap.set(property_id, value)
    
    def release(self):
        """Release camera resources."""
        self.stop_capture()
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
