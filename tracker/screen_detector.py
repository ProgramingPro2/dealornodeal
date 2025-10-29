"""
Screen Detection and Perspective Correction Module

Handles detection of the game screen edges and applies perspective
transformation to correct for odd viewing angles.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ScreenDetector:
    """Detects game screen and applies perspective correction."""
    
    def __init__(self, min_area: int = 10000, max_corners: int = 4):
        """
        Initialize screen detector.
        
        Args:
            min_area: Minimum area for screen detection
            max_corners: Maximum number of corners to detect
        """
        self.min_area = min_area
        self.max_corners = max_corners
        self.transform_matrix = None
        self.inverse_transform = None
        
    def detect_screen_edges(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect screen edges using contour detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Array of 4 corner points if screen detected, None otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for rectangular screen
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners) and large enough
            if len(approx) == 4 and cv2.contourArea(approx) > self.min_area:
                # Order corners: top-left, top-right, bottom-right, bottom-left
                corners = self._order_points(approx.reshape(4, 2))
                return corners
                
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the format: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left: smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right: largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right: smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left: largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def apply_perspective_correction(self, frame: np.ndarray, corners: np.ndarray, 
                                   output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """
        Apply perspective transformation to correct viewing angle.
        
        Args:
            frame: Input frame
            corners: 4 corner points of the screen
            output_size: Desired output size (width, height)
            
        Returns:
            Perspective-corrected frame
        """
        # Define destination points (rectangle)
        width, height = output_size
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(corners, dst)
        self.inverse_transform = cv2.getPerspectiveTransform(dst, corners)
        
        # Apply transformation
        corrected = cv2.warpPerspective(frame, self.transform_matrix, output_size)
        
        return corrected
    
    def detect_and_correct(self, frame: np.ndarray, 
                          output_size: Tuple[int, int] = (800, 600)) -> Tuple[np.ndarray, bool]:
        """
        Detect screen and apply perspective correction in one step.
        
        Args:
            frame: Input frame
            output_size: Desired output size
            
        Returns:
            Tuple of (corrected_frame, success_flag)
        """
        corners = self.detect_screen_edges(frame)
        
        if corners is not None:
            corrected = self.apply_perspective_correction(frame, corners, output_size)
            return corrected, True
        else:
            return frame, False
    
    def transform_point_to_original(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from corrected coordinates back to original frame coordinates.
        
        Args:
            point: Point in corrected coordinates (x, y)
            
        Returns:
            Point in original coordinates (x, y)
        """
        if self.inverse_transform is None:
            return point
            
        # Convert to homogeneous coordinates
        point_hom = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        
        # Apply inverse transformation
        original_hom = self.inverse_transform @ point_hom
        
        # Convert back to Cartesian coordinates
        x = original_hom[0, 0] / original_hom[2, 0]
        y = original_hom[1, 0] / original_hom[2, 0]
        
        return (x, y)
    
    def draw_detected_screen(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected screen corners on frame for debugging.
        
        Args:
            frame: Input frame
            corners: Detected corner points
            
        Returns:
            Frame with corners drawn
        """
        if corners is not None:
            # Draw corners
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), tuple(corner.astype(int) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw lines connecting corners
            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
        
        return frame
