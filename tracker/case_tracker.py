"""
Core Case Tracking Module

Implements the main tracking logic using blob detection initialization
and optical flow tracking with ID maintenance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from .screen_detector import ScreenDetector
from .ocr_helper import OCRHelper


class GameState(Enum):
    """Game state enumeration."""
    WAITING = "waiting"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    COMPLETED = "completed"


@dataclass
class Case:
    """Represents a tracked case."""
    id: int
    value: Optional[int]
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    last_seen: float
    confidence: float
    trajectory: List[Tuple[float, float]]


class CaseTracker:
    """Main case tracking class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize case tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.screen_detector = ScreenDetector()
        self.ocr_helper = OCRHelper(
            engine=config.get('ocr', {}).get('engine', 'tesseract'),
            confidence_threshold=config.get('ocr', {}).get('confidence_threshold', 60)
        )
        
        # Tracking state
        self.state = GameState.WAITING
        self.cases: Dict[int, Case] = {}
        self.next_case_id = 0
        self.max_disappeared = config.get('processing', {}).get('max_disappeared', 10)
        self.max_distance = config.get('processing', {}).get('max_distance', 100)
        
        # Optical flow parameters
        self.flow_params = {
            'winSize': tuple(config.get('processing', {}).get('flow_win_size', (15, 15))),
            'maxLevel': config.get('processing', {}).get('flow_max_level', 2),
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                        config.get('processing', {}).get('flow_criteria', (10, 0.03))[0],
                        config.get('processing', {}).get('flow_criteria', (10, 0.03))[1])
        }
        
        # Blob detection parameters
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.minArea = config.get('processing', {}).get('blob_min_area', 100)
        self.blob_params.maxArea = config.get('processing', {}).get('blob_max_area', 5000)
        self.blob_params.thresholdStep = 10
        self.blob_params.minThreshold = 50
        self.blob_params.maxThreshold = 200
        self.blob_params.minRepeatability = 2
        self.blob_params.minDistBetweenBlobs = 10
        self.blob_params.filterByColor = True
        self.blob_params.blobColor = 255
        self.blob_params.filterByArea = True
        self.blob_params.filterByCircularity = False
        self.blob_params.filterByConvexity = False
        self.blob_params.filterByInertia = False
        
        self.blob_detector = cv2.SimpleBlobDetector_create(self.blob_params)
        
        # Expected values
        self.expected_values = config.get('values', {}).get('expected_values', 
            [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400])
        self.start_triggers = config.get('values', {}).get('start_triggers', [200, 100])
        self.end_triggers = config.get('values', {}).get('end_triggers', [1, 16])
        
        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_keypoints = None
        
        # Results
        self.final_results: List[Tuple[int, int]] = []  # (case_number, value)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and update tracking state.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with tracking results and debug info
        """
        current_time = time.time()
        results = {
            'state': self.state.value,
            'cases': {},
            'debug_image': frame.copy(),
            'game_started': False,
            'game_ended': False
        }
        
        # Apply screen detection and perspective correction
        corrected_frame, screen_detected = self.screen_detector.detect_and_correct(frame)
        
        if not screen_detected:
            return results
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        
        # State machine
        if self.state == GameState.WAITING:
            self._handle_waiting_state(corrected_frame, results)
        elif self.state == GameState.INITIALIZING:
            self._handle_initializing_state(corrected_frame, gray, results)
        elif self.state == GameState.TRACKING:
            self._handle_tracking_state(corrected_frame, gray, results)
        elif self.state == GameState.COMPLETED:
            self._handle_completed_state(results)
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        return results
    
    def _handle_waiting_state(self, frame: np.ndarray, results: Dict[str, Any]):
        """Handle waiting for game to start."""
        if self.ocr_helper.detect_game_start(frame, self.start_triggers):
            self.state = GameState.INITIALIZING
            results['game_started'] = True
    
    def _handle_initializing_state(self, frame: np.ndarray, gray: np.ndarray, results: Dict[str, Any]):
        """Handle initialization phase - detect case values."""
        # Detect blobs (cases)
        keypoints = self.blob_detector.detect(gray)
        
        if len(keypoints) >= 12:  # Need at least 12 cases to be confident
            # Extract case values using OCR
            value_to_bbox = self.ocr_helper.read_case_values(frame, self.expected_values)
            
            if len(value_to_bbox) >= 8:  # Need at least 8 values
                # Match blobs to values
                self._match_blobs_to_values(keypoints, value_to_bbox)
                self.state = GameState.TRACKING
                results['initialization_complete'] = True
            else:
                # Not enough values detected, keep trying
                pass
        else:
            # Not enough blobs detected, keep trying
            pass
    
    def _handle_tracking_state(self, frame: np.ndarray, gray: np.ndarray, results: Dict[str, Any]):
        """Handle tracking phase - track cases during shuffle."""
        if self.prev_gray is not None and len(self.cases) > 0:
            # Prepare points for optical flow
            points = np.array([[case.centroid[0], case.centroid[1]] for case in self.cases.values()], 
                             dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, points, None, **self.flow_params
            )
            
            # Update case positions
            valid_cases = []
            for i, (case_id, case) in enumerate(self.cases.items()):
                if status[i, 0] == 1 and error[i, 0] < 50:  # Valid tracking
                    new_centroid = (float(new_points[i, 0, 0]), float(new_points[i, 0, 1]))
                    case.centroid = new_centroid
                    case.last_seen = time.time()
                    case.trajectory.append(new_centroid)
                    
                    # Keep only recent trajectory points
                    if len(case.trajectory) > 20:
                        case.trajectory = case.trajectory[-20:]
                    
                    valid_cases.append(case_id)
                else:
                    # Case lost, try to recover using trajectory prediction
                    if len(case.trajectory) >= 2:
                        # Predict next position based on recent movement
                        recent_points = case.trajectory[-2:]
                        dx = recent_points[1][0] - recent_points[0][0]
                        dy = recent_points[1][1] - recent_points[0][1]
                        predicted_centroid = (case.centroid[0] + dx, case.centroid[1] + dy)
                        case.centroid = predicted_centroid
                        case.trajectory.append(predicted_centroid)
                        valid_cases.append(case_id)
            
            # Remove cases that have been lost for too long
            current_time = time.time()
            cases_to_remove = []
            for case_id, case in self.cases.items():
                if current_time - case.last_seen > self.max_disappeared * 0.033:  # Assuming 30 FPS
                    cases_to_remove.append(case_id)
            
            for case_id in cases_to_remove:
                del self.cases[case_id]
        
        # Check for game end
        if self.ocr_helper.detect_game_end(frame, self.end_triggers):
            self._finalize_tracking(frame)
            self.state = GameState.COMPLETED
            results['game_ended'] = True
    
    def _handle_completed_state(self, results: Dict[str, Any]):
        """Handle completed state - return final results."""
        results['final_results'] = self.final_results
        results['tracking_complete'] = True
    
    def _match_blobs_to_values(self, keypoints: List[cv2.KeyPoint], value_to_bbox: Dict[int, Tuple[int, int, int, int]]):
        """Match detected blobs to OCR values."""
        self.cases = {}
        self.next_case_id = 0
        
        for value, (vx, vy, vw, vh) in value_to_bbox.items():
            # Find closest blob to this value
            best_keypoint = None
            min_distance = float('inf')
            
            for kp in keypoints:
                distance = np.sqrt((kp.pt[0] - (vx + vw/2))**2 + (kp.pt[1] - (vy + vh/2))**2)
                if distance < min_distance and distance < 100:  # Within reasonable distance
                    min_distance = distance
                    best_keypoint = kp
            
            if best_keypoint is not None:
                case = Case(
                    id=self.next_case_id,
                    value=value,
                    centroid=best_keypoint.pt,
                    bbox=(int(best_keypoint.pt[0] - best_keypoint.size/2), 
                          int(best_keypoint.pt[1] - best_keypoint.size/2),
                          int(best_keypoint.size), int(best_keypoint.size)),
                    last_seen=time.time(),
                    confidence=1.0,
                    trajectory=[best_keypoint.pt]
                )
                self.cases[self.next_case_id] = case
                self.next_case_id += 1
    
    def _finalize_tracking(self, frame: np.ndarray):
        """Finalize tracking by matching to final case numbers."""
        # Read final case numbers (1-16)
        case_to_bbox = self.ocr_helper.read_final_case_numbers(frame, range(1, 17))
        
        if len(case_to_bbox) >= 8:  # Need at least 8 case numbers
            # Match tracked cases to final positions
            final_mapping = {}
            
            for case_num, (cx, cy, cw, ch) in case_to_bbox.items():
                case_center = (cx + cw/2, cy + ch/2)
                
                # Find closest tracked case
                best_case = None
                min_distance = float('inf')
                
                for case in self.cases.values():
                    distance = np.sqrt((case.centroid[0] - case_center[0])**2 + 
                                     (case.centroid[1] - case_center[1])**2)
                    if distance < min_distance and distance < 150:  # Within reasonable distance
                        min_distance = distance
                        best_case = case
                
                if best_case is not None:
                    final_mapping[case_num] = best_case.value
            
            # Create final results sorted by value (highest first)
            self.final_results = sorted(final_mapping.items(), key=lambda x: x[1], reverse=True)
    
    def get_case_info(self) -> Dict[int, Dict[str, Any]]:
        """Get current case information for visualization."""
        case_info = {}
        for case_id, case in self.cases.items():
            case_info[case_id] = {
                'id': case.id,
                'value': case.value,
                'centroid': case.centroid,
                'bbox': case.bbox,
                'confidence': case.confidence,
                'last_seen': case.last_seen
            }
        return case_info
    
    def reset(self):
        """Reset tracker to initial state."""
        self.state = GameState.WAITING
        self.cases = {}
        self.next_case_id = 0
        self.prev_gray = None
        self.prev_keypoints = None
        self.final_results = []
    
    def draw_debug_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw debug information on frame."""
        debug_frame = frame.copy()
        
        # Draw cases
        for case in self.cases.values():
            x, y, w, h = case.bbox
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_frame, (int(case.centroid[0]), int(case.centroid[1])), 5, (0, 255, 0), -1)
            
            # Draw case ID and value
            label = f"ID:{case.id}"
            if case.value is not None:
                label += f" Val:{case.value}"
            cv2.putText(debug_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw state
        cv2.putText(debug_frame, f"State: {self.state.value}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return debug_frame
