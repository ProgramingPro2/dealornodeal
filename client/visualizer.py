"""
Visualization Module

Handles real-time visualization overlay for the tracker.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time


class Visualizer:
    """Real-time visualization overlay for tracking results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.overlay_alpha = config.get('visualization', {}).get('overlay_alpha', 0.7)
        self.show_tracking = config.get('visualization', {}).get('show_tracking', True)
        self.show_confidence = config.get('visualization', {}).get('show_confidence', True)
        self.highlight_best = config.get('visualization', {}).get('highlight_best', True)
        
        # Colors (BGR format)
        self.colors = {
            'tracking': (0, 255, 0),      # Green
            'lost': (0, 0, 255),          # Red
            'best': (0, 255, 255),        # Yellow
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0),      # Black
            'state_waiting': (128, 128, 128),    # Gray
            'state_initializing': (255, 165, 0),  # Orange
            'state_tracking': (0, 255, 0),        # Green
            'state_completed': (0, 0, 255)        # Red
        }
    
    def draw_tracking_overlay(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw tracking overlay on frame.
        
        Args:
            frame: Input frame
            results: Tracking results from CaseTracker
            
        Returns:
            Frame with overlay drawn
        """
        if not self.show_tracking:
            return frame
        
        overlay = frame.copy()
        
        # Draw state indicator
        self._draw_state_indicator(overlay, results)
        
        # Draw cases
        if 'cases' in results:
            self._draw_cases(overlay, results['cases'])
        
        # Draw final results
        if results.get('final_results'):
            self._draw_final_results(overlay, results['final_results'])
        
        # Draw game status
        self._draw_game_status(overlay, results)
        
        # Blend overlay with original frame
        output = cv2.addWeighted(frame, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0)
        
        return output
    
    def _draw_state_indicator(self, frame: np.ndarray, results: Dict[str, Any]):
        """Draw current state indicator."""
        state = results.get('state', 'waiting')
        state_colors = {
            'waiting': self.colors['state_waiting'],
            'initializing': self.colors['state_initializing'],
            'tracking': self.colors['state_tracking'],
            'completed': self.colors['state_completed']
        }
        
        color = state_colors.get(state, self.colors['text'])
        
        # Draw state background
        cv2.rectangle(frame, (10, 10), (200, 60), self.colors['background'], -1)
        cv2.rectangle(frame, (10, 10), (200, 60), color, 2)
        
        # Draw state text
        cv2.putText(frame, f"State: {state.upper()}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_cases(self, frame: np.ndarray, cases: Dict[int, Any]):
        """Draw tracked cases."""
        if not cases:
            return
        
        # Find best case (highest value)
        best_case = None
        best_value = -1
        for case in cases.values():
            if case.get('value') and case['value'] > best_value:
                best_value = case['value']
                best_case = case
        
        for case_id, case in cases.items():
            centroid = case.get('centroid', (0, 0))
            bbox = case.get('bbox', (0, 0, 0, 0))
            value = case.get('value')
            confidence = case.get('confidence', 0)
            
            # Determine color
            if self.highlight_best and case == best_case:
                color = self.colors['best']
                thickness = 3
            else:
                color = self.colors['tracking']
                thickness = 2
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw centroid
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, color, -1)
            
            # Draw case info
            info_text = f"ID:{case_id}"
            if value is not None:
                info_text += f" Val:{value}"
            if self.show_confidence:
                info_text += f" ({confidence:.1f})"
            
            # Draw text background
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 5, y - 5), 
                         self.colors['background'], -1)
            
            # Draw text
            cv2.putText(frame, info_text, (x + 2, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_final_results(self, frame: np.ndarray, final_results: List[Tuple[int, int]]):
        """Draw final results ranking."""
        if not final_results:
            return
        
        # Create results panel
        panel_width = 300
        panel_height = min(400, len(final_results) * 30 + 50)
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['background'], -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['text'], 2)
        
        # Draw title
        cv2.putText(frame, "FINAL RANKING", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Draw results
        y_offset = 50
        for i, (case_num, value) in enumerate(final_results[:10]):  # Show top 10
            # Highlight top 3
            if i < 3:
                color = self.colors['best']
                thickness = 2
            else:
                color = self.colors['text']
                thickness = 1
            
            result_text = f"{i+1:2d}. Case {case_num:2d}: ${value:3d}"
            cv2.putText(frame, result_text, (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            y_offset += 25
    
    def _draw_game_status(self, frame: np.ndarray, results: Dict[str, Any]):
        """Draw game status indicators."""
        status_y = frame.shape[0] - 30
        
        # Game started indicator
        if results.get('game_started'):
            cv2.putText(frame, "GAME STARTED!", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['best'], 2)
        
        # Game ended indicator
        if results.get('game_ended'):
            cv2.putText(frame, "GAME ENDED!", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['state_completed'], 2)
        
        # Tracking complete indicator
        if results.get('tracking_complete'):
            cv2.putText(frame, "TRACKING COMPLETE!", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['tracking'], 2)
    
    def draw_debug_info(self, frame: np.ndarray, debug_info: Dict[str, Any]) -> np.ndarray:
        """
        Draw debug information.
        
        Args:
            frame: Input frame
            debug_info: Debug information dictionary
            
        Returns:
            Frame with debug info drawn
        """
        debug_frame = frame.copy()
        
        # Draw FPS
        if 'fps' in debug_info:
            cv2.putText(debug_frame, f"FPS: {debug_info['fps']:.1f}", 
                       (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.colors['text'], 2)
        
        # Draw processing time
        if 'processing_time' in debug_info:
            cv2.putText(debug_frame, f"Process: {debug_info['processing_time']:.1f}ms", 
                       (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.colors['text'], 2)
        
        # Draw case count
        if 'case_count' in debug_info:
            cv2.putText(debug_frame, f"Cases: {debug_info['case_count']}", 
                       (frame.shape[1] - 100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.colors['text'], 2)
        
        return debug_frame
    
    def create_results_window(self, final_results: List[Tuple[int, int]], 
                            window_name: str = "Final Results") -> None:
        """
        Create a separate window showing final results.
        
        Args:
            final_results: List of (case_number, value) tuples
            window_name: Window name
        """
        if not final_results:
            return
        
        # Create results image
        img_height = len(final_results) * 40 + 100
        img_width = 400
        results_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Draw title
        cv2.putText(results_img, "DEAL OR NO DEAL RESULTS", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # Draw results
        y_offset = 80
        for i, (case_num, value) in enumerate(final_results):
            # Highlight top 3
            if i < 3:
                color = self.colors['best']
                thickness = 2
            else:
                color = self.colors['text']
                thickness = 1
            
            result_text = f"{i+1:2d}. Case {case_num:2d}: ${value:3d}"
            cv2.putText(results_img, result_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
            y_offset += 40
        
        # Show window
        cv2.imshow(window_name, results_img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
