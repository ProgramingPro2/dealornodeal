"""
Session Manager Module

Handles tracking sessions for the API server.
"""

import uuid
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import threading

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import CaseTracker


@dataclass
class TrackingSession:
    """Represents a tracking session."""
    session_id: str
    tracker: CaseTracker
    created_at: float
    last_activity: float
    video_file_path: Optional[str] = None
    live_results: Optional[list] = None
    video_results: Optional[list] = None
    verification_complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    """Manages tracking sessions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize session manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sessions: Dict[str, TrackingSession] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.max_session_age = 3600  # 1 hour
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def create_session(self) -> str:
        """
        Create a new tracking session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        tracker = CaseTracker(self.config)
        
        session = TrackingSession(
            session_id=session_id,
            tracker=tracker,
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[TrackingSession]:
        """
        Get a tracking session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Tracking session or None if not found
        """
        return self.sessions.get(session_id)
    
    def process_frame(self, session_id: str, frame_data: bytes) -> Dict[str, Any]:
        """
        Process a frame for a session.
        
        Args:
            session_id: Session ID
            frame_data: Frame data as bytes
            
        Returns:
            Processing results
        """
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        with session.lock:
            try:
                import cv2
                import numpy as np
                
                # Decode frame
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {'error': 'Failed to decode frame'}
                
                # Process frame
                results = session.tracker.process_frame(frame)
                
                # Update session activity
                session.last_activity = time.time()
                
                # Store live results if tracking is complete
                if results.get('tracking_complete') and results.get('final_results'):
                    session.live_results = results['final_results']
                
                return results
                
            except Exception as e:
                return {'error': f'Processing failed: {str(e)}'}
    
    def upload_video(self, session_id: str, video_file_path: str) -> Dict[str, Any]:
        """
        Upload video for verification.
        
        Args:
            session_id: Session ID
            video_file_path: Path to video file
            
        Returns:
            Upload results
        """
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        with session.lock:
            session.video_file_path = video_file_path
            
            # Process video for verification
            try:
                verification_results = self._process_video_for_verification(video_file_path)
                session.video_results = verification_results
                session.verification_complete = True
                
                return {
                    'success': True,
                    'video_results': verification_results,
                    'verification_complete': True
                }
                
            except Exception as e:
                return {'error': f'Video processing failed: {str(e)}'}
    
    def _process_video_for_verification(self, video_file_path: str) -> list:
        """
        Process video file for verification.
        
        Args:
            video_file_path: Path to video file
            
        Returns:
            Verification results
        """
        import cv2
        
        # Create a new tracker for video processing
        video_tracker = CaseTracker(self.config)
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every few frames to save computation
            if frame_count % 3 == 0:  # Process every 3rd frame
                results = video_tracker.process_frame(frame)
                
                # If tracking is complete, return results
                if results.get('tracking_complete') and results.get('final_results'):
                    cap.release()
                    return results['final_results']
            
            frame_count += 1
        
        cap.release()
        return []
    
    def get_results(self, session_id: str) -> Dict[str, Any]:
        """
        Get results for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session results
        """
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        with session.lock:
            return {
                'session_id': session_id,
                'live_results': session.live_results,
                'video_results': session.video_results,
                'verification_complete': session.verification_complete,
                'created_at': session.created_at,
                'last_activity': session.last_activity
            }
    
    def compare_results(self, session_id: str) -> Dict[str, Any]:
        """
        Compare live and video results.
        
        Args:
            session_id: Session ID
            
        Returns:
            Comparison results
        """
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        with session.lock:
            live_results = session.live_results or []
            video_results = session.video_results or []
            
            # Compare results
            match_status = "unknown"
            recommended_result = live_results
            
            if live_results and video_results:
                # Simple comparison - check if top 3 cases match
                live_top3 = [case for case, _ in live_results[:3]]
                video_top3 = [case for case, _ in video_results[:3]]
                
                if set(live_top3) == set(video_top3):
                    match_status = "match"
                    recommended_result = video_results  # Prefer video results
                else:
                    match_status = "mismatch"
                    # Use video results as they're more reliable
                    recommended_result = video_results
            elif video_results:
                match_status = "video_only"
                recommended_result = video_results
            elif live_results:
                match_status = "live_only"
                recommended_result = live_results
            
            return {
                'session_id': session_id,
                'live_results': live_results,
                'video_results': video_results,
                'match_status': match_status,
                'recommended_result': recommended_result,
                'verification_complete': session.verification_complete
            }
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Clean up video file if it exists
            if session.video_file_path and os.path.exists(session.video_file_path):
                try:
                    os.remove(session.video_file_path)
                except:
                    pass
            
            del self.sessions[session_id]
            return True
        
        return False
    
    def _start_cleanup_thread(self):
        """Start cleanup thread for old sessions."""
        def cleanup_loop():
            while True:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_sessions()
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions."""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.max_session_age:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.delete_session(session_id)
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        current_time = time.time()
        active_sessions = 0
        completed_sessions = 0
        
        for session in self.sessions.values():
            if session.last_activity > current_time - 300:  # Active in last 5 minutes
                active_sessions += 1
            if session.verification_complete:
                completed_sessions += 1
        
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': active_sessions,
            'completed_sessions': completed_sessions
        }
