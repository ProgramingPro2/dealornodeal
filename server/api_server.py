"""
FastAPI Server for Deal or No Deal Tracker

Provides REST API endpoints for remote tracking processing.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import yaml
import base64
import os
import time
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deal or No Deal Tracker API",
    description="API for tracking cases in Deal or No Deal arcade game",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session manager
session_manager: Optional[SessionManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    global session_manager
    
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = get_default_config()
    
    # Initialize session manager
    session_manager = SessionManager(config)
    logger.info("API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API server shutting down")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'camera': {'width': 640, 'height': 480, 'fps': 30, 'device_id': 0},
        'processing': {'resize_width': 640, 'resize_height': 480, 'skip_frames': 2},
        'visualization': {'show_tracking': True, 'show_confidence': True, 'highlight_best': True},
        'values': {'expected_values': [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400]}
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    return session_manager.get_session_stats()


@app.post("/api/track/init")
async def init_tracking():
    """Initialize a new tracking session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    try:
        session_id = session_manager.create_session()
        logger.info(f"Created new tracking session: {session_id}")
        return {"session_id": session_id, "status": "initialized"}
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/api/track/frame")
async def process_frame(
    session_id: str = Form(...),
    frame_base64: str = Form(...),
    timestamp: float = Form(...)
):
    """Process a frame for tracking."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(frame_base64)
        
        # Process frame
        results = session_manager.process_frame(session_id, frame_data)
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process frame for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


@app.post("/api/track/upload_video")
async def upload_video(
    session_id: str = Form(...),
    video_file: UploadFile = File(...)
):
    """Upload video for verification."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    try:
        # Save uploaded file
        video_filename = f"session_{session_id}_{int(time.time())}.mp4"
        video_path = os.path.join("/tmp", video_filename)
        
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Process video
        results = session_manager.upload_video(session_id, video_path)
        
        if 'error' in results:
            # Clean up file on error
            if os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=400, detail=results['error'])
        
        logger.info(f"Video uploaded and processed for session {session_id}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to upload video for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


@app.get("/api/track/results/{session_id}")
async def get_results(session_id: str):
    """Get results for a session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    results = session_manager.get_results(session_id)
    
    if 'error' in results:
        raise HTTPException(status_code=404, detail=results['error'])
    
    return results


@app.get("/api/track/compare/{session_id}")
async def compare_results(session_id: str):
    """Compare live and video results for a session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    results = session_manager.compare_results(session_id)
    
    if 'error' in results:
        raise HTTPException(status_code=404, detail=results['error'])
    
    return results


@app.delete("/api/track/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a tracking session."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    if session_manager.delete_session(session_id):
        logger.info(f"Deleted session: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/track/sessions")
async def list_sessions():
    """List all active sessions."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    sessions = []
    for session_id, session in session_manager.sessions.items():
        sessions.append({
            'session_id': session_id,
            'created_at': session.created_at,
            'last_activity': session.last_activity,
            'verification_complete': session.verification_complete
        })
    
    return {"sessions": sessions, "count": len(sessions)}


def run_server(host: str = "0.0.0.0", port: int = 8000, config_path: str = "config.yaml"):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to configuration file
    """
    # Set environment variable for config path
    os.environ["CONFIG_PATH"] = config_path
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deal or No Deal Tracker API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.config)
