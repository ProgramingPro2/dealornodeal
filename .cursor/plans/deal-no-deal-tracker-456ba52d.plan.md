<!-- 456ba52d-0895-4656-9d47-5cffb9b8237d 632112df-b5c3-47bc-99aa-27d057ad93ef -->
# Deal or No Deal Case Tracker System

## Architecture

### Two-Mode System

1. **Local Mode**: Python client processes video locally using OpenCV
2. **Remote Mode**: Python client captures frames → sends to API server → receives results

### Components to Build

#### 1. Core Tracking Library (`tracker/case_tracker.py`)

- **Screen Detection & Perspective Correction**
        - Detect game screen edges using contour detection
        - Apply perspective transform to correct odd viewing angles
        - Auto-crop to normalized rectangular view

- **Game State Detection**
        - OCR detection for "200" and "100" to trigger START
        - OCR detection for "1" and "16" to trigger STOP
        - Uses Tesseract OCR or EasyOCR (lightweight)

- **Value Recognition Phase** (Initial frame)
        - Detect 16 case regions using blob detection
        - OCR to read values: 2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400
        - Store initial centroid positions with their values

- **Tracking Phase** (During shuffle)
        - Initialize optical flow tracking points on each case centroid
        - Track using Lucas-Kanade sparse optical flow
        - Maintain case IDs using motion prediction and nearest-neighbor matching
        - Handle occlusions by predicting trajectory

- **Final Mapping Phase**
        - Detect final case numbers (1-16) via OCR
        - Map tracked IDs to final positions
        - Output ranked list: case number → value (sorted high to low)

#### 2. Client Application (`client/tracker_client.py`)

- Webcam capture interface
- Mode selection: local vs remote processing
- Real-time visualization overlay showing:
        - Tracked case positions with IDs
        - Current best case(s) highlighted
        - Progress indicator
- Final output: sorted list displayed on screen

#### 3. API Server (`server/api_server.py`)

- Flask/FastAPI REST endpoint
- Accepts: video frames as base64 or multipart upload
- Returns: tracking state and final results as JSON
- Stateful tracking per session ID

#### 4. Configuration (`config.yaml`)

- Server URL for remote mode
- OCR engine selection
- Tracking parameters (blob sizes, flow params)
- Value list for validation

## Technical Implementation Details

### Libraries Required

- **OpenCV**: Blob detection, optical flow, perspective transforms
- **NumPy**: Array operations
- **Tesseract/EasyOCR**: Number recognition
- **Flask/FastAPI**: API server
- **Requests**: Client-server communication

### Tracking Algorithm Specifics

```python
# Initialization (when 200/100 detected)
1. Perspective correction → normalized view
2. Blob detection → 16 regions
3. OCR on regions → value mapping
4. Initialize optical flow points

# Per-frame tracking
1. Calculate sparse optical flow (cv2.calcOpticalFlowPyrLK)
2. Update centroid positions
3. Match to previous IDs using Hungarian algorithm
4. Detect when "1" and "16" appear → STOP

# Final output
1. OCR final case numbers
2. Map: case_number[i] = tracked_value[i]
3. Sort by value descending
4. Return: [(case_num, value), ...]
```

### API Endpoint Design

```
POST /api/track/init
  - Start new tracking session
  - Returns: session_id

POST /api/track/frame
  - Body: {session_id, frame_base64, timestamp}
  - Returns: {status, tracked_positions, debug_image}
  - Note: Client also saves frames locally during capture

POST /api/track/upload_video
  - Body: {session_id, video_file (multipart)}
  - Triggers full reprocessing of complete video
  - Returns: {reprocessed_results, confidence_score, discrepancies}

GET /api/track/results/{session_id}
  - Returns: {complete, ranked_cases, has_video_verification}

GET /api/track/compare/{session_id}
  - Returns: {live_results, video_results, match_status, recommended_result}
```

### Dual Processing Strategy

**Phase 1: Real-time tracking (during game)**

- Pi captures frames at reduced resolution
- Sends frames to server via WebSocket or HTTP POST
- Server processes in real-time and returns preliminary results
- Pi simultaneously records full video locally (full resolution)

**Phase 2: Video verification (after game ends)**

- Pi uploads complete recorded video to server
- Server reprocesses entire video with higher quality settings:
    - Full resolution analysis
    - Multi-pass tracking for accuracy
    - Confidence scoring on final results
- Compares live results vs video results
- If discrepancy detected, flags for review and uses higher-confidence result

**Benefits:**

- Immediate feedback during game
- Post-game verification catches tracking errors
- Full video useful for debugging/improving algorithm
- User gets instant results but with verification safety net

## File Structure

```
/tracker/
  __init__.py
  case_tracker.py       # Core tracking logic
  screen_detector.py    # Perspective correction
  ocr_helper.py         # OCR utilities
  
/client/
  tracker_client.py     # Main client app
  webcam_capture.py     # Camera interface
  visualizer.py         # Overlay display
  
/server/
  api_server.py         # FastAPI server
  session_manager.py    # Track multiple sessions
  
/tests/
  test_tracker.py
  test_ocr.py
  
config.yaml
requirements.txt
README.md
```

## Key Advantages

- **No heavy ML models** (YOLO/detection networks) - uses classical CV only
- **Works on CPU** - optical flow and blob detection are fast
- **Flexible deployment** - local or remote processing
- **Robust to angles** - perspective correction handles tilted screens
- **Auto-detection** - no manual start/stop needed

## Expected Performance

### On Raspberry Pi 5

- **Local mode**: 10-15 FPS (Pi 5 has decent CPU, 2.4GHz quad-core)
- **Remote mode**: Best option - offload heavy processing to server, 15-20 FPS
- **Memory**: 4GB/8GB variants handle OpenCV well

### On Raspberry Pi 4B+

- **Local mode**: 5-10 FPS (slower CPU, will struggle with real-time)
- **Remote mode**: Recommended - server does heavy lifting, Pi just captures/displays
- **Memory**: Use 4GB+ model for stability

### Pi-Specific Optimizations

- **Reduce frame resolution** before processing (640x480 instead of 1080p)
- **Use PiCamera2 library** for efficient camera access (if using Pi Camera module)
- **Skip frames** during shuffle (process every 2nd or 3rd frame)
- **Disable visualization** on Pi in remote mode (just capture and send)
- **Use lightweight OCR**: Tesseract with limited language data or custom digit recognition
- **Compile OpenCV with NEON optimizations** for ARM CPU

### Recommended Setup for Pi

1. **Pi 4B+**: Use remote mode exclusively, Pi acts as thin client
2. **Pi 5**: Can run local mode for testing, but remote mode still faster
3. **Server**: Run on desktop/laptop/cloud for best performance
4. **Network**: Use local WiFi (not internet) for low latency

### Accuracy

- 90%+ if screen clearly visible and cases don't completely overlap
- Same accuracy on Pi vs desktop (just slower processing)

### To-dos

- [x] Initialize project structure, create virtual environment, and setup requirements.txt with OpenCV, NumPy, Tesseract/EasyOCR, Flask/FastAPI
- [x] Implement screen detection and perspective correction module to auto-crop and normalize viewing angle
- [x] Build OCR helper module for detecting game start (200/100), stop (1/16), and reading case values
- [x] Implement main case tracking logic: blob detection initialization, optical flow tracking, and ID maintenance
- [x] Build client application with webcam capture, visualization overlay, and mode selection (local/remote)
- [ ] Create API server with endpoints for session management and frame processing
- [ ] Test complete system with sample video or live webcam, verify ranking output accuracy