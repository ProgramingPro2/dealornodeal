# Deal or No Deal Case Tracker

A computer vision system for tracking case positions and values in the Deal or No Deal arcade game. The system can run locally or use a remote server for processing, with automatic game detection and case tracking through the shuffle phase.

## Features

- **Automatic Game Detection**: Detects game start (200/100 values) and end (1/16 values)
- **Perspective Correction**: Handles odd viewing angles and auto-crops to game screen
- **Case Tracking**: Uses blob detection + optical flow to track cases during shuffle
- **Dual Processing**: Real-time tracking + post-game video verification
- **Raspberry Pi Optimized**: Designed to run efficiently on Pi 4B+ and Pi 5
- **Web Interface**: Real-time visualization with case highlighting
- **Remote Processing**: Optional API server for offloading heavy computation

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install Tesseract OCR (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Or on macOS
brew install tesseract
```

### 2. Run Local Mode

```bash
# Run with default camera
python -m client.tracker_client --mode local

# Run with specific camera
python -m client.tracker_client --mode local --config config.yaml
```

### 3. Run Remote Mode

```bash
# Terminal 1: Start API server
python -m server.api_server --host 0.0.0.0 --port 8000

# Terminal 2: Start client
python -m client.tracker_client --mode remote --config config.yaml
```

## Usage

### Controls
- `q`: Quit application
- `r`: Reset tracker
- `s`: Save screenshot

### Game Flow
1. **Position camera** to view the arcade screen clearly
2. **Start the tracker** - it will wait for game to begin
3. **Game starts** when it detects values 200 and 100 on screen
4. **Initialization** - tracker reads all case values
5. **Tracking** - follows cases during shuffle phase
6. **Game ends** when it detects case numbers 1 and 16
7. **Results** - displays ranked list of cases by value

## Configuration

Edit `config.yaml` to customize:

```yaml
# Camera settings
camera:
  width: 640
  height: 480
  fps: 30
  device_id: 0

# Processing settings
processing:
  resize_width: 640
  resize_height: 480
  skip_frames: 2  # Process every Nth frame during shuffle

# OCR settings
ocr:
  engine: "tesseract"  # or "easyocr"
  confidence_threshold: 60

# Game values
values:
  expected_values: [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400]
  start_triggers: [200, 100]
  end_triggers: [1, 16]
```

## Raspberry Pi Setup

### Pi 4B+ (Recommended: Remote Mode)
```bash
# Install dependencies
sudo apt update
sudo apt install python3-venv python3-opencv tesseract-ocr

# Run as thin client
python -m client.tracker_client --mode remote
```

### Pi 5 (Can run local mode)
```bash
# Install with optimizations
sudo apt install python3-venv python3-opencv tesseract-ocr

# For better performance, compile OpenCV with NEON
pip install opencv-python-headless

# Run locally
python -m client.tracker_client --mode local
```

## API Endpoints

### Server Endpoints
- `GET /health` - Health check
- `GET /stats` - Server statistics
- `POST /api/track/init` - Initialize tracking session
- `POST /api/track/frame` - Process frame
- `POST /api/track/upload_video` - Upload video for verification
- `GET /api/track/results/{session_id}` - Get results
- `GET /api/track/compare/{session_id}` - Compare live vs video results

### Example API Usage
```python
import requests
import base64
import cv2

# Initialize session
response = requests.post("http://localhost:8000/api/track/init")
session_id = response.json()["session_id"]

# Process frame
ret, frame = cv2.VideoCapture(0).read()
_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode('utf-8')

response = requests.post("http://localhost:8000/api/track/frame", 
                        data={"session_id": session_id, 
                              "frame_base64": frame_base64,
                              "timestamp": time.time()})
results = response.json()
```

## Architecture

### Local Mode
```
Camera → Client → CaseTracker → Visualization
```

### Remote Mode
```
Camera → Client → API Server → CaseTracker → Results
       ↓
   Video Recording → Upload → Verification
```

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different `device_id` values (0, 1, 2...)
   - On Linux: `ls /dev/video*`

2. **Poor tracking accuracy**
   - Ensure good lighting
   - Position camera perpendicular to screen
   - Adjust `confidence_threshold` in config

3. **High CPU usage on Pi**
   - Use remote mode
   - Reduce frame resolution
   - Increase `skip_frames` value

4. **OCR not working**
   - Install Tesseract: `sudo apt install tesseract-ocr`
   - Try EasyOCR: change `engine: "easyocr"` in config

### Performance Tips

- **Pi 4B+**: Use remote mode, reduce resolution to 480p
- **Pi 5**: Can handle local mode at 720p
- **Desktop**: Full resolution local processing
- **Network**: Use local WiFi, not internet for remote mode

## Development

### Project Structure
```
dealornodeal/
├── tracker/           # Core tracking modules
│   ├── case_tracker.py
│   ├── screen_detector.py
│   └── ocr_helper.py
├── client/            # Client application
│   ├── tracker_client.py
│   ├── webcam_capture.py
│   └── visualizer.py
├── server/            # API server
│   ├── api_server.py
│   └── session_manager.py
├── tests/             # Test files
├── config.yaml        # Configuration
└── requirements.txt   # Dependencies
```

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Core logic goes in `tracker/`
2. UI components go in `client/`
3. API endpoints go in `server/`
4. Update `config.yaml` for new settings

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Open an issue on GitHub
- Review the configuration options
