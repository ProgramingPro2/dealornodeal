#!/usr/bin/env python3
"""
System test script for Deal or No Deal Tracker.

This script tests the core components without requiring a camera.
"""

import sys
import os
import numpy as np
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker import CaseTracker, ScreenDetector, OCRHelper


def create_test_image():
    """Create a test image with simulated game screen."""
    # Create a black image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw a rectangle to simulate the game screen
    cv2.rectangle(img, (100, 100), (700, 500), (50, 50, 50), -1)
    cv2.rectangle(img, (100, 100), (700, 500), (255, 255, 255), 2)
    
    # Add some text to simulate game values
    cv2.putText(img, "200", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, "100", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, "400", (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Add some circles to simulate cases
    for i in range(16):
        x = 150 + (i % 4) * 120
        y = 300 + (i // 4) * 80
        cv2.circle(img, (x, y), 30, (100, 100, 100), -1)
        cv2.circle(img, (x, y), 30, (255, 255, 255), 2)
        cv2.putText(img, str(i+1), (x-10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img


def test_screen_detector():
    """Test screen detection functionality."""
    print("Testing Screen Detector...")
    
    detector = ScreenDetector()
    test_img = create_test_image()
    
    # Test screen detection
    corners = detector.detect_screen_edges(test_img)
    
    if corners is not None:
        print("✓ Screen detection working")
        print(f"  Detected corners: {len(corners)}")
        
        # Test perspective correction
        corrected, success = detector.detect_and_correct(test_img)
        if success:
            print("✓ Perspective correction working")
        else:
            print("✗ Perspective correction failed")
    else:
        print("✗ Screen detection failed")
    
    return corners is not None


def test_ocr_helper():
    """Test OCR functionality."""
    print("\nTesting OCR Helper...")
    
    # Create a larger, clearer test image with numbers
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    
    # Add background
    img.fill(50)
    
    # Draw numbers with better contrast and spacing
    cv2.putText(img, "200", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    cv2.putText(img, "100", (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    cv2.putText(img, "400", (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    
    # Add some borders to make it more realistic
    cv2.rectangle(img, (40, 80), (180, 160), (100, 100, 100), 2)
    cv2.rectangle(img, (240, 80), (380, 160), (100, 100, 100), 2)
    cv2.rectangle(img, (440, 80), (580, 160), (100, 100, 100), 2)
    
    try:
        ocr = OCRHelper(engine="tesseract", confidence_threshold=30)
        numbers = ocr.extract_numbers(img)
        
        if numbers:
            print("✓ OCR working")
            print(f"  Detected numbers: {[n[0] for n in numbers]}")
            
            # Test game start detection
            if ocr.detect_game_start(img, [200, 100]):
                print("✓ Game start detection working")
            else:
                print("✗ Game start detection failed")
        else:
            print("✗ OCR failed to detect numbers")
            # Save the test image for debugging
            cv2.imwrite("test_ocr_image.jpg", img)
            print("  Saved test image as test_ocr_image.jpg for debugging")
            
    except Exception as e:
        print(f"✗ OCR error: {e}")
        return False
    
    return len(numbers) > 0


def test_case_tracker():
    """Test case tracker functionality."""
    print("\nTesting Case Tracker...")
    
    try:
        # Load config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        tracker = CaseTracker(config)
        test_img = create_test_image()
        
        # Test frame processing
        results = tracker.process_frame(test_img)
        
        print("✓ Case tracker initialized")
        print(f"  State: {results.get('state', 'unknown')}")
        print(f"  Cases detected: {len(results.get('cases', {}))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Case tracker error: {e}")
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("Testing Imports...")
    
    try:
        from tracker import CaseTracker, ScreenDetector, OCRHelper
        from client import TrackerClient, WebcamCapture, Visualizer
        from server import app, run_server, SessionManager
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DEAL OR NO DEAL TRACKER - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Screen Detector", test_screen_detector),
        ("OCR Helper", test_ocr_helper),
        ("Case Tracker", test_case_tracker),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nTo run the tracker:")
        print("  Local mode:  python run_local.py")
        print("  Remote mode: python run_remote.py")
        print("  Server:      python run_server.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
