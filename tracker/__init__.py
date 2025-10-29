"""
Deal or No Deal Case Tracker

A computer vision system for tracking case positions and values
in the Deal or No Deal arcade game.
"""

__version__ = "1.0.0"
__author__ = "Deal or No Deal Tracker"

from .case_tracker import CaseTracker
from .screen_detector import ScreenDetector
from .ocr_helper import OCRHelper

__all__ = ["CaseTracker", "ScreenDetector", "OCRHelper"]
