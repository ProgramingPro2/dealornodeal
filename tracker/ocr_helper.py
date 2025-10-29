"""
OCR Helper Module

Handles text recognition for game state detection and case value reading.
Supports both Tesseract and EasyOCR engines.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pytesseract
import easyocr
import re


class OCRHelper:
    """OCR helper for reading numbers and detecting game states."""
    
    def __init__(self, engine: str = "tesseract", confidence_threshold: int = 60):
        """
        Initialize OCR helper.
        
        Args:
            engine: OCR engine to use ("tesseract" or "easyocr")
            confidence_threshold: Minimum confidence for text recognition
        """
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        
        # Initialize OCR engine
        if engine == "easyocr":
            self.reader = easyocr.Reader(['en'])
        elif engine == "tesseract":
            # Configure tesseract
            self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_numbers_tesseract(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Extract numbers using Tesseract OCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of (text, confidence, bbox) tuples
        """
        # Get detailed data from tesseract
        data = pytesseract.image_to_data(
            image, config=self.tesseract_config, output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if conf >= 0 and text:  # Accept any confidence >= 0
                # Split concatenated numbers (e.g., "200100" -> ["200", "100"])
                import re
                numbers = re.findall(r'\d+', text)
                
                for num in numbers:
                    if num.isdigit():
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        # Split bbox for multiple numbers (rough approximation)
                        if len(numbers) > 1:
                            w = w // len(numbers)
                        results.append((num, conf, (x, y, w, h)))
        
        return results
    
    def extract_numbers_easyocr(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Extract numbers using EasyOCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of (text, confidence, bbox) tuples
        """
        results = self.reader.readtext(image)
        
        extracted = []
        for (bbox, text, conf) in results:
            # Clean text and check if it's a number
            clean_text = re.sub(r'[^0-9]', '', text)
            if clean_text and conf > self.confidence_threshold / 100.0:  # EasyOCR uses 0-1 scale
                # Convert bbox to (x, y, w, h) format
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                w, h = x2 - x1, y2 - y1
                
                extracted.append((clean_text, conf * 100, (x1, y1, w, h)))
        
        return extracted
    
    def extract_numbers(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Extract numbers from image using configured OCR engine.
        
        Args:
            image: Input image
            
        Returns:
            List of (text, confidence, bbox) tuples
        """
        preprocessed = self.preprocess_for_ocr(image)
        
        if self.engine == "tesseract":
            return self.extract_numbers_tesseract(preprocessed)
        elif self.engine == "easyocr":
            return self.extract_numbers_easyocr(preprocessed)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def detect_game_start(self, image: np.ndarray, start_triggers: List[int] = [200, 100]) -> bool:
        """
        Detect if game has started by looking for start trigger values.
        
        Args:
            image: Input image
            start_triggers: List of values that indicate game start
            
        Returns:
            True if start detected, False otherwise
        """
        numbers = self.extract_numbers(image)
        
        for text, conf, bbox in numbers:
            try:
                value = int(text)
                if value in start_triggers:
                    return True
            except ValueError:
                continue
        
        return False
    
    def detect_game_end(self, image: np.ndarray, end_triggers: List[int] = [1, 16]) -> bool:
        """
        Detect if game has ended by looking for end trigger values.
        
        Args:
            image: Input image
            end_triggers: List of values that indicate game end
            
        Returns:
            True if end detected, False otherwise
        """
        numbers = self.extract_numbers(image)
        
        for text, conf, bbox in numbers:
            try:
                value = int(text)
                if value in end_triggers:
                    return True
            except ValueError:
                continue
        
        return False
    
    def read_case_values(self, image: np.ndarray, expected_values: List[int]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Read case values from image and return mapping of value to bounding box.
        
        Args:
            image: Input image
            expected_values: List of expected values to look for
            
        Returns:
            Dictionary mapping value to bbox (x, y, w, h)
        """
        numbers = self.extract_numbers(image)
        value_to_bbox = {}
        
        for text, conf, bbox in numbers:
            try:
                value = int(text)
                if value in expected_values:
                    value_to_bbox[value] = bbox
            except ValueError:
                continue
        
        return value_to_bbox
    
    def read_final_case_numbers(self, image: np.ndarray, expected_range: range = range(1, 17)) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Read final case numbers (1-16) from image.
        
        Args:
            image: Input image
            expected_range: Range of expected case numbers
            
        Returns:
            Dictionary mapping case number to bbox (x, y, w, h)
        """
        numbers = self.extract_numbers(image)
        case_to_bbox = {}
        
        for text, conf, bbox in numbers:
            try:
                case_num = int(text)
                if case_num in expected_range:
                    case_to_bbox[case_num] = bbox
            except ValueError:
                continue
        
        return case_to_bbox
    
    def draw_ocr_results(self, image: np.ndarray, 
                        results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw OCR results on image for debugging.
        
        Args:
            image: Input image
            results: OCR results from extract_numbers
            
        Returns:
            Image with OCR results drawn
        """
        output = image.copy()
        
        for text, conf, (x, y, w, h) in results:
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text and confidence
            label = f"{text} ({conf:.1f}%)"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output
