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
        import re
        
        # Try multiple PSM modes for better detection
        psm_modes = [6, 7, 8, 11, 13]  # Different page segmentation modes
        all_results = []
        
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
            
            try:
                # Get detailed data from tesseract
                data = pytesseract.image_to_data(
                    image, config=config, output_type=pytesseract.Output.DICT
                )
                
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i]) if data['conf'][i] != -1 else 0
                    
                    if text and len(text) > 0:
                        # Extract all numbers from the text
                        numbers = re.findall(r'\d+', text)
                        
                        for num_str in numbers:
                            if num_str.isdigit() and len(num_str) <= 3:  # Valid case values are 1-3 digits
                                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                                
                                # Try to split concatenated numbers by known values
                                if len(num_str) > 3:
                                    # Split by known patterns (e.g., "200100" -> "200", "100")
                                    potential_splits = self._split_concatenated_numbers(num_str)
                                    for split_num in potential_splits:
                                        all_results.append((split_num, conf, (x, y, w, h)))
                                else:
                                    all_results.append((num_str, conf, (x, y, w, h)))
            except Exception as e:
                continue
        
        # Deduplicate results (keep highest confidence for each unique number+position)
        unique_results = {}
        for text, conf, bbox in all_results:
            key = (text, bbox[0] // 50, bbox[1] // 50)  # Group by text and approximate position
            if key not in unique_results or conf > unique_results[key][1]:
                unique_results[key] = (text, conf, bbox)
        
        return list(unique_results.values())
    
    def _split_concatenated_numbers(self, num_str: str) -> List[str]:
        """Split concatenated numbers based on known expected values."""
        expected = [str(v) for v in [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 80, 100, 150, 200, 400]]
        results = []
        
        # Try to find expected values in the string
        i = 0
        while i < len(num_str):
            found = False
            # Try matching from longest to shortest
            for length in [3, 2, 1]:
                if i + length <= len(num_str):
                    candidate = num_str[i:i+length]
                    if candidate in expected:
                        results.append(candidate)
                        i += length
                        found = True
                        break
            if not found:
                i += 1
        
        return results if results else [num_str]
    
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
        # First try direct extraction
        if self.engine == "tesseract":
            results = self.extract_numbers_tesseract(image)
        elif self.engine == "easyocr":
            results = self.extract_numbers_easyocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
        
        # If no results, try with preprocessing
        if not results:
            preprocessed = self.preprocess_for_ocr(image)
            if self.engine == "tesseract":
                results = self.extract_numbers_tesseract(preprocessed)
            elif self.engine == "easyocr":
                results = self.extract_numbers_easyocr(preprocessed)
        
        # If still no results, try region-based detection
        if not results:
            results = self._extract_numbers_region_based(image)
        
        return results
    
    def _extract_numbers_region_based(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Extract numbers by detecting text regions first, then OCR each region.
        
        Args:
            image: Input image
            
        Returns:
            List of (text, confidence, bbox) tuples
        """
        import re
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Find contours (potential text regions)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (skip very small regions)
            if w < 20 or h < 20 or w > image.shape[1] * 0.8:
                continue
            
            # Add padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # OCR this region
            try:
                config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(region, config=config).strip()
                
                # Extract numbers
                numbers = re.findall(r'\d+', text)
                for num_str in numbers:
                    if num_str.isdigit() and 1 <= len(num_str) <= 3:
                        results.append((num_str, 75, (x, y, w, h)))
            except:
                pass
        
        return results
    
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
