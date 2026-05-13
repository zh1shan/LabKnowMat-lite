import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool

class OCRTextLocator(BaseAtomicTool):
    """
    Tool for detecting texts in the chart and locating their bounding box centers.
    Uses PaddleOCR to extract text and bounding boxes.
    """
    name = "ocr_text_locator"
    description = (
        "Extracts all text elements from the chart image and their center coordinates. "
        "Useful for finding axis labels, ticks, and legends. "
        "Supports English and digits, and can recognize text rotated 90 degrees."
    )

    def __init__(self):
        # Lazy initialization to avoid slow imports if not used
        self.ocr = None

    def _initialize_ocr(self):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            import logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            # lang='en' supports English and digits. 
            # use_angle_cls=True allows recognizing rotated text (e.g., 90 degrees left/right)
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def run(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Run text detection and localization.

        Args:
            image (np.ndarray): The input chart image as a numpy array (BGR or RGB).
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing 'text', 'cx', 'cy', and 'box'.
                Example: [{"text": "50", "cx": 93, "cy": 523, "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...]
        """
        self._initialize_ocr()
        
        results = []
        # PaddleOCR expects a numpy array
        # Return format of ocr.ocr: [[[box_points], (text, confidence)], ...]
        # Note: Depending on PaddleOCR version, it might return a list of lists.
        ocr_result = self.ocr.ocr(image)
        
        if not ocr_result or not ocr_result[0]:
            return results

        # Sometimes paddleocr returns list of lists depending on version/image
        # If cls=True was used it might be different, but we removed cls=True
        lines = ocr_result[0]
        if lines is None:
            return results
            
        for line in lines:
            if not line or len(line) < 2:
                continue
            box = line[0]
            text = line[1][0]
            # Replace easily confused characters
            if text == 'O' or text == 'o':
                text = '0'
            elif text == 'l' or text == 'I':
                text = '1'
            # Calculate center point
            # Box is typically 4 points: [top-left, top-right, bottom-right, bottom-left]
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            cx = sum(x_coords) / 4.0
            cy = sum(y_coords) / 4.0
            
            results.append({
                "text": text,
                "cx": cx,
                "cy": cy,
                "box": box
            })

        return results

