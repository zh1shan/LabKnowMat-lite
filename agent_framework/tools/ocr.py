import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool

class OCRTextLocator(BaseAtomicTool):
    """
    Tool for detecting texts in the chart and locating their bounding box centers.
    """
    name = "ocr_text_locator"
    description = (
        "Extracts all text elements from the chart image and their center coordinates. "
        "Useful for finding axis labels, ticks, and legends."
    )

    def run(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Run text detection and localization.

        Args:
            image (np.ndarray): The input chart image.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing 'text', 'cx', 'cy'.
                Example: [{"text": "50", "cx": 93, "cy": 523}, ...]
        """
        # TODO: Implement OCR logic using PaddleOCR or similar
        return []
