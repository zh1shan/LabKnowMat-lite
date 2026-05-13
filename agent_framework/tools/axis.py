import cv2
import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool

class AxisLineLocator(BaseAtomicTool):
    """
    Tool for extracting candidate axis lines (horizontal and vertical) from a chart image.
    Uses multi-round iterations to detect dashed and light-colored lines if standard detection fails.
    """
    name = "axis_line_locator"
    description = (
        "Extracts candidate horizontal and vertical axis lines from the chart image. "
        "Returns their coordinates and type. Supports multi-round iterations to find dashed/faint lines."
    )

    def _extract_candidates(self, binary: np.ndarray, gap: int) -> List[Dict[str, Any]]:
        H, W = binary.shape
        min_w = W // 4
        min_h = H // 4
        
        # Extract horizontal lines
        if gap > 0:
            h_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))
            h_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_close_kernel)
        else:
            h_binary = binary
            
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1))
        horizontal_lines_img = cv2.morphologyEx(h_binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Extract vertical lines
        if gap > 0:
            v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))
            v_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_close_kernel)
        else:
            v_binary = binary
            
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_h))
        vertical_lines_img = cv2.morphologyEx(v_binary, cv2.MORPH_OPEN, vertical_kernel)
        
        candidates = []
        
        # Find horizontal line endpoints
        num_labels_h, labels_h = cv2.connectedComponents(horizontal_lines_img, connectivity=8)
        for i in range(1, num_labels_h):
            mask = (labels_h == i).astype(np.uint8) * 255
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x_coords = pts[:, 0, 0]
                y_coords = pts[:, 0, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                # Ensure the actual extracted component length meets the requirement
                if x_max - x_min >= min_w:
                    y_center = int(np.mean(y_coords))
                    candidates.append({
                        "type": "horizontal",
                        "p1": (int(x_min), int(y_center)),
                        "p2": (int(x_max), int(y_center))
                    })
                
        # Find vertical line endpoints
        num_labels_v, labels_v = cv2.connectedComponents(vertical_lines_img, connectivity=8)
        for i in range(1, num_labels_v):
            mask = (labels_v == i).astype(np.uint8) * 255
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x_coords = pts[:, 0, 0]
                y_coords = pts[:, 0, 1]
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                if y_max - y_min >= min_h:
                    x_center = int(np.mean(x_coords))
                    candidates.append({
                        "type": "vertical",
                        "p1": (int(x_center), int(y_min)),
                        "p2": (int(x_center), int(y_max))
                    })
                
        return candidates

    def run(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract candidate lines.

        Args:
            image (np.ndarray): The input chart image.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing 'id', 'type', 'p1', and 'p2'.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Multi-round iteration strategies
        strategies = [
            {"thresh": 200, "gap": 0},    # Round 1: Strict solid dark lines
            {"thresh": 220, "gap": 15},   # Round 2: Light gray and small dashed lines
            {"thresh": 240, "gap": 30},   # Round 3: Very light and larger dashed lines
            {"thresh": 250, "gap": 50},   # Round 4: Extreme light/dashed lines
        ]

        all_candidates = []
        for round_idx, strategy in enumerate(strategies):
            thresh = strategy["thresh"]
            gap = strategy["gap"]
            
            # Since chart background is usually white, use THRESH_BINARY_INV to make lines white
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            
            candidates = self._extract_candidates(binary, gap)
            
            # If we find at least 2 candidate lines (e.g., an x and a y axis), we can stop
            if len(candidates) >= 2:
                all_candidates = candidates
                break
            
            # Keep the max candidates found if we never hit >= 2
            if len(candidates) > len(all_candidates):
                all_candidates = candidates
                
        # Assign IDs to the final candidates
        for i, cand in enumerate(all_candidates):
            cand["id"] = i + 1

        return all_candidates

class AxisTickMapper(BaseAtomicTool):
    """
    Tool for establishing a mathematical mapping between pixel coordinates and numeric values.
    """
    name = "axis_tick_mapper"
    description = (
        "Calculates the linear or logarithmic mapping between pixels and physical values "
        "based on provided text-pixel pairs."
    )

    def run(self, image: np.ndarray, tick_data: List[Dict[str, float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Map pixel coordinates to numerical values.

        Args:
            image (np.ndarray): The input chart image (might not be used, but kept for consistency).
            tick_data (List[Dict[str, float]]): List of dicts with 'px' and 'val' keys.
                Example: [{"px": 523, "val": 50}, {"px": 28, "val": 110}]
        
        Returns:
            Dict[str, Any]: Mapping parameters including 'slope', 'intercept', and 'scale_type'.
        """
        if tick_data is None:
            return {"error": "tick_data is required"}
            
        # TODO: Implement linear/log regression logic to find the mapping
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "scale_type": "linear"
        }
