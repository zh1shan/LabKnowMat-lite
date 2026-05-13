import cv2
import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool

class AxisLineLocator(BaseAtomicTool):
    """
    Tool for extracting candidate axis lines (horizontal and vertical) from a chart image.
    """
    name = "axis_line_locator"
    description = (
        "Extracts candidate horizontal and vertical axis lines from the chart image. "
        "Returns their coordinates and type."
    )

    def run(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract candidate lines.

        Args:
            image (np.ndarray): The input chart image.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing 'id', 'type', 'p1', and 'p2'.
                Example: [{"id": 1, "type": "horizontal", "p1": (100, 500), "p2": (600, 500)}, ...]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 由于图表底色一般为白，线为黑，使用 THRESH_BINARY_INV 使线段变为前景(白色)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        H, W = binary.shape
        min_w = W // 4
        min_h = H // 4
        
        # 提取水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1))
        horizontal_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 提取垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_h))
        vertical_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        candidates = []
        line_id = 1
        
        # 提取水平线端点
        num_labels_h, labels_h, stats_h, centroids_h = cv2.connectedComponentsWithStats(horizontal_lines_img, connectivity=8)
        for i in range(1, num_labels_h):
            mask = (labels_h == i).astype(np.uint8) * 255
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x_coords = pts[:, 0, 0]
                y_coords = pts[:, 0, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_center = int(np.mean(y_coords))
                candidates.append({
                    "id": line_id,
                    "type": "horizontal",
                    "p1": (int(x_min), int(y_center)),
                    "p2": (int(x_max), int(y_center))
                })
                line_id += 1
                
        # 提取垂直线端点
        num_labels_v, labels_v, stats_v, centroids_v = cv2.connectedComponentsWithStats(vertical_lines_img, connectivity=8)
        for i in range(1, num_labels_v):
            mask = (labels_v == i).astype(np.uint8) * 255
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x_coords = pts[:, 0, 0]
                y_coords = pts[:, 0, 1]
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                x_center = int(np.mean(x_coords))
                candidates.append({
                    "id": line_id,
                    "type": "vertical",
                    "p1": (int(x_center), int(y_min)),
                    "p2": (int(x_center), int(y_max))
                })
                line_id += 1
                
        return candidates

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
