import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool
from ._color_cluster_utils import extract_colors_density

class ColorClusterPixelCounter(BaseAtomicTool):
    """
    Counts the number of core pixels for N specific colors to calculate area or proportions.
    """
    name: str = "color_cluster_pixel_counter"
    description: str = (
        "Counts the number of core pixels for specific colors. "
        "Useful for calculating proportions or areas in stacked bar charts or irregular pie charts. "
        "For regular circular pie charts, prioritize using pie_slice_extractor. "
        "Returns the mean RGB color of each cluster and the total number of its core pixels."
    )

    def run(self, image: np.ndarray, num_colors: int, core_ratio: float = 0.9, target_rect: List[int] = None) -> Dict[str, Any]:
        """
        Execute the tool's core function.

        Args:
            image (np.ndarray): The input chart image (BGR or RGB).
            num_colors (int): The number of colors to cluster into (c).
            core_ratio (float): Ratio of core pixels to extract. Defaults to 0.9.
            target_rect (List[int]): Optional [x_min, y_min, x_max, y_max].

        Returns:
            Dict[str, Any]: A flat JSON-compatible dictionary containing the extraction results.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[..., ::-1].copy() # BGR to RGB
        else:
            image_rgb = image.copy()
            
        results = extract_colors_density(
            image_rgb=image_rgb,
            n_colors=num_colors,
            core_ratio=core_ratio,
            target_rect=target_rect
        )
        
        output = []
        for res in results:
            output.append({
                "cluster_idx": res['cluster_idx'],
                "mean_color": res['mean_rgb'],
                "pixel_count": res['core_count']
            })
            
        return {"clusters": output}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_colors": {
                    "type": "integer",
                    "description": "The expected number of colors in the target area (c)."
                },
                "core_ratio": {
                    "type": "number",
                    "description": "Ratio of core pixels to extract (r). Defaults to 0.9 for calculating areas.",
                    "default": 0.9
                },
                "target_rect": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional bounding box [x_min, y_min, x_max, y_max] to restrict the processing area."
                }
            },
            "required": ["num_colors"]
        }
