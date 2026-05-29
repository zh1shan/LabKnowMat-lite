import numpy as np
import random
from typing import Dict, Any, List
from .base import BaseAtomicTool
from ._color_cluster_utils import extract_colors_density

class ColorClusterPointSampler(BaseAtomicTool):
    """
    Extracts physical coordinate points for N specific colors by random sampling 
    from the highest density core color areas.
    """
    name: str = "color_cluster_point_sampler"
    description: str = (
        "Extracts physical coordinate points (x,y) for specific colors. "
        "Useful for line charts to sample physical points on lines of different colors. "
        "Note: DO NOT use this for Scatter plots. If it is a Scatter plot, "
        "please prioritize using scatter_point_extractor_v1 instead. "
        "Returns the mean RGB color of each cluster and a list of sampled (x,y) physical coordinates."
    )

    def run(self, image: np.ndarray, num_colors: int, sample_size: int, core_ratio: float = 0.5, target_rect: List[int] = None) -> Dict[str, Any]:
        """
        Execute the tool's core function.

        Args:
            image (np.ndarray): The input chart image (BGR or RGB).
            num_colors (int): The number of colors to cluster into (c).
            sample_size (int): Number of points to randomly sample from the core pixels (n).
            core_ratio (float): Ratio of core pixels to extract. Defaults to 0.5.
            target_rect (List[int]): Optional [x_min, y_min, x_max, y_max].

        Returns:
            Dict[str, Any]: A flat JSON-compatible dictionary containing the extraction results.
        """
        # Ensure image is in RGB since extract_colors_density expects RGB
        # LabKnowMat-lite images might be BGR if directly from cv2, or RGB if preprocessed.
        # Check standard convention in other tools. Usually BaseAtomicTool receives cv2 BGR image.
        # For safety, let's assume it's BGR if it has 3 channels and the standard OpenCV is used, 
        # but the doc says "OpenCV loaded numpy image matrix". Let's convert BGR to RGB.
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
            cluster_idx = res['cluster_idx']
            mean_rgb = res['mean_rgb']
            core_coords = res['core_coords']
            
            # Sample coordinates
            if len(core_coords) > sample_size:
                sampled_coords = random.sample(core_coords, sample_size)
            else:
                sampled_coords = core_coords
                
            output.append({
                "cluster_idx": cluster_idx,
                "mean_color": mean_rgb,
                "sampled_points": sampled_coords
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
                "sample_size": {
                    "type": "integer",
                    "description": "The number of points to randomly sample from the core pixels for each color (n). Decide based on chart complexity."
                },
                "core_ratio": {
                    "type": "number",
                    "description": "Ratio of core pixels to extract (r). Defaults to 0.5 for line charts.",
                    "default": 0.5
                },
                "target_rect": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional bounding box [x_min, y_min, x_max, y_max] to restrict the processing area."
                }
            },
            "required": ["num_colors", "sample_size"]
        }
