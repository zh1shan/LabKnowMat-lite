import numpy as np
from typing import Dict, Any, List
from .base import BaseAtomicTool


class RegionIdentifier(BaseAtomicTool):
    name = "region_identifier"
    description = (
        "Identifies the connected color region containing a given seed point using flood fill in Lab color space. "
        "Returns the seed point's color (hex and RGB) and the region's pixel count. "
        "Useful for determining the area of a specific colored block in a chart."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "seed_point": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "The [x, y] pixel coordinate of the seed point."
                },
                "blur_sigma": {
                    "type": "number",
                    "description": "Gaussian blur sigma for preprocessing. Larger values tolerate more noise. Default is 0.8."
                },
                "tolerance": {
                    "type": "number",
                    "description": "Per-channel tolerance in Lab color space for flood fill. Larger values merge more similar colors. Default is 25.0."
                },
                "connectivity": {
                    "type": "integer",
                    "description": "Pixel neighborhood connectivity: 4 (4-connected) or 8 (8-connected). Default is 4."
                }
            },
            "required": ["seed_point"]
        }

    def run(self, image: np.ndarray, seed_point: List[int],
            blur_sigma: float = 0.8, tolerance: float = 25.0,
            connectivity: int = 4, **kwargs) -> Dict[str, Any]:
        from .region_utils import preprocess, flood_fill_region

        seed_x, seed_y = int(seed_point[0]), int(seed_point[1])

        image_lab = preprocess(image, blur_sigma=blur_sigma)
        region_mask = flood_fill_region(image_lab, (seed_x, seed_y),
                                        tolerance=tolerance, connectivity=connectivity)

        bgr = image[seed_y, seed_x]
        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        pixel_count = int(np.count_nonzero(region_mask))

        return {
            "color_hex": color_hex,
            "color_rgb": [r, g, b],
            "pixel_count": pixel_count
        }
