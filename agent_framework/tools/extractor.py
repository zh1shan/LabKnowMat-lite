import numpy as np
import cv2
import sys
import os
from typing import Dict, Any, List, Optional
from PIL import Image, ImageEnhance
from .base import BaseAtomicTool

# We will lazily import torch and sam3 to avoid overhead if tools are not used
def _enhance_image(img: Image.Image, color: float = 1.20, contrast: float = 1.20) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    return img

class ColorPointExtractor(BaseAtomicTool):
    """
    Tool for extracting pixel coordinates of scatter points or line chart inflection points 
    based on a specific color.
    """
    name = "color_point_extractor"
    description = (
        "Extracts pixel coordinates of scattered points or line chart points matching "
        "a specific target color."
    )

    def run(self, image: np.ndarray, target_color: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Args:
            image (np.ndarray): The input chart image.
            target_color (str, optional): Target color in HEX format or color name.
        
        Returns:
            List[Dict[str, Any]]: List of points with their coordinates and colors.
                Example: [{"cx": 167, "cy": 203, "color": "#201955"}, ...]
        """
        # TODO: Implement color clustering and point detection logic
        return []

class SAM3BoxExtractor(BaseAtomicTool):
    """
    Tool for extracting bounding boxes of chart elements (like bars, pie slices) using SAM3 
    and text prompts.
    """
    name = "sam3_box_extractor"
    description = (
        "Uses SAM3 with text prompts (e.g., 'bar', 'pie slice') to extract bounding boxes "
        "of specific chart elements."
    )

    def __init__(self):
        self.processor = None

    def _init_model(self):
        if self.processor is None:
            # Import torch and sam3 here to avoid loading overhead when tool is not used
            import torch
            
            # Make sure sam3 can be imported. Assume it's installed or in PYTHONPATH
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sam3_repo = os.path.abspath(os.path.join(current_dir, "..", "..", "sam3"))
            if os.path.exists(sam3_repo) and sam3_repo not in sys.path:
                sys.path.insert(0, sam3_repo)

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)

    def run(self, image: np.ndarray, text_prompt: str = "chart bar", **kwargs) -> List[List[float]]:
        """
        Args:
            image (np.ndarray): The input chart image.
            text_prompt (str): Text prompt guiding the segmentation (default: "chart bar").
            
        Returns:
            List[List[float]]: List of bounding boxes [x_min, y_min, x_max, y_max].
        """
        self._init_model()
        
        # Convert numpy array (BGR or RGB depending on cv2, assume BGR from cv2) to PIL Image RGB
        if image.ndim == 3 and image.shape[2] == 3:
            # Assuming image is BGR from cv2
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            
        pil_img = Image.fromarray(img_rgb).convert("RGB")
        
        # Apply enhancement like the reference script
        enhanced_img = _enhance_image(pil_img, color=1.20, contrast=1.20)
        
        inference_state = self.processor.set_image(enhanced_img)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output.get("masks")
        if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
            return []
            
        import torch
        masks = masks.detach().cpu()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
            
        binary_masks = masks > 0.5
        
        boxes = []
        for idx in range(binary_masks.shape[0]):
            mask_bool = binary_masks[idx].numpy()
            if not np.any(mask_bool):
                continue
                
            ys, xs = np.nonzero(mask_bool)
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))
            boxes.append([x_min, y_min, x_max, y_max])
            
        return boxes

class PieSliceExtractor(BaseAtomicTool):
    """
    Tool for extracting individual slices of a pie chart, capturing their centroid, color, and polygon outline.
    """
    name = "pie_slice_extractor"
    description = (
        "Extracts pie chart slices. Identifies each slice's geometric centroid, primary color, "
        "and polygon outline. Useful for matching labels to pie segments."
    )
    
    def __init__(self):
        self.processor = None

    def _init_model(self):
        if self.processor is None:
            import torch
            
            # Make sure sam3 can be imported
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sam3_repo = os.path.abspath(os.path.join(current_dir, "..", "..", "sam3"))
            if os.path.exists(sam3_repo) and sam3_repo not in sys.path:
                sys.path.insert(0, sam3_repo)

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)

    def run(self, image: np.ndarray, target_color: Optional[str] = None, text_prompt: str = "pie chart slice", **kwargs) -> List[Dict[str, Any]]:
        """
        Args:
            image (np.ndarray): The input chart image.
            target_color (str, optional): A specific color to target a single slice. If None, extracts all slices.
            text_prompt (str): Text prompt guiding the segmentation (default: "pie chart slice").
            
        Returns:
            List[Dict[str, Any]]: List of slices with centroid, color, and polygon coordinates.
        """
        self._init_model()
        
        # Load utilities for pie processing
        from .pie_utils import (
            fill_holes_binary,
            estimate_circle_from_mask,
            build_label_map,
            angle_profile_labels,
            smooth_circular_labels,
            extract_boundaries_from_profile,
            build_slice_masks
        )
        
        if image.ndim == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            
        pil_img = Image.fromarray(img_rgb).convert("RGB")
        
        # Sam3-pie doesn't explicitly enhance image, but since it's the same pipeline, 
        # let's just pass original unless specified. Reference script pie2 doesn't use enhance_image.
        
        inference_state = self.processor.set_image(pil_img)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output.get("masks")
        if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
            return []
            
        import torch
        masks = masks.detach().cpu()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
            
        results = []
        for pie_idx in range(masks.shape[0]):
            pie_mask_bool = (masks[pie_idx].numpy() > 0.5)
            
            if not np.any(pie_mask_bool):
                continue
                
            mask_filled = fill_holes_binary(pie_mask_bool)
            
            try:
                circle = estimate_circle_from_mask(mask_filled)
            except ValueError:
                continue
                
            label_map, _, _ = build_label_map(
                image_rgb=img_rgb,
                mask=mask_filled,
                k=0,
                k_min=2,
                k_max=12,
                max_samples=20000,
                seed=0,
                preprocess="stretch",
                preprocess_alpha=1.5,
                preprocess_power=2.0,
                preprocess_clip=3.0,
            )
            
            profile = angle_profile_labels(
                label_map=label_map,
                circle=circle,
                n_angles=720,
            )
            profile_s = smooth_circular_labels(profile, window=7)
            boundaries = extract_boundaries_from_profile(profile_s, min_span_deg=3.0)
            
            slices = build_slice_masks(mask_filled, (circle.cx, circle.cy), boundaries)
            
            for slice_idx, slice_mask in enumerate(slices):
                if not np.any(slice_mask):
                    continue
                    
                ys, xs = np.nonzero(slice_mask)
                centroid = [float(np.mean(xs)), float(np.mean(ys))]
                
                # Extract main color
                pixels = img_rgb[ys, xs]
                # A simple way to find main color: calculate median
                median_color = np.median(pixels, axis=0).astype(int)
                hex_color = f"#{median_color[0]:02x}{median_color[1]:02x}{median_color[2]:02x}"
                
                # Extract polygon
                # cv2.findContours needs uint8 array
                mask_u8 = (slice_mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                polygon = []
                if contours:
                    # Take the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in largest_contour]
                
                # If target color is provided, optionally filter (this is a simple approximation)
                # For now, we return all slices and let the caller filter.
                
                results.append({
                    "centroid": centroid,
                    "color": hex_color,
                    "polygon": polygon
                })
                
        return results

class GridColorSampler(BaseAtomicTool):
    """
    Tool specifically for heatmaps to sample colors at the center of grid cells.
    """
    name = "grid_color_sampler"
    description = (
        "Samples the color at the center of each cell in a grid area, specifically for heatmaps."
    )

    def run(self, image: np.ndarray, bbox: List[float] = None, n_rows: int = 1, n_cols: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """
        Args:
            image (np.ndarray): The input chart image.
            bbox (List[float]): Bounding box of the grid [x_min, y_min, x_max, y_max].
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            
        Returns:
            List[Dict[str, Any]]: List of cell samples with row, col, and color.
                Example: [{"row": 0, "col": 1, "color": "#ff0000"}, ...]
        """
        if bbox is None:
            return []
            
        # TODO: Implement grid partitioning and center color sampling logic
        return []
