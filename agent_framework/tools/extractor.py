import numpy as np
from typing import Dict, Any, List, Optional, Sequence, Tuple
from PIL import Image
import torch
from .base import BaseAtomicTool

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
        return []

class SAM3BoxExtractor(BaseAtomicTool):
    """
    Tool for extracting bounding boxes of chart elements (like bars) using SAM3 
    and text prompts.
    """
    name = "sam3_box_extractor"
    description = (
        "Uses SAM3 with text prompts (e.g., 'chart bar') to extract bounding boxes "
        "of specific chart elements."
    )

    def _initialize_sam3(self):
        if not hasattr(self, 'processor') or self.processor is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                raise ImportError("SAM3 is not installed. Please read SAM3_SETUP.md for instructions.")
            
            # To ensure compatibility with CUDA/CPU, let SAM3 handle device placement or force cuda if available
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)

    def run(self, image: np.ndarray, text_prompt: str = "chart bar", **kwargs) -> List[List[float]]:
        self._initialize_sam3()
        
        # Convert OpenCV BGR image to RGB PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assuming BGR input from cv2
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output.get("masks")
        if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
            return []
            
        masks = masks.detach().cpu()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
            
        binary_masks = masks > 0.5
        
        boxes = []
        for idx in range(binary_masks.shape[0]):
            mask_bool = binary_masks[idx].numpy()
            ys, xs = np.nonzero(mask_bool)
            if len(xs) == 0:
                continue
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            boxes.append([x_min, y_min, x_max, y_max])
            
        return boxes


def _build_slice_intervals(boundaries_deg: Sequence[float]) -> List[Tuple[float, float]]:
    if not boundaries_deg or len(boundaries_deg) < 2:
        return [(0.0, 360.0)]
    edges = sorted(set(float(b) % 360.0 for b in boundaries_deg))
    if len(edges) < 2:
        return [(0.0, 360.0)]
    intervals = []
    n = len(edges)
    for i in range(n):
        start = edges[i]
        end = edges[(i + 1) % n]
        if start == end:
            continue
        intervals.append((start, end))
    return intervals if intervals else [(0.0, 360.0)]

def _angle_in_interval(angles: np.ndarray, start: float, end: float) -> np.ndarray:
    if start < end:
        return (angles >= start) & (angles < end)
    return (angles >= start) | (angles < end)

def _build_slice_masks(
    mask_filled: np.ndarray,
    circle_center: Tuple[float, float],
    boundaries_deg: Sequence[float],
) -> List[np.ndarray]:
    h, w = mask_filled.shape
    cy, cx = float(circle_center[1]), float(circle_center[0])

    ys, xs = np.indices((h, w))
    angles = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0

    intervals = _build_slice_intervals(boundaries_deg)
    slices = []
    for start, end in intervals:
        slice_mask = mask_filled & _angle_in_interval(angles, start, end)
        if np.any(slice_mask):
            slices.append(slice_mask)
    return slices


class PieSliceExtractor(BaseAtomicTool):
    """
    Tool for extracting individual slices of a pie chart, capturing their centroid, color, and polygon outline.
    """
    name = "pie_slice_extractor"
    description = (
        "Extracts pie chart slices. Identifies each slice's geometric centroid, primary color, "
        "and polygon outline. Useful for matching labels to pie segments."
    )

    def _initialize_sam3(self):
        if not hasattr(self, 'processor') or self.processor is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                raise ImportError("SAM3 is not installed. Please read SAM3_SETUP.md for instructions.")
            
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)

    def run(self, image: np.ndarray, target_color: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        self._initialize_sam3()
        
        try:
            from .pie_slice_boundaries import (
                angle_profile_labels,
                build_label_map,
                estimate_circle_from_mask,
                extract_boundaries_from_profile,
                fill_holes_binary,
                smooth_circular_labels,
            )
        except ImportError:
            raise ImportError("pie_slice_boundaries.py is missing from agent_framework/tools.")
        
        import cv2 # used for contour extraction
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt="pie chart slice")
        
        masks = output.get("masks")
        if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
            return []
            
        results = []
            
        for pie_idx in range(masks.shape[0]):
            mask_tensor = masks[pie_idx].detach().cpu()
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze(0)
            pie_mask_bool = mask_tensor.numpy() > 0.5
            
            if not np.any(pie_mask_bool):
                continue
                
            mask_filled = fill_holes_binary(pie_mask_bool)
            
            try:
                circle = estimate_circle_from_mask(mask_filled)
            except ValueError:
                continue
                
            label_map, _, _ = build_label_map(
                image_rgb=image_rgb,
                mask=mask_filled,
                k=0,
                k_min=2,
                k_max=12,
                max_samples=20000,
                seed=0,
            )

            profile = angle_profile_labels(
                label_map=label_map,
                circle=circle,
                n_angles=720,
            )
            profile_s = smooth_circular_labels(profile, window=7)
            boundaries = extract_boundaries_from_profile(profile_s, min_span_deg=3.0)

            slices = _build_slice_masks(mask_filled, (circle.cx, circle.cy), boundaries)
            
            for slice_idx, slice_mask in enumerate(slices):
                # Calculate centroid
                ys, xs = np.nonzero(slice_mask)
                if len(xs) == 0:
                    continue
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                
                # Get primary color inside slice
                # We can sample the median or mean color of the slice
                slice_pixels = image_rgb[ys, xs]
                median_color = np.median(slice_pixels, axis=0)
                hex_color = "#{:02x}{:02x}{:02x}".format(int(median_color[0]), int(median_color[1]), int(median_color[2]))
                
                # Extract polygon
                mask_u8 = (slice_mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Take the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = largest_contour.squeeze().tolist()
                    if not isinstance(polygon[0], list):
                        polygon = [polygon] # handle rare shape case
                else:
                    polygon = []
                    
                results.append({
                    "centroid": [cx, cy],
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
        if bbox is None:
            return []
        return []
