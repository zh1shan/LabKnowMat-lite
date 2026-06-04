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

class ScatterPointExtractorV1(BaseAtomicTool):
    """
    Tool for detecting scatter points in a chart, extracting their coordinates, 
    and clustering them by color (V1 using Template Matching + OCR filtering).
    """
    name = "scatter_point_extractor_v1"
    description = (
        "Detects scatter points using template matching and color masking. "
        "Automatically filters out text using OCR, and clusters the detected points by color."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_rect": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [x_min, y_min, x_max, y_max] to restrict detection area."
                },
                "match_threshold": {
                    "type": "number",
                    "description": "NCC template matching threshold. Default is 0.75."
                },
                "color_cluster": {
                    "type": "number",
                    "description": "Lab space distance threshold for color clustering. Default is 15.0."
                }
            },
            "required": []
        }

    def run(self, image: np.ndarray, target_rect: Optional[List[int]] = None, 
            match_threshold: float = 0.75, color_cluster: float = 15.0, 
            max_templates: int = 3, s_thresh: int = 30, v_thresh: int = 30, 
            mask_ratio: float = 0.1, ocr_expand: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (np.ndarray): The input chart image (BGR format from cv2).
            target_rect (List[int], optional): Optional [x_min, y_min, x_max, y_max] to restrict detection area.
            match_threshold (float): NCC template matching threshold.
            color_cluster (float): Lab space distance threshold for color clustering.
            
        Returns:
            Dict[str, Any]: Contains 'point_count', 'cluster_count', 'points', and 'clusters'.
        """
        from .scatter_utils import build_text_mask_from_ocr, detect_points_template, cluster_colors, rgb_to_hex
        from .ocr import OCRTextLocator
        
        # 1. Run OCR to build text mask
        ocr_tool = OCRTextLocator()
        ocr_results = ocr_tool.run(image)
        text_mask = build_text_mask_from_ocr(image.shape, ocr_results, ocr_expand)
        
        # 2. Detect points using template matching
        points = detect_points_template(
            image_bgr=image,
            match_threshold=match_threshold,
            max_templates=max_templates,
            s_thresh=s_thresh,
            v_thresh=v_thresh,
            mask_ratio=mask_ratio,
            text_mask=text_mask,
            target_rect=target_rect
        )
        
        # 3. Cluster colors
        clusters, assignments = cluster_colors(points, color_cluster)
        
        # 4. Format output
        points_json = []
        for (x, y, rgb), cluster_id in zip(points, assignments):
            points_json.append({
                "x": int(x),
                "y": int(y),
                "cluster_id": int(cluster_id),
            })
            
        clusters_json = []
        for cluster in clusters:
            mean_rgb = np.clip(np.round(cluster.mean_rgb), 0, 255)
            clusters_json.append({
                "id": cluster.id,
                "mean_rgb": [int(v) for v in mean_rgb],
                "mean_hex": rgb_to_hex(mean_rgb),
                "count": cluster.count,
            })
            
        return {
            "point_count": len(points_json),
            "cluster_count": len(clusters_json),
            "points": points_json,
            "clusters": clusters_json
        }

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

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text_prompt": {
                    "type": "string",
                    "description": "Text prompt guiding the segmentation (default: 'chart bar')."
                }
            },
            "required": []
        }

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
    Tool for extracting individual slices of a pie chart, capturing their centroid, color, and pixel count.
    """
    name = "pie_slice_extractor"
    description = (
        "Extracts pie chart slices. Identifies each slice's geometric centroid, primary color, "
        "and pixel count. Useful for matching labels to pie segments and calculating proportions."
    )
    
    def __init__(self):
        self.processor = None

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_color": {
                    "type": "string",
                    "description": "A specific color in HEX format or color name to target a single slice. If None, extracts all slices."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Text prompt guiding the segmentation (default: 'pie chart slice')."
                }
            },
            "required": []
        }

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
            List[Dict[str, Any]]: List of slices with centroid, color, and pixel count.
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
                pixel_count = len(xs)
                
                # Extract main color
                pixels = img_rgb[ys, xs]
                # A simple way to find main color: calculate median
                median_color = np.median(pixels, axis=0).astype(int)
                hex_color = f"#{median_color[0]:02x}{median_color[1]:02x}{median_color[2]:02x}"
                
                results.append({
                    "centroid": centroid,
                    "color": hex_color,
                    "pixel_count": pixel_count
                })
                
        return results

class HeatmapDigitizerTool(BaseAtomicTool):
    """
    Tool for automatically detecting heatmap and legend regions, and digitizing the heatmap 
    into a normalized 128x128 numerical matrix.
    """
    name = "heatmap_digitizer"
    description = (
        "Automatically detects heatmap data and legend regions. "
        "Returns their bounding boxes, the legend orientation, a color-to-value mapping, "
        "and a 128x128 normalized numerical matrix representing the heatmap data."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sat_thresh": {
                    "type": "integer",
                    "description": "Saturation threshold for region detection. Default is 55."
                },
                "val_thresh": {
                    "type": "integer",
                    "description": "Value threshold for region detection. Default is 80."
                },
                "min_area": {
                    "type": "integer",
                    "description": "Minimum area for valid rectangles. Default is 300."
                }
            },
            "required": []
        }

    def run(self, image: np.ndarray, sat_thresh: int = 55, val_thresh: int = 80, min_area: int = 300, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (np.ndarray): The input chart image.
            sat_thresh (int): Saturation threshold for region detection.
            val_thresh (int): Value threshold for region detection.
            min_area (int): Minimum area for valid rectangles.
            
        Returns:
            Dict[str, Any]: Contains 'heatmap_rect', 'legend_rect', 'legend_orientation',
                            'color_to_value_map', and 'normalized_matrix'.
        """
        from .heatmap_utils import detect_regions, extract_heatmap_data
        
        try:
            heatmap_rect, legend_rect, legend_orientation = detect_regions(
                image, sat_thresh=sat_thresh, val_thresh=val_thresh, min_area=min_area
            )
        except ValueError as e:
            return {"error": str(e)}
            
        result = extract_heatmap_data(
            image, heatmap_rect, legend_rect, legend_orientation, output_size=(128, 128)
        )
        
        return result
