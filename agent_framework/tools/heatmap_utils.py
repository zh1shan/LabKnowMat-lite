import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from scipy.interpolate import griddata

def _saturation_mask(bgr: np.ndarray, sat_thresh: int, val_thresh: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s >= sat_thresh) & (v >= val_thresh)
    return (mask.astype(np.uint8) * 255)

def _find_rectangles(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, list]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        rects.append((x, y, w, h))
    return clean, rects

def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1

def _suppress_overlaps(rects: list) -> list:
    rects_sorted = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for rect in rects_sorted:
        if any(_rects_overlap(rect, other) for other in kept):
            continue
        kept.append(rect)
    return kept

def detect_regions(bgr: np.ndarray, sat_thresh: int = 55, val_thresh: int = 80, min_area: int = 300) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], str]:
    """
    Detects the heatmap data region and the legend region.
    Returns (heatmap_rect, legend_rect, legend_orientation).
    rects are in (x, y, w, h) format.
    """
    mask = _saturation_mask(bgr, sat_thresh, val_thresh)
    _, rects = _find_rectangles(mask, min_area)
    kept_rects = _suppress_overlaps(rects)
    
    if len(kept_rects) != 2:
        raise ValueError(f"Expected 2 rectangles, got {len(kept_rects)}")
        
    def _aspect_ratio(r):
        w, h = r[2], r[3]
        if w == 0 or h == 0: return 0.0
        return max(w, h) / min(w, h)
        
    ratios = [_aspect_ratio(r) for r in kept_rects]
    
    legend_index = 0 if ratios[0] >= ratios[1] else 1
    heatmap_index = 1 - legend_index
    
    heatmap_rect = kept_rects[heatmap_index]
    legend_rect = kept_rects[legend_index]
    
    lw, lh = legend_rect[2], legend_rect[3]
    legend_orientation = "horizontal" if lw >= lh else "vertical"
    
    return heatmap_rect, legend_rect, legend_orientation

def extract_heatmap_data(image: np.ndarray, heatmap_rect: Tuple[int, int, int, int], legend_rect: Tuple[int, int, int, int], legend_orientation: str, output_size: Tuple[int, int] = (128, 128)) -> Dict[str, Any]:
    """
    Extracts the color scale from the legend and maps the heatmap region to a normalized matrix of size `output_size`.
    """
    hx, hy, hw, hh = heatmap_rect
    lx, ly, lw, lh = legend_rect
    
    # Crop regions
    heatmap_img = image[hy:hy+hh, hx:hx+hw]
    legend_img = image[ly:ly+lh, lx:lx+lw]
    
    # 1. Extract color scale from legend
    if legend_orientation == "horizontal":
        line = legend_img.mean(axis=0) # shape (lw, 3)
        # Horizontal legend goes from left (0) to right (1)
        color_scale_y_values = np.linspace(0, 1, lw)
    else:
        line = legend_img.mean(axis=1) # shape (lh, 3)
        # Vertical legend goes from bottom (0) to top (1) in the original script.
        # Image coordinates: top is 0, bottom is lh. So we go from 1 to 0.
        color_scale_y_values = np.linspace(1, 0, lh)
        
    color_scale_image = line.reshape((-1, 3)).astype(np.uint8) # shape (N, 3)
    
    # 2. Resize heatmap image to output_size (128x128) using Nearest Neighbor
    heatmap_resized = cv2.resize(heatmap_img, output_size, interpolation=cv2.INTER_NEAREST)
    
    # 3. Interpolate using griddata or KDTree
    # The original uses scipy griddata on a 256x256x256 grid.
    # To be exactly like the original, we can do nearest neighbor interpolation 
    # directly from the color scale to the heatmap pixels to save memory and time.
    # Actually, KDTree is much faster and cleaner for finding nearest color in RGB space.
    from scipy.spatial import KDTree
    
    # Build KDTree from the legend colors
    tree = KDTree(color_scale_image)
    
    # Flatten the resized heatmap to (N, 3)
    flat_heatmap = heatmap_resized.reshape((-1, 3))
    
    # Query nearest colors
    distances, indices = tree.query(flat_heatmap)
    
    # Map indices to values
    normalized_flat = color_scale_y_values[indices]
    
    # Reshape back to (128, 128)
    normalized_matrix = normalized_flat.reshape(output_size[1], output_size[0])
    
    # Prepare color mapping info (sample a few points for description if needed)
    color_map_info = []
    # Take 10 evenly spaced points from the legend
    idx_samples = np.linspace(0, len(color_scale_image)-1, 10).astype(int)
    for i in idx_samples:
        color = color_scale_image[i]
        val = color_scale_y_values[i]
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}" # Convert BGR to RGB hex
        color_map_info.append({"color": hex_color, "value": float(val)})
    
    return {
        "heatmap_rect": [hx, hy, hx+hw, hy+hh], # convert to [x_min, y_min, x_max, y_max]
        "legend_rect": [lx, ly, lx+lw, ly+lh],
        "legend_orientation": legend_orientation,
        "color_to_value_map": color_map_info,
        "normalized_matrix": normalized_matrix.tolist()
    }