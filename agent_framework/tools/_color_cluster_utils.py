import numpy as np
import cv2
from skimage import color
from sklearn.cluster import KMeans

def extract_colors_density(image_rgb, n_colors, core_ratio=0.5, s_thresh=0.15, v_thresh=0.15, target_rect=None, text_mask=None):
    """
    Extract core colors and their physical coordinates using a density histogram method.
    
    Args:
    - image_rgb: Original RGB image (numpy array)
    - n_colors: The forced number of color clusters (K value)
    - core_ratio: The ratio of core pixels to total pixels in the cluster (e.g. 0.5 for top 50%)
    - s_thresh: Saturation threshold to filter out grays
    - v_thresh: Value threshold to filter out blacks
    - target_rect: Optional [x_min, y_min, x_max, y_max] to crop the working area
    
    Returns:
    - results: List of dictionaries containing cluster info, including mean RGB of core pixels and their (x, y) coordinates.
    """
    
    # 1. Crop if target_rect is provided
    if target_rect:
        x_min, y_min, x_max, y_max = target_rect
        h, w = image_rgb.shape[:2]
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
        working_img = image_rgb[y_min:y_max, x_min:x_max]
        offset_x, offset_y = x_min, y_min
    else:
        working_img = image_rgb
        offset_x, offset_y = 0, 0
        
    if working_img.size == 0:
        return []
        
    # 2. Filter out black, white, gray (in HSV space)
    img_hsv = color.rgb2hsv(working_img)
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]
    valid_mask = (S > s_thresh) & (V > v_thresh)
    
    if text_mask is not None:
        h_w, w_w = working_img.shape[:2]
        cropped = text_mask[offset_y:offset_y + h_w, offset_x:offset_x + w_w]
        valid_mask = valid_mask & (cropped == 0)
    
    # Get y, x coordinates of valid pixels within the working_img
    y_idx, x_idx = np.where(valid_mask)
    valid_pixels_rgb = working_img[y_idx, x_idx]
    
    if len(valid_pixels_rgb) == 0:
        return []
        
    # 3. Convert to LAB space
    valid_pixels_rgb_3d = valid_pixels_rgb.reshape(-1, 1, 3)
    valid_pixels_lab_3d = color.rgb2lab(valid_pixels_rgb_3d)
    valid_pixels_lab = valid_pixels_lab_3d.reshape(-1, 3)
    
    # 4. K-Means clustering in LAB space
    sample_size = min(20000, len(valid_pixels_lab))
    if len(valid_pixels_lab) > sample_size:
        indices = np.random.choice(len(valid_pixels_lab), sample_size, replace=False)
        train_data = valid_pixels_lab[indices]
    else:
        train_data = valid_pixels_lab
        
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(train_data)
    
    labels = kmeans.predict(valid_pixels_lab)
    
    results = []
    total_pixels = len(valid_pixels_lab)
    
    # 5. Density histogram method to extract core colors
    for cluster_idx in range(n_colors):
        cluster_mask = (labels == cluster_idx)
        cluster_pixels = valid_pixels_lab[cluster_mask]
        cluster_pixel_count = len(cluster_pixels)
        
        if cluster_pixel_count == 0:
            continue
            
        weight = cluster_pixel_count / total_pixels
        
        # Quantize LAB coords to integers
        quantized_pixels = np.round(cluster_pixels).astype(int)
        unique_coords, counts = np.unique(quantized_pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sort_indices = np.argsort(-counts)
        sorted_coords = unique_coords[sort_indices]
        sorted_counts = counts[sort_indices]
        
        target_count = cluster_pixel_count * core_ratio
        current_count = 0
        core_coords_set = set()
        
        for coord, count in zip(sorted_coords, sorted_counts):
            core_coords_set.add(tuple(coord))
            current_count += count
            if current_count >= target_count:
                break
                
        # Boolean array for core pixels within the current cluster
        is_core_list = [tuple(qp) in core_coords_set for qp in quantized_pixels]
        is_core_arr = np.array(is_core_list, dtype=bool)
        
        core_lab_pixels = cluster_pixels[is_core_arr]
        
        if len(core_lab_pixels) > 0:
            mean_lab = np.mean(core_lab_pixels, axis=0)
            mean_rgb = color.lab2rgb(mean_lab.reshape(1, 1, 3)) * 255
            mean_rgb = mean_rgb.reshape(3).clip(0, 255).astype(int).tolist()
        else:
            mean_lab = kmeans.cluster_centers_[cluster_idx]
            mean_rgb = color.lab2rgb(mean_lab.reshape(1, 1, 3)) * 255
            mean_rgb = mean_rgb.reshape(3).clip(0, 255).astype(int).tolist()
            
        # Map back to physical coordinates
        cluster_pixel_indices = np.where(cluster_mask)[0]
        core_pixel_indices = cluster_pixel_indices[is_core_arr]
        
        core_y = y_idx[core_pixel_indices] + offset_y
        core_x = x_idx[core_pixel_indices] + offset_x
        
        # Coordinates as [x, y]
        core_coords = np.column_stack((core_x, core_y)).tolist()
        
        results.append({
            'cluster_idx': cluster_idx,
            'weight': weight,
            'mean_rgb': mean_rgb,
            'core_count': len(core_coords),
            'core_coords': core_coords
        })
        
    results.sort(key=lambda x: x['weight'], reverse=True)
    return results
