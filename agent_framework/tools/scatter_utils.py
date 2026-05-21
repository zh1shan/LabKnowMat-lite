import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

@dataclass
class Cluster:
    id: int
    mean_lab: np.ndarray
    mean_rgb: np.ndarray
    count: int

    def update(self, lab: np.ndarray, rgb: np.ndarray) -> None:
        self.count += 1
        self.mean_lab = self.mean_lab + (lab - self.mean_lab) / self.count
        self.mean_rgb = self.mean_rgb + (rgb - self.mean_rgb) / self.count

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[..., ::-1]

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_1 = rgb.reshape(1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb_1, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return lab

def rgb_to_hex(rgb: np.ndarray) -> str:
    rgb_int = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb_int[0], rgb_int[1], rgb_int[2])

def build_candidate_mask(image_bgr: np.ndarray, s_thresh: int, v_thresh: int) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_color = (s > s_thresh) & (v > v_thresh)
    mask_color = mask_color.astype(np.uint8) * 255

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(gray, 21)
    diff = cv2.absdiff(gray, background)
    otsu_thresh, mask_shape = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_thresh < 8:
        _, mask_shape = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_shape = cv2.morphologyEx(mask_shape, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.bitwise_or(mask_color, mask_shape)

def extract_template_candidates(
    image_gray: np.ndarray,
    mask: np.ndarray,
    min_area: int,
) -> List[Tuple[int, int, int, int, float]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if min(w, h) <= 1:
            continue
        roi = image_gray[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        contrast = float(np.std(roi))
        components.append((x, y, w, h, contrast))

    if not components:
        return []

    areas = np.array([w * h for _, _, w, h, _ in components], dtype=np.float32)
    median_area = float(np.median(areas))
    min_keep = max(min_area, int(median_area * 0.3))
    max_keep = int(median_area * 5.0)

    filtered = []
    for x, y, w, h, contrast in components:
        area = w * h
        if area < min_keep or area > max_keep:
            continue
        filtered.append((x, y, w, h, contrast))

    return filtered

def group_candidates_by_size(candidates: List[Tuple[int, int, int, int, float]]) -> Dict[Tuple[int, int], List[int]]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, (_, _, w, h, _) in enumerate(candidates):
        key = (int(round(w / 2.0)) * 2, int(round(h / 2.0)) * 2)
        groups.setdefault(key, []).append(idx)
    return groups

def select_templates(
    image_gray: np.ndarray,
    candidates: List[Tuple[int, int, int, int, float]],
    max_templates: int,
) -> List[np.ndarray]:
    if not candidates:
        return []

    groups = group_candidates_by_size(candidates)
    ranked_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    templates = []
    for _, indices in ranked_groups[:max_templates]:
        best_idx = max(indices, key=lambda i: candidates[i][4])
        x, y, w, h, _ = candidates[best_idx]
        template = image_gray[y : y + h, x : x + w]
        if template.size == 0:
            continue
        templates.append(template)

    return templates

def local_maxima_nms(score_map: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    if score_map.size == 0:
        return []

    dilated = cv2.dilate(score_map, np.ones((3, 3), np.uint8))
    maxima = (score_map >= threshold) & (score_map == dilated)
    ys, xs = np.where(maxima)
    points = [(int(x), int(y), float(score_map[y, x])) for x, y in zip(xs, ys)]
    points.sort(key=lambda p: p[2], reverse=True)
    return points

def suppress_nearby(points: List[Tuple[int, int, float]], min_dist: float) -> List[Tuple[int, int, float]]:
    kept: List[Tuple[int, int, float]] = []
    min_dist_sq = float(min_dist * min_dist)
    for x, y, score in points:
        keep = True
        for kx, ky, _ in kept:
            dx = float(x - kx)
            dy = float(y - ky)
            if dx * dx + dy * dy <= min_dist_sq:
                keep = False
                break
        if keep:
            kept.append((x, y, score))
    return kept

def mean_color_in_rect(image_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.array([0, 0, 0], dtype=np.float32)
    roi = image_bgr[y0:y1, x0:x1]
    mean_bgr = cv2.mean(roi)[:3]
    return np.array(mean_bgr, dtype=np.float32)

def build_text_mask_from_ocr(image_shape: Tuple[int, int], ocr_results: List[Dict[str, Any]], expand: int) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in ocr_results:
        if 'box' not in item:
            continue
        box = np.array(item['box']).astype(np.int32)
        
        # Calculate bounding rect for the polygon box
        x, y, w, h = cv2.boundingRect(box)
        
        x0 = max(0, x - expand)
        y0 = max(0, y - expand)
        x1 = min(width, x + w + expand)
        y1 = min(height, y + h + expand)
        
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
            
    return mask

def detect_points_template(
    image_bgr: np.ndarray,
    match_threshold: float,
    max_templates: int,
    s_thresh: int,
    v_thresh: int,
    mask_ratio: float,
    text_mask: np.ndarray,
    target_rect: List[int] = None
) -> List[Tuple[int, int, np.ndarray]]:
    
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)

    candidate_mask = build_candidate_mask(image_bgr, s_thresh, v_thresh)
    
    if text_mask is not None:
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(text_mask))
        
    if target_rect is not None:
        rect_mask = np.zeros_like(candidate_mask)
        x_min, y_min, x_max, y_max = target_rect
        rect_mask[y_min:y_max, x_min:x_max] = 255
        candidate_mask = cv2.bitwise_and(candidate_mask, rect_mask)

    candidates = extract_template_candidates(image_gray, candidate_mask, min_area=3)
    templates = select_templates(image_gray, candidates, max_templates=max_templates)

    if not templates:
        return []

    detections: List[Tuple[int, int, float, int, int]] = []
    
    # If target_rect is used, we only care about matches inside it
    if target_rect is not None:
        rx_min, ry_min, rx_max, ry_max = target_rect
    else:
        rx_min, ry_min, rx_max, ry_max = 0, 0, image_gray.shape[1], image_gray.shape[0]

    for template in templates:
        th, tw = template.shape[:2]
        if th < 2 or tw < 2:
            continue
        if th >= image_gray.shape[0] or tw >= image_gray.shape[1]:
            continue

        score_map = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        peaks = local_maxima_nms(score_map, match_threshold)

        min_dist = max(2.0, min(th, tw) * 0.6)
        peaks = suppress_nearby(peaks, min_dist)

        for px, py, score in peaks:
            # Check if center of template match is inside target_rect
            cx = int(round(px + tw / 2.0))
            cy = int(round(py + th / 2.0))
            if rx_min <= cx <= rx_max and ry_min <= cy <= ry_max:
                detections.append((px, py, score, tw, th))

    if not detections:
        return []

    detections.sort(key=lambda d: d[2], reverse=True)
    merged: List[Tuple[int, int, float, int, int]] = []
    for x, y, score, tw, th in detections:
        cx = int(round(x + tw / 2.0))
        cy = int(round(y + th / 2.0))
        min_dist = max(2.0, min(tw, th) * 0.6)
        too_close = False
        for mx, my, _, mw, mh in merged:
            dx = float(cx - mx)
            dy = float(cy - my)
            if dx * dx + dy * dy <= float(min_dist * min_dist):
                too_close = True
                break
        if too_close:
            continue
        merged.append((cx, cy, score, tw, th))

    points: List[Tuple[int, int, np.ndarray]] = []
    for cx, cy, _, tw, th in merged:
        if text_mask is not None:
            if 0 <= cy < text_mask.shape[0] and 0 <= cx < text_mask.shape[1]:
                if text_mask[cy, cx] > 0:
                    continue
                    
        x0 = int(round(cx - tw / 2.0))
        y0 = int(round(cy - th / 2.0))
        
        roi_mask = candidate_mask[max(0, y0) : min(candidate_mask.shape[0], y0 + th), max(0, x0) : min(candidate_mask.shape[1], x0 + tw)]
        if roi_mask.size == 0:
            continue
        cover = float(cv2.countNonZero(roi_mask)) / float(roi_mask.size)
        if cover < mask_ratio:
            continue
            
        mean_bgr = mean_color_in_rect(image_bgr, x0, y0, tw, th)
        mean_rgb = bgr_to_rgb(mean_bgr)
        points.append((int(cx), int(cy), mean_rgb))

    return points

def cluster_colors(points: List[Tuple[int, int, np.ndarray]], threshold: float) -> Tuple[List[Cluster], List[int]]:
    clusters: List[Cluster] = []
    assignments: List[int] = []

    for _, _, rgb in points:
        lab = rgb_to_lab(rgb)
        best_idx = -1
        best_dist = float("inf")
        for cluster in clusters:
            dist = float(np.linalg.norm(lab - cluster.mean_lab))
            if dist < best_dist:
                best_dist = dist
                best_idx = cluster.id
        if best_idx != -1 and best_dist <= threshold:
            cluster = clusters[best_idx]
            cluster.update(lab, rgb)
            assignments.append(cluster.id)
        else:
            new_id = len(clusters)
            clusters.append(Cluster(id=new_id, mean_lab=lab, mean_rgb=rgb.astype(np.float32), count=1))
            assignments.append(new_id)

    return clusters, assignments