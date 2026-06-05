import cv2
import numpy as np

BLUR_SIGMA = 0.8
TOLERANCE = 25.0
CONNECTIVITY = 4


def preprocess(image_bgr, blur_sigma=BLUR_SIGMA):
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    ksize = int(blur_sigma * 6) | 1
    image_lab = cv2.GaussianBlur(image_lab, (ksize, ksize), blur_sigma)
    return image_lab


def flood_fill_region(image_lab, seed_point, tolerance=TOLERANCE, connectivity=CONNECTIVITY):
    h, w = image_lab.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = connectivity | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

    lo_diff = (tolerance, tolerance, tolerance, 0)
    up_diff = (tolerance, tolerance, tolerance, 0)

    cv2.floodFill(image_lab.copy(), mask, seed_point, 0, lo_diff, up_diff, flags)

    region_mask = mask[1:-1, 1:-1]
    return region_mask > 0


def extract_edge(region_mask):
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(region_mask.astype(np.uint8) * 255, kernel)
    edge = region_mask.astype(np.uint8) * 255 - eroded
    return edge > 0
