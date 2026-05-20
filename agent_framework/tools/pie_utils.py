#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pie chart slice boundary extraction from image + coarse circular mask.

Goal
- Input: an image and a binary-ish mask (rough circle with small holes).
- Fill holes in the mask (treat as solid disk).
- Inside the disk, segment by color distribution (k-means in Lab).
- Extract slice boundaries as angular transition lines from the disk center.
- Output: image overlaid with slice boundary lines.

This script intentionally keeps dependencies light (Pillow + numpy).
Optional speed/quality improvements are used when scipy is available.

Example
  python pie_slice_boundaries.py \
    --image /path/to/img.jpg \
    --mask  /path/to/mask.png \
    --out   /path/to/out.png \
        --k auto --k-max 10 --debug-dir /tmp/pie_debug

"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


def _ensure_dir(path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_mask_bool(path: str, threshold: int = 127) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    return arr > threshold


def save_debug_mask(mask: np.ndarray, path: str) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)


def fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask.

    Uses scipy.ndimage.binary_fill_holes when available; otherwise uses a
    border flood-fill on the background and inverts.
    """

    mask = mask.astype(bool)

    # Fast path if scipy is available.
    try:
        from scipy.ndimage import binary_fill_holes  # type: ignore

        return binary_fill_holes(mask)
    except Exception:
        pass

    # Flood fill on background connected to the padded border.
    h, w = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    bg = ~padded

    visited = np.zeros_like(bg, dtype=bool)
    q: deque[Tuple[int, int]] = deque()

    # Seed all border pixels that are background.
    H, W = bg.shape
    for x in range(W):
        if bg[0, x]:
            q.append((0, x))
            visited[0, x] = True
        if bg[H - 1, x] and not visited[H - 1, x]:
            q.append((H - 1, x))
            visited[H - 1, x] = True
    for y in range(H):
        if bg[y, 0] and not visited[y, 0]:
            q.append((y, 0))
            visited[y, 0] = True
        if bg[y, W - 1] and not visited[y, W - 1]:
            q.append((y, W - 1))
            visited[y, W - 1] = True

    # 4-connected fill.
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and bg[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # visited == background connected to border
    holes = bg & (~visited)
    filled = padded | holes
    return filled[1 : h + 1, 1 : w + 1]


def estimate_circle_from_mask(mask: np.ndarray) -> Circle:
    """Estimate circle center/radius from (filled) disk-like mask."""

    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        raise ValueError("Mask contains too few foreground pixels")

    cx = float(xs.mean())
    cy = float(ys.mean())

    area = float(mask.sum())
    r_area = math.sqrt(max(area, 1.0) / math.pi)

    # Also estimate radius from max distance to center; clamp to area-based.
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    r_max = float(np.sqrt(d2.max()))

    # Robust-ish: take a weighted blend to avoid spikes.
    r = 0.7 * r_area + 0.3 * r_max
    return Circle(cx=cx, cy=cy, r=r)


def rgb_to_lab_u8(rgb: np.ndarray) -> np.ndarray:
    """Approximate RGB->Lab conversion.

    Uses a standard sRGB->XYZ->Lab conversion with D65 white.
    Returns float32 Lab with L in [0,100], a/b roughly [-128,127].

    Note: Implemented to avoid OpenCV dependency.
    """

    rgb = rgb.astype(np.float32) / 255.0

    def inv_gamma(u: np.ndarray) -> np.ndarray:
        return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

    r, g, b = inv_gamma(rgb[..., 0]), inv_gamma(rgb[..., 1]), inv_gamma(rgb[..., 2])

    # sRGB to XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by reference white D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x /= Xn
    y /= Yn
    z /= Zn

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b2 = 200.0 * (fy - fz)

    lab = np.stack([L, a, b2], axis=-1).astype(np.float32)
    return lab


def preprocess_lab_for_kmeans(
    lab: np.ndarray,
    mask: np.ndarray,
    method: str = "stretch",
    alpha: float = 1.5,
    power: float = 2.0,
    clip: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Non-linear preprocessing for Lab before clustering.

    Goal:
    - Keep close samples nearly unchanged.
    - Increase distances for moderately far samples.
    - Avoid overflow/boundary distortion via clipping.

    method:
      - "none": no preprocessing
      - "stretch": non-linear radial stretch in standardized space
    """

    method = (method or "none").strip().lower()
    if method in ("none", "off", "false", "0"):
        return lab.astype(np.float32, copy=False)

    if method != "stretch":
        raise ValueError(f"Unknown preprocess method: {method}")

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if power <= 0:
        raise ValueError("power must be positive")
    if clip <= 0:
        raise ValueError("clip must be positive")

    lab_f = lab.astype(np.float32, copy=False)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return lab_f

    pts = lab_f[ys, xs].reshape(-1, 3)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    std = np.maximum(std, eps)

    z = (lab_f - mean) / std
    z = np.clip(z, -clip, clip)

    # Non-linear stretch: scale grows with |z|.
    # scale = 1 + alpha * (|z|/clip)^power
    abs_z = np.abs(z)
    scale = 1.0 + alpha * (abs_z / clip) ** power
    z2 = z * scale

    max_out = clip * (1.0 + alpha)
    z2 = np.clip(z2, -max_out, max_out)

    return z2.astype(np.float32, copy=False)


def kmeans(
    data: np.ndarray,
    k: int,
    n_iter: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means on float32 array data of shape (N, D).

    Returns (centers, labels).
    """

    if k <= 0:
        raise ValueError("k must be positive")

    rng = np.random.default_rng(seed)
    n, d = data.shape
    if n < k:
        raise ValueError(f"Not enough samples for k={k} (n={n})")

    # kmeans++ init (lightweight)
    centers = np.empty((k, d), dtype=np.float32)
    idx0 = int(rng.integers(0, n))
    centers[0] = data[idx0]

    closest_d2 = np.sum((data - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_d2 / max(closest_d2.sum(), 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[i] = data[idx]
        d2 = np.sum((data - centers[i]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2)

    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(n_iter):
        # Assign
        d2 = (
            np.sum(data**2, axis=1, keepdims=True)
            - 2.0 * (data @ centers.T)
            + np.sum(centers**2, axis=1, keepdims=True).T
        )
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centers[j] = data[int(rng.integers(0, n))]
            else:
                centers[j] = data[mask].mean(axis=0)

    return centers, labels


def davies_bouldin_index(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> float:
    """Compute Davies–Bouldin index (lower is better).

    data: (N,D) float
    labels: (N,) int in [0,k)
    centers: (k,D)
    """

    k = int(centers.shape[0])
    if k < 2:
        return float("inf")

    scatters = np.zeros((k,), dtype=np.float32)
    for i in range(k):
        pts = data[labels == i]
        if pts.size == 0:
            scatters[i] = 0.0
            continue
        d = np.linalg.norm(pts - centers[i], axis=1)
        scatters[i] = float(d.mean())

    # Pairwise center distances
    cd = centers.astype(np.float32)
    diff = cd[:, None, :] - cd[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 1e-12)

    # R_ij = (s_i + s_j) / d(c_i, c_j), i!=j
    s_sum = scatters[:, None] + scatters[None, :]
    r = s_sum / dist
    np.fill_diagonal(r, -np.inf)

    d_i = np.max(r, axis=1)
    return float(np.mean(d_i))


def choose_k_auto(
    sample: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    n_iter: int = 30,
) -> Tuple[int, np.ndarray, dict]:
    """Choose k by minimizing Davies–Bouldin index over a range."""

    n = int(sample.shape[0])
    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    k_max = min(k_max, n)  # cannot exceed sample size

    best_k = k_min
    best_centers = None
    best_score = float("inf")
    scores = {}

    data = sample.astype(np.float32, copy=False)
    for k in range(k_min, k_max + 1):
        centers, labels = kmeans(data, k=k, n_iter=n_iter, seed=seed)
        score = davies_bouldin_index(data, labels, centers)
        scores[str(k)] = score
        if score < best_score:
            best_score = score
            best_k = k
            best_centers = centers

    assert best_centers is not None
    return best_k, best_centers, {"metric": "davies_bouldin", "scores": scores, "best": best_k}


def build_label_map(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    k: int,
    k_min: int,
    k_max: int,
    max_samples: int,
    seed: int,
    preprocess: str = "stretch",
    preprocess_alpha: float = 1.5,
    preprocess_power: float = 2.0,
    preprocess_clip: float = 3.0,
) -> Tuple[np.ndarray, int, Optional[dict]]:
    """Cluster colors within mask and return (label_map, chosen_k, k_debug).

    If k <= 0, automatically chooses k in [k_min, k_max].
    """

    h, w, _ = image_rgb.shape
    lab = rgb_to_lab_u8(image_rgb)
    lab = preprocess_lab_for_kmeans(
        lab=lab,
        mask=mask,
        method=preprocess,
        alpha=preprocess_alpha,
        power=preprocess_power,
        clip=preprocess_clip,
    )

    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n == 0:
        raise ValueError("Empty mask")

    # Sample for kmeans fitting.
    rng = np.random.default_rng(seed)
    if n > max_samples:
        sel = rng.choice(n, size=max_samples, replace=False)
        sample = lab[ys[sel], xs[sel]]
    else:
        sample = lab[ys, xs]

    sample_f = sample.reshape(-1, 3).astype(np.float32)
    k_debug: Optional[dict] = None
    chosen_k = int(k)
    if chosen_k <= 0:
        chosen_k, centers, k_debug = choose_k_auto(
            sample=sample_f,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
        )
    else:
        centers, _ = kmeans(sample_f, k=chosen_k, seed=seed)

    # Assign all masked pixels to nearest center.
    pixels = lab[ys, xs].reshape(-1, 3).astype(np.float32)
    d2 = (
        np.sum(pixels**2, axis=1, keepdims=True)
        - 2.0 * (pixels @ centers.T)
        + np.sum(centers**2, axis=1, keepdims=True).T
    )
    labels = np.argmin(d2, axis=1).astype(np.int16)

    label_map = np.full((h, w), -1, dtype=np.int16)
    label_map[ys, xs] = labels
    return label_map, chosen_k, k_debug


def _mode_ignore_neg(values: np.ndarray) -> int:
    values = values[values >= 0]
    if values.size == 0:
        return -1
    c = Counter(values.tolist())
    return int(c.most_common(1)[0][0])


def angle_profile_labels(
    label_map: np.ndarray,
    circle: Circle,
    n_angles: int = 720,
    r_start: float = 0.12,
    r_end: float = 0.98,
    n_r_samples: int = 80,
) -> np.ndarray:
    """For each angle, vote the dominant label along the radius."""

    h, w = label_map.shape
    cx, cy, r = circle.cx, circle.cy, circle.r

    angles = np.linspace(0.0, 2 * math.pi, num=n_angles, endpoint=False)
    rs = np.linspace(r * r_start, r * r_end, num=n_r_samples)

    out = np.full((n_angles,), -1, dtype=np.int16)

    for i, theta in enumerate(angles):
        xs = cx + rs * math.cos(theta)
        ys = cy + rs * math.sin(theta)
        xi = np.clip(xs.round().astype(np.int32), 0, w - 1)
        yi = np.clip(ys.round().astype(np.int32), 0, h - 1)
        vals = label_map[yi, xi]
        out[i] = _mode_ignore_neg(vals)

    # Fill isolated -1 by nearest neighbor circularly.
    if np.any(out < 0):
        idx = np.arange(n_angles)
        valid = out >= 0
        if np.any(valid):
            valid_idx = idx[valid]
            valid_vals = out[valid]
            for i in idx[~valid]:
                # circular distance
                d = np.minimum((valid_idx - i) % n_angles, (i - valid_idx) % n_angles)
                out[i] = valid_vals[int(np.argmin(d))]

    return out


def smooth_circular_labels(labels: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return labels.copy()
    n = len(labels)
    pad = window // 2
    extended = np.concatenate([labels[-pad:], labels, labels[:pad]])
    smoothed = labels.copy()
    for i in range(n):
        chunk = extended[i : i + window]
        smoothed[i] = _mode_ignore_neg(chunk.astype(np.int32))
    return smoothed


def extract_boundaries_from_profile(
    labels: np.ndarray,
    min_span_deg: float,
) -> List[float]:
    """Return boundary angles (degrees) where label changes.

    labels: length N circular profile.
    """

    n = len(labels)
    if n == 0:
        return []

    def circular_segments(arr: np.ndarray) -> List[Tuple[int, int, int]]:
        """Return circular RLE segments as (start_idx, length, label)."""

        change = arr != np.roll(arr, 1)
        starts = np.where(change)[0]
        if starts.size == 0:
            return [(0, n, int(arr[0]))]

        out: List[Tuple[int, int, int]] = []
        m = int(starts.size)
        for i in range(m):
            s = int(starts[i])
            e = int(starts[(i + 1) % m])
            length = (e - s) % n
            if length == 0:
                length = n
            out.append((s, int(length), int(arr[s])))
        return out

    # Remove tiny segments by relabeling them into a neighbor label.
    min_span = max(1, int(round((min_span_deg / 360.0) * n)))
    work = labels.astype(np.int32, copy=True)

    while True:
        segs = circular_segments(work)
        if len(segs) <= 1:
            return []

        lengths = [ln for _, ln, _ in segs]
        j = int(np.argmin(lengths))
        if lengths[j] >= min_span:
            break

        # Avoid collapsing to a single segment; keep at least 2.
        if len(segs) <= 2:
            break

        prev_i = (j - 1) % len(segs)
        next_i = (j + 1) % len(segs)
        prev_len = segs[prev_i][1]
        next_len = segs[next_i][1]
        target_label = segs[prev_i][2] if prev_len >= next_len else segs[next_i][2]

        s, ln, _ = segs[j]
        idx = (s + np.arange(ln, dtype=np.int32)) % n
        work[idx] = target_label

    boundaries = [(s % n) * (360.0 / n) for s, _, _ in circular_segments(work)]
    boundaries = sorted(set([round(b, 3) for b in boundaries]))
    return boundaries


def build_slice_intervals(boundaries_deg: Sequence[float]) -> List[Tuple[float, float]]:
    if not boundaries_deg or len(boundaries_deg) < 2:
        return [(0.0, 360.0)]
    edges = sorted(set(float(b) % 360.0 for b in boundaries_deg))
    if len(edges) < 2:
        return [(0.0, 360.0)]
    intervals: List[Tuple[float, float]] = []
    n = len(edges)
    for i in range(n):
        start = edges[i]
        end = edges[(i + 1) % n]
        if start == end:
            continue
        intervals.append((start, end))
    return intervals if intervals else [(0.0, 360.0)]

def angle_in_interval(angles: np.ndarray, start: float, end: float) -> np.ndarray:
    if start < end:
        return (angles >= start) & (angles < end)
    return (angles >= start) | (angles < end)

def build_slice_masks(
    mask_filled: np.ndarray,
    circle_center: Tuple[float, float],
    boundaries_deg: Sequence[float],
) -> List[np.ndarray]:
    h, w = mask_filled.shape
    cy, cx = float(circle_center[1]), float(circle_center[0])

    ys, xs = np.indices((h, w))
    angles = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0

    intervals = build_slice_intervals(boundaries_deg)
    slices: List[np.ndarray] = []
    for start, end in intervals:
        slice_mask = mask_filled & angle_in_interval(angles, start, end)
        if np.any(slice_mask):
            slices.append(slice_mask)
    return slices
def draw_boundaries(
    image_rgb: np.ndarray,
    circle: Circle,
    boundaries_deg: Sequence[float],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
    draw_circle: bool = True,
) -> Image.Image:
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)

    cx, cy, r = circle.cx, circle.cy, circle.r

    if draw_circle:
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=(0, 255, 0), width=max(1, width))

    for deg in boundaries_deg:
        theta = math.radians(deg)
        x2 = cx + r * math.cos(theta)
        y2 = cy + r * math.sin(theta)
        draw.line([(cx, cy), (x2, y2)], fill=color, width=width)

    return img


def colorize_label_map(label_map: np.ndarray) -> Image.Image:
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    uniq = np.unique(label_map[label_map >= 0])
    rng = np.random.default_rng(0)
    palette = {int(u): rng.integers(0, 255, size=3, dtype=np.uint8) for u in uniq}

    for u, col in palette.items():
        out[label_map == u] = col

    return Image.fromarray(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Extract pie-slice boundaries from image+mask")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--mask", required=True, help="Path to input mask (white=foreground)")
    p.add_argument("--out", required=True, help="Path to output overlay image")

    p.add_argument("--threshold", type=int, default=127, help="Mask binarization threshold")
    p.add_argument(
        "--k",
        default="auto",
        help="Number of color clusters, or 'auto' to estimate",
    )
    p.add_argument("--k-min", type=int, default=2, help="Min k when --k auto")
    p.add_argument("--k-max", type=int, default=12, help="Max k when --k auto")
    p.add_argument("--max-samples", type=int, default=20000, help="Max pixels for kmeans fitting")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    p.add_argument(
        "--preprocess",
        default="stretch",
        help="Color preprocess before clustering: stretch|none",
    )
    p.add_argument("--preprocess-alpha", type=float, default=1.5, help="Stretch strength")
    p.add_argument("--preprocess-power", type=float, default=2.0, help="Stretch nonlinearity")
    p.add_argument("--preprocess-clip", type=float, default=3.0, help="Preprocess clip in std units")

    p.add_argument("--angles", type=int, default=720, help="Angular resolution for boundary scan")
    p.add_argument("--min-span-deg", type=float, default=6.0, help="Minimum slice span; merges smaller")
    p.add_argument("--smooth-window", type=int, default=7, help="Circular majority smoothing window")

    p.add_argument("--line-width", type=int, default=3, help="Boundary line width")
    p.add_argument("--no-circle", action="store_true", help="Do not draw outer circle")

    p.add_argument("--debug-dir", default=None, help="Optional directory to dump intermediates")

    args = p.parse_args()

    _ensure_dir(args.debug_dir)

    image = load_image_rgb(args.image)
    mask = load_mask_bool(args.mask, threshold=args.threshold)
    mask_filled = fill_holes_binary(mask)

    circle = estimate_circle_from_mask(mask_filled)

    if args.debug_dir:
        save_debug_mask(mask, os.path.join(args.debug_dir, "mask_raw.png"))
        save_debug_mask(mask_filled, os.path.join(args.debug_dir, "mask_filled.png"))

    k_str = str(args.k).strip().lower()
    if k_str == "auto":
        k_val = 0
    else:
        k_val = int(k_str)

    label_map, chosen_k, k_debug = build_label_map(
        image_rgb=image,
        mask=mask_filled,
        k=k_val,
        k_min=args.k_min,
        k_max=args.k_max,
        max_samples=args.max_samples,
        seed=args.seed,
        preprocess=args.preprocess,
        preprocess_alpha=args.preprocess_alpha,
        preprocess_power=args.preprocess_power,
        preprocess_clip=args.preprocess_clip,
    )

    if args.debug_dir:
        colorize_label_map(label_map).save(os.path.join(args.debug_dir, "labels.png"))

    profile = angle_profile_labels(
        label_map=label_map,
        circle=circle,
        n_angles=args.angles,
    )

    profile_s = smooth_circular_labels(profile, window=args.smooth_window)
    boundaries = extract_boundaries_from_profile(profile_s, min_span_deg=args.min_span_deg)

    overlay = draw_boundaries(
        image_rgb=image,
        circle=circle,
        boundaries_deg=boundaries,
        width=args.line_width,
        draw_circle=(not args.no_circle),
    )

    overlay.save(args.out)

    if args.debug_dir:
        with open(os.path.join(args.debug_dir, "boundaries.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "circle": {"cx": circle.cx, "cy": circle.cy, "r": circle.r},
                    "k": chosen_k,
                    "k_selection": k_debug,
                    "angles": args.angles,
                    "boundaries_deg": boundaries,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
