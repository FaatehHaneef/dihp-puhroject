"""
Segmentation — HSV skin mask, morphology (open/close/dilate), connected components.
"""

import cv2
import numpy as np


def get_skin_mask_func(hsv_frame, h_range=(0, 20), s_range=(10, 150), v_range=(60, 255)):
    """Per-pixel HSV-range skin mask with hue wrap-around at 170-180."""
    h, w = hsv_frame.shape[:2]
    h_lo, h_hi = h_range
    s_lo, s_hi = s_range
    v_lo, v_hi = v_range

    mask = np.zeros((h, w), dtype=np.uint8)
    for yy in range(h):
        for xx in range(w):
            hue = int(hsv_frame[yy, xx, 0])
            sat = int(hsv_frame[yy, xx, 1])
            val = int(hsv_frame[yy, xx, 2])

            hue_ok = (h_lo <= hue <= h_hi) or (170 <= hue <= 180)
            if hue_ok and s_lo <= sat <= s_hi and v_lo <= val <= v_hi:
                mask[yy, xx] = 255
    return mask


def _build_elliptical_kernel(ksize):
    """Elliptical structuring element as a 2D 0/1 list."""
    pad = ksize // 2
    r = pad if pad > 0 else 1
    kernel = [[0] * ksize for _ in range(ksize)]
    for dy in range(ksize):
        for dx in range(ksize):
            ny = (dy - pad) / r
            nx = (dx - pad) / r
            if ny * ny + nx * nx <= 1.0:
                kernel[dy][dx] = 1
    return kernel


def _erosion(mask, kernel):
    h, w = mask.shape
    k = len(kernel)
    pad = k // 2
    out = np.zeros((h, w), dtype=np.uint8)
    for yy in range(h):
        for xx in range(w):
            keep = True
            for dy in range(k):
                py = yy + dy - pad
                if py < 0 or py >= h:
                    if any(kernel[dy][dx] for dx in range(k)):
                        keep = False
                        break
                    continue
                for dx in range(k):
                    if kernel[dy][dx] == 0:
                        continue
                    px = xx + dx - pad
                    if px < 0 or px >= w or mask[py, px] == 0:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                out[yy, xx] = 255
    return out


def _dilation(mask, kernel):
    h, w = mask.shape
    k = len(kernel)
    pad = k // 2
    out = np.zeros((h, w), dtype=np.uint8)
    for yy in range(h):
        for xx in range(w):
            hit = False
            for dy in range(k):
                py = yy + dy - pad
                if py < 0 or py >= h:
                    continue
                for dx in range(k):
                    if kernel[dy][dx] == 0:
                        continue
                    px = xx + dx - pad
                    if px < 0 or px >= w:
                        continue
                    if mask[py, px] != 0:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                out[yy, xx] = 255
    return out


def apply_morphology_manual(mask, morph_kernel_size=5):
    """Open -> Close -> Dilate using manual erode/dilate."""
    kernel = _build_elliptical_kernel(morph_kernel_size)
    opened = _dilation(_erosion(mask, kernel), kernel)
    closed = _erosion(_dilation(opened, kernel), kernel)
    return _dilation(closed, kernel)


def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb


def connected_component_analysis_func(mask):
    """8-connectivity two-pass labeling with union-find."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]  # parent[0] is background, never used as a label

    # Pass 1: provisional labeling
    next_label = 1
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx] == 0:
                continue
            # 8-connectivity backward neighbors
            neighbors = []
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                py, px = yy + dy, xx + dx
                if 0 <= py < h and 0 <= px < w and labels[py, px] != 0:
                    neighbors.append(labels[py, px])
            if not neighbors:
                labels[yy, xx] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                # Pick the smallest neighbor label and union the rest
                smallest = neighbors[0]
                for n in neighbors[1:]:
                    if n < smallest:
                        smallest = n
                labels[yy, xx] = smallest
                for n in neighbors:
                    if n != smallest:
                        _uf_union(parent, smallest, n)

    # Pass 2: replace each label with its root, then renumber compactly
    root_to_new = {}
    new_id = 1
    centroids_sum_x = [0.0]
    centroids_sum_y = [0.0]
    blob_sizes_list = [0]

    for yy in range(h):
        for xx in range(w):
            lbl = labels[yy, xx]
            if lbl == 0:
                continue
            root = _uf_find(parent, lbl)
            if root not in root_to_new:
                root_to_new[root] = new_id
                centroids_sum_x.append(0.0)
                centroids_sum_y.append(0.0)
                blob_sizes_list.append(0)
                new_id += 1
            compact = root_to_new[root]
            labels[yy, xx] = compact
            centroids_sum_x[compact] += xx
            centroids_sum_y[compact] += yy
            blob_sizes_list[compact] += 1

    num_labels = new_id  # includes background

    centroids = []
    for i in range(1, num_labels):
        if blob_sizes_list[i] > 0:
            cx = centroids_sum_x[i] / blob_sizes_list[i]
            cy = centroids_sum_y[i] / blob_sizes_list[i]
        else:
            cx, cy = 0.0, 0.0
        centroids.append((cx, cy))

    blob_sizes = np.array(blob_sizes_list[1:], dtype=np.int32)
    return labels, num_labels, centroids, blob_sizes

# =============================================================================


def get_skin_mask(hsv_frame, h_range=(0, 20), s_range=(10, 150), v_range=(60, 255)):
    """HSV-range skin mask with hue wrap-around (cv2 path)."""
    lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower, upper)

    lower_wrap = np.array([170, s_range[0], v_range[0]], dtype=np.uint8)
    upper_wrap = np.array([180, s_range[1], v_range[1]], dtype=np.uint8)
    mask_wrap = cv2.inRange(hsv_frame, lower_wrap, upper_wrap)
    return cv2.bitwise_or(mask, mask_wrap)


def apply_morphology(mask, morph_kernel_size=5):
    """Open -> Close -> Dilate with an elliptical kernel (cv2 path)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.dilate(mask, kernel, iterations=1)


def connected_component_analysis(mask):
    """8-connectivity labelling (cv2 path). Returns (labels, num_labels, centroids, blob_sizes)."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blob_sizes = stats[1:, cv2.CC_STAT_AREA]
    return labels, num_labels, centroids[1:], blob_sizes


def extract_hand_regions(mask, top_n=2):
    """Top-N largest blobs as (contour, bbox) tuples."""
    labels, num_labels, centroids, blob_sizes = connected_component_analysis(mask)
    if len(blob_sizes) == 0:
        return []
    top_indices = np.argsort(blob_sizes)[::-1][:top_n]

    hand_regions = []
    for idx in top_indices:
        label_id = idx + 1
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(contour)
            hand_regions.append((contour, bbox))
    return hand_regions


def extract_face_region(mask):
    """Largest blob as (contour, bbox)."""
    labels, num_labels, centroids, blob_sizes = connected_component_analysis(mask)
    if len(blob_sizes) == 0:
        return None, None
    largest_idx = np.argmax(blob_sizes)
    label_id = largest_idx + 1
    component_mask = (labels == label_id).astype(np.uint8) * 255

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(contour)
        return contour, bbox
    return None, None


