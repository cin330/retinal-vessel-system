# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation
from skimage.feature import peak_local_max


def otsu_segment(gray: np.ndarray, invert: bool = False) -> tuple[np.ndarray, dict]:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return mask, {"threshold": float(threshold)}


def watershed_segment(gray: np.ndarray, sigma: float = 1.5, foreground: int = 35,
                      min_distance: int = 12, cancel=None, spatial_scale=1.0) -> tuple[np.ndarray, dict]:
    """局部暗结构增强 → 自适应候选 → 距离峰标记 → 标记控制分水岭。"""
    smooth = filters.gaussian(gray, sigma=sigma, preserve_range=True).astype(np.float32)
    # 黑帽增强局部暗线；黑色拍摄边缘不作为前景。
    contrast = morphology.black_tophat(smooth, footprint=morphology.disk(max(1, round(9 * spatial_scale))))
    field = gray > 5
    max_contrast = float(contrast[field].max()) if field.any() else 0.0
    if max_contrast <= 1e-6:
        return np.zeros_like(gray), {"marker_count": 0, "region_count": 0, "contrast_threshold": 0.0}
    threshold = max_contrast * foreground / 100
    block = max(3, (round(35 * spatial_scale) // 2) * 2 + 1) if spatial_scale != 1 else 35
    local = filters.threshold_local(contrast, block_size=block, method="gaussian")
    candidates = (contrast > np.maximum(local, threshold)) & field
    # 显式按连通分量大小过滤，兼容 scikit-image 0.23–0.26 的阈值 API。
    candidate_labels, _ = ndi.label(candidates)
    sizes = np.bincount(candidate_labels.ravel())
    keep = sizes >= max(2, round(12 * spatial_scale ** 2))
    keep[0] = False
    candidates = keep[candidate_labels]
    if cancel:
        cancel()
    distance = ndi.distance_transform_edt(candidates)
    components, count = ndi.label(candidates)
    markers = np.zeros(gray.shape, np.int32)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=components, exclude_border=False)
    for label, (y, x) in enumerate(peaks, 1):
        markers[y, x] = label
    # 细小连通分量也必须有种子；仅补充缺失的分量。
    seeded = set(components[markers > 0].tolist())
    missing = [i for i in range(1, count + 1) if i not in seeded]
    positions = ndi.maximum_position(distance, labels=components, index=missing) if missing else []
    next_marker = len(peaks) + 1
    for y, x in positions:
        markers[y, x] = next_marker
        next_marker += 1
    if cancel:
        cancel()
    labels = segmentation.watershed(-distance, markers, mask=candidates, watershed_line=True)
    mask = (labels > 0).astype(np.uint8) * 255
    return mask, {"marker_count": next_marker - 1, "region_count": int(labels.max()),
                  "contrast_threshold": round(threshold, 4), "candidate_components": int(count)}

