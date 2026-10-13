# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import math

import cv2
import numpy as np

from retinal_vessel_system.core.state import ViewState
from retinal_vessel_system.processing.basic import crop_image


def as_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image


def view_orientation(image: np.ndarray, view: ViewState, binary: bool = False) -> np.ndarray:
    """导出统一 ROI 和旋转方向；观察倍率/平移不改变导出像素。"""
    image = crop_image(image, view.roi)
    if view.rotation_deg == 0:
        return image.copy()
    h, w = image.shape[:2]
    # Qt 屏幕坐标正角为顺时针，OpenCV 的正角为逆时针。
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -view.rotation_deg, 1)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = math.ceil(w * cos + h * sin - 1e-8), math.ceil(h * cos + w * sin - 1e-8)
    matrix[0, 2] += (nw - w) / 2
    matrix[1, 2] += (nh - h) / 2
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR)


def labeled_tile(image: np.ndarray, label: str, width: int = 360, height: int = 310) -> np.ndarray:
    tile = np.full((height, width, 3), 248, np.uint8)
    image = as_rgb(image)
    scale = min((width - 24) / image.shape[1], (height - 54) / image.shape[0])
    w, h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    x, y = (width - w) // 2, 42 + (height - 48 - h) // 2
    tile[y:y + h, x:x + w] = resized
    cv2.putText(tile, label, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (31, 41, 55), 1, cv2.LINE_AA)
    return tile


def before_after(original: np.ndarray, processed: np.ndarray, view: ViewState, binary=False) -> np.ndarray:
    left = as_rgb(view_orientation(original, view))
    right = as_rgb(view_orientation(processed, view, binary))
    header = np.full((40, left.shape[1] * 2, 3), 248, np.uint8)
    cv2.putText(header, "Original", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (31, 41, 55), 1, cv2.LINE_AA)
    cv2.putText(header, "Processed", (left.shape[1] + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (31, 41, 55), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack([left, right])])


def algorithm_grid(original: np.ndarray, results: dict, view: ViewState) -> np.ndarray:
    tiles = [labeled_tile(view_orientation(original, view), "Original")]
    for key, result in results.items():
        tiles.append(labeled_tile(view_orientation(result.image, view, result.binary), key.upper() if key == "otsu" else key.title()))
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 248))
    return np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])

