# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import cv2
import numpy as np
from skimage import filters


def normalize_gradient(gradient: np.ndarray, threshold: int = 0) -> np.ndarray:
    magnitude = np.abs(gradient).astype(np.float32)
    maximum = float(magnitude.max())
    scaled = (magnitude * (255 / maximum)).astype(np.uint8) if maximum > 0 else np.zeros(magnitude.shape, np.uint8)
    return np.where(scaled >= threshold, 255, 0).astype(np.uint8) if threshold > 0 else scaled


def detect_edges(gray: np.ndarray, algorithm: str, kernel: int = 3, threshold: int = 0,
                 canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
    if algorithm == "canny":
        if canny_high <= canny_low:
            raise ValueError("Canny 高阈值必须大于低阈值")
        return cv2.Canny(gray, canny_low, canny_high)
    if algorithm in ("sobel", "scharr"):
        operator = cv2.Scharr if algorithm == "scharr" else cv2.Sobel
        extra = {} if algorithm == "scharr" else {"ksize": kernel}
        dx = operator(gray, cv2.CV_32F, 1, 0, **extra)
        dy = operator(gray, cv2.CV_32F, 0, 1, **extra)
        gradient = cv2.magnitude(dx, dy)
    elif algorithm == "prewitt":
        gradient = filters.prewitt(gray)
    elif algorithm == "roberts":
        gradient = filters.roberts(gray)
    elif algorithm == "laplacian":
        gradient = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel)
    else:
        raise ValueError(f"未知边缘算法：{algorithm}")
    return normalize_gradient(gradient, threshold)

