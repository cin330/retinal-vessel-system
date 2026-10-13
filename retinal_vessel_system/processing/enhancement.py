# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, median_filter, minimum_filter


def gaussian_filter(image: np.ndarray, kernel: int = 5, sigma: float = 0) -> np.ndarray:
    if not 1 <= kernel <= 31 or kernel % 2 != 1:
        raise ValueError("高斯核应为 1–31 的奇数")
    return image.copy() if kernel == 1 else cv2.GaussianBlur(image, (kernel, kernel), sigma)


def adaptive_median_filter(image: np.ndarray, max_kernel: int = 7, cancel=None) -> np.ndarray:
    """经典两阶段自适应中值：A 判断中值有效性，B 判断原像素是否为脉冲。"""
    if image.ndim != 2:
        raise ValueError("自适应中值滤波使用单通道图像")
    if not 3 <= max_kernel <= 15 or max_kernel % 2 != 1:
        raise ValueError("最大窗口应为 3–15 的奇数")
    output = image.copy()
    pending = np.ones(image.shape, dtype=bool)
    for size in range(3, max_kernel + 1, 2):
        if cancel:
            cancel()
        lo = minimum_filter(image, size=size, mode="reflect")
        hi = maximum_filter(image, size=size, mode="reflect")
        mid = median_filter(image, size=size, mode="reflect")
        stage_a = pending & (mid > lo) & (mid < hi)
        stage_b = (image > lo) & (image < hi)
        output[stage_a & ~stage_b] = mid[stage_a & ~stage_b]
        pending[stage_a] = False
        if not pending.any():
            break
    output[pending] = mid[pending]
    return output

