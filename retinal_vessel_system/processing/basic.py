# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import cv2
import numpy as np


def algorithm_input(rgb: np.ndarray, channel: str = "gray") -> np.ndarray:
    return rgb[:, :, 1].copy() if channel == "green" else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def gray_preview(rgb: np.ndarray, percent: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.addWeighted(rgb, 1 - percent / 100, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), percent / 100, 0)


def window_image(image: np.ndarray, width: int, level: int) -> np.ndarray:
    # 标准端点语义；宽度默认 255/中心 128 时保留原 8 位像素。
    if width == 255 and level == 128:
        return image
    low = level - width / 2
    lookup = np.clip((np.arange(256, dtype=np.float32) - low) * (255 / width), 0, 255).astype(np.uint8)
    return cv2.LUT(image, lookup).reshape(image.shape)


def clamp_roi(roi, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in roi)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def crop_image(image: np.ndarray, roi) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = clamp_roi(roi, image.shape[1], image.shape[0])
    return image[y:y + h, x:x + w]

