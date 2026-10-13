# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"无法读取图像：{path.name}（{exc}）") from exc
    if bgr is None or bgr.size == 0:
        raise ValueError(f"文件不是有效的 JPG、PNG、BMP 或 TIFF 图像：{path.name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, image: np.ndarray):
    path = Path(path)
    encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(path.suffix or ".png", encoded_image)
    if not ok:
        raise OSError(f"图像编码失败：{path.name}")
    encoded.tofile(path)

