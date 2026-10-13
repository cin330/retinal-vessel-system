# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_qimage(image: np.ndarray) -> QImage:
    array = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = array.shape[:2]
    fmt = QImage.Format.Format_Grayscale8 if array.ndim == 2 else QImage.Format.Format_RGB888
    # copy 避免 Qt 持有生命周期短于控件的 NumPy 缓冲区。
    return QImage(array.data, w, h, array.strides[0], fmt).copy()


def to_pixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(image))

