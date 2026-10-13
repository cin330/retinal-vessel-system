# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from retinal_vessel_system.config import STYLE


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(STYLE)
    return application


@pytest.fixture
def fundus():
    # 确定性的暗线/亮盘图；用于行为验证，不冒充真实血管标注。
    image = np.full((96, 128, 3), (180, 100, 60), np.uint8)
    cv2.line(image, (10, 12), (110, 80), (45, 25, 18), 4)
    cv2.line(image, (60, 40), (100, 15), (45, 25, 18), 3)
    cv2.circle(image, (25, 65), 10, (245, 225, 170), -1)
    return image


def wait_until(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("等待后台处理超时")

