# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import fields
import math

from PySide6.QtCore import QObject, QPointF, QTimer, Signal
from PySide6.QtGui import QTransform

from retinal_vessel_system.core.state import ViewState
from retinal_vessel_system.processing.basic import clamp_roi


class SyncedViews(QObject):
    changed = Signal(object)

    def __init__(self, left, right, state: ViewState, parent=None):
        super().__init__(parent)
        self.views, self.state = (left, right), state
        self.width = self.height = 0
        self._locked_scale = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.apply)
        for view in self.views:
            view.zoom_requested.connect(self.zoom_by)
            view.pan_requested.connect(self.pan_by)
            view.roi_requested.connect(self.set_roi)
            view.resized.connect(lambda: self._resize_timer.start(0))

    def set_size(self, width, height):
        self.width, self.height = width, height
        self.reset()

    def reset(self):
        self._locked_scale = None
        default = ViewState()
        for field in fields(default):
            setattr(self.state, field.name, getattr(default, field.name))
        self._refresh_crop()
        self.apply()

    def fit(self):
        self._locked_scale = None
        self.state.fit, self.state.zoom = True, 1.0
        self.state.pan_x = self.state.pan_y = 0
        self.apply()

    def actual_size(self):
        self._locked_scale = None
        self.state.fit, self.state.zoom = False, 1.0
        self.state.pan_x = self.state.pan_y = 0
        self.apply()

    def zoom_by(self, factor: float):
        self.set_zoom(round(self.state.zoom * factor * 100))

    def set_zoom(self, percent):
        self.state.zoom = min(400, max(25, percent)) / 100
        self.apply()

    def set_rotation(self, degrees):
        self.state.rotation_deg = degrees
        self.apply()

    def begin_rotation(self):
        self._locked_scale = self._fit_scale()

    def end_rotation(self):
        # 保留本次旋转比例，直到用户明确适应窗口或更改 ROI。
        if self.state.fit:
            self._locked_scale = self._locked_scale or self._fit_scale()

    def pan_by(self, x, y):
        self.state.pan_x += x
        self.state.pan_y += y
        self.apply()

    def set_roi(self, roi):
        self._locked_scale = None
        self.state.roi = clamp_roi(roi, self.width, self.height) if roi and self.width else None
        self.state.pan_x = self.state.pan_y = 0
        self._refresh_crop()
        self.apply()

    def _refresh_crop(self):
        for view in self.views:
            view.update_crop(self.state.roi)

    def _fit_scale(self):
        x, y, w, h = self.state.roi or (0, 0, self.width, self.height)
        radians = math.radians(self.state.rotation_deg)
        rw = abs(w * math.cos(radians)) + abs(h * math.sin(radians))
        rh = abs(w * math.sin(radians)) + abs(h * math.cos(radians))
        vw = min(v.viewport().width() for v in self.views) - 24
        vh = min(v.viewport().height() for v in self.views) - 24
        return min(max(1, vw) / rw, max(1, vh) / rh) if rw and rh else 1

    def apply(self):
        if not self.width:
            return
        x, y, w, h = self.state.roi or (0, 0, self.width, self.height)
        scale = self.state.zoom
        if self.state.fit:
            scale *= self._locked_scale if self._locked_scale is not None else self._fit_scale()
        transform = QTransform().rotate(self.state.rotation_deg).scale(scale, scale)
        center = QPointF(x + w / 2 + self.state.pan_x, y + h / 2 + self.state.pan_y)
        for view in self.views:
            view.setTransform(transform)
            view.centerOn(center)
        self.changed.emit(self.state)

