# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import math

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QRubberBand

from retinal_vessel_system.utils.qt_image import to_pixmap


class ImageView(QGraphicsView):
    zoom_requested = Signal(float)
    pan_requested = Signal(float, float)
    roi_requested = Signal(object)
    resized = Signal()

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.clip = QGraphicsRectItem()
        self.clip.setPen(QPen(Qt.PenStyle.NoPen))
        self.clip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape)
        self.scene().addItem(self.clip)
        self.item = QGraphicsPixmapItem(self.clip)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#F8FAFC"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder = placeholder
        self.image = None
        self.logical_size = (0, 0)
        self.crop_mode = False
        self._last = None
        self._origin = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_image(self, image: np.ndarray, roi=None, logical_size=None):
        self.image = image
        size = tuple(logical_size or (image.shape[1], image.shape[0]))
        if size != self.logical_size:
            self.logical_size = size
            w, h = size
            extent = max(w, h) * 16
            self.setSceneRect(QRectF(-extent, -extent, 2 * extent + w, 2 * extent + h))
        self.item.setPixmap(to_pixmap(image))
        self.item.setTransform(QTransform.fromScale(size[0] / image.shape[1], size[1] / image.shape[0]))
        self.update_crop(roi)

    def update_crop(self, roi):
        if self.image is None:
            return
        w, h = self.logical_size
        self.clip.setRect(QRectF(*roi) if roi else QRectF(0, 0, w, h))
        self.viewport().update()

    def set_crop_mode(self, enabled: bool):
        self.crop_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        if self.image is not None:
            self.zoom_requested.emit(1.15 ** (event.angleDelta().y() / 120))
        event.accept()

    def mousePressEvent(self, event):
        if self.image is not None and event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position().toPoint()
            if self.crop_mode:
                self._origin = self._last
                self._rubber.setGeometry(QRect(self._origin, self._origin))
                self._rubber.show()
            else:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last is not None:
            position = event.position().toPoint()
            if self.crop_mode:
                self._rubber.setGeometry(QRect(self._origin, position).normalized())
            else:
                delta = self.mapToScene(self._last) - self.mapToScene(position)
                self.pan_requested.emit(delta.x(), delta.y())
            self._last = position
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._last is not None:
            if self.crop_mode:
                rectangle = self._rubber.geometry()
                self._rubber.hide()
                if rectangle.width() > 3 and rectangle.height() > 3:
                    # 旋转后的屏幕矩形逆映射四角，取原图坐标的包围矩形。
                    corners = [self.mapToScene(p) for p in (rectangle.topLeft(), rectangle.topRight(),
                                                           rectangle.bottomLeft(), rectangle.bottomRight())]
                    x = max(0, math.floor(min(p.x() for p in corners)))
                    y = max(0, math.floor(min(p.y() for p in corners)))
                    right = min(self.logical_size[0], math.ceil(max(p.x() for p in corners)))
                    bottom = min(self.logical_size[1], math.ceil(max(p.y() for p in corners)))
                    if right > x and bottom > y:
                        self.roi_requested.emit((x, y, right - x, bottom - y))
            self._last = self._origin = None
            self.set_crop_mode(self.crop_mode)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#94A3B8"))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            painter.end()

