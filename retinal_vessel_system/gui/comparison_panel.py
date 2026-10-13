# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import json
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from retinal_vessel_system.config import ALGORITHM_NAMES, COMPARISON_ALGORITHMS
from retinal_vessel_system.utils.qt_image import to_pixmap


class ResultCard(QPushButton):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumWidth(100)
        self.setFixedHeight(140)
        self._result = None
        self._pixmap = None
        self._scaled_size = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)
        self.thumbnail = QLabel('等待图像')
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFixedHeight(87)
        self.thumbnail.setMinimumWidth(0)
        layout.addWidget(self.thumbnail)
        self.title = QLabel(label)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFixedHeight(18)
        layout.addWidget(self.title)
        self.detail = QLabel()
        self.detail.setObjectName('muted')
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setFixedHeight(17)
        layout.addWidget(self.detail)
        for widget in (self.thumbnail, self.title, self.detail):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setEnabled(False)

    def set_result(self, result, current=True, loaded=False):
        if result is not self._result:
            self._result = result
            self._pixmap = to_pixmap(result.image) if result else None
            self._scaled_size = None
            self._scale_thumbnail()
        if result is None:
            self.thumbnail.setText('计算中' if loaded else '等待图像')
            self.detail.clear()
        else:
            self.detail.setText(f'{result.elapsed_ms:.1f} ms' if current else '待更新')
            self.setToolTip(json.dumps(asdict(result.params), ensure_ascii=False, indent=2))
        # 缓存过期时仍允许选择算法，但控制器会计算当前参数，不能恢复旧快照。
        self.setEnabled(loaded)

    def _scale_thumbnail(self):
        if self._pixmap is None:
            self.thumbnail.clear()
            return
        size = (max(1, self.width() - 20), 87)
        if size == self._scaled_size:
            return
        self.thumbnail.setPixmap(self._pixmap.scaled(*size, Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        self._scaled_size = size

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_thumbnail()


class ComparisonPanel(QFrame):
    selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QHBoxLayout()
        title = QLabel('算法对比')
        title.setObjectName('sectionTitle')
        heading.addWidget(title)
        hint = QLabel('点击结果查看 · 上传后自动计算全部算法')
        hint.setObjectName('muted')
        heading.addWidget(hint)
        heading.addStretch()
        layout.addLayout(heading)
        row = QHBoxLayout()
        row.setSpacing(8)
        for key in COMPARISON_ALGORITHMS:
            label = 'Otsu' if key == 'otsu' else '分水岭' if key == 'watershed' else ALGORITHM_NAMES[key]
            card = ResultCard(label)
            card.clicked.connect(lambda checked=False, algorithm=key: self.selected.emit(algorithm))
            row.addWidget(card, 1)
            self.cards[key] = card
        layout.addLayout(row)

    def set_results(self, results, current_keys=None, loaded=False):
        current_keys = set(results) if current_keys is None else current_keys
        for key, card in self.cards.items():
            card.set_result(results.get(key), key in current_keys, loaded)

    def set_selected(self, algorithm):
        for key, card in self.cards.items():
            card.setChecked(key == algorithm and card.isEnabled())
