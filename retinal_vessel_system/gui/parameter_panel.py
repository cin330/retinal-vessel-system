# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QScrollArea, QSlider, QTabWidget,
                             QVBoxLayout, QWidget)

from retinal_vessel_system.config import ALGORITHM_NAMES
from retinal_vessel_system.core.state import ProcessParams


class SliderRow(QWidget):
    changed = Signal(object)
    drag_started = Signal()
    drag_finished = Signal()

    def __init__(self, label, minimum, maximum, default, step=1, factor=1, suffix="", parent=None):
        super().__init__(parent)
        self.minimum, self.step, self.factor, self.suffix = minimum, step, factor, suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 5)
        layout.setSpacing(4)
        heading = QHBoxLayout()
        heading.addWidget(QLabel(label))
        self.value_label = QLabel()
        self.value_label.setObjectName("value")
        heading.addWidget(self.value_label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(heading)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, (maximum - minimum) // step)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._change)
        self.slider.sliderPressed.connect(self.drag_started)
        self.slider.sliderReleased.connect(self.drag_finished)
        self.set_value(default)

    def value(self):
        raw = self.minimum + self.slider.value() * self.step
        return raw if self.factor == 1 else raw / self.factor

    def _change(self):
        value = self.value()
        self.value_label.setText((f"{value:.1f}" if self.factor != 1 else str(value)) + self.suffix)
        self.changed.emit(value)

    def set_value(self, value):
        target = round((value * self.factor - self.minimum) / self.step)
        if self.slider.value() == target and self.value_label.text():
            return
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(target)
        self.slider.blockSignals(blocked)
        value = self.value()
        self.value_label.setText((f"{value:.1f}" if self.factor != 1 else str(value)) + self.suffix)

    def set_maximum(self, maximum):
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, (maximum - self.minimum) // self.step))
        self.slider.blockSignals(blocked)


class ParameterPanel(QFrame):
    params_changed = Signal(object, str)
    view_changed = Signal(str, object)
    restore_requested = Signal()
    interaction_started = Signal(str)
    interaction_finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("parameters")
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self.params = ProcessParams()
        self.rows = {}
        self.view_rows = {}
        self._image_size = (0, 0)
        layout = QVBoxLayout(self)
        title = QLabel("参数控制")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.processing_body, process_layout = self._scroll_tab("图像处理")
        self.geometry_body, geometry_layout = self._scroll_tab("同步视图")
        self.algorithm_title = QLabel()
        self.algorithm_title.setObjectName("sectionTitle")
        process_layout.addWidget(self.algorithm_title)
        self.algorithm_hint = QLabel()
        self.algorithm_hint.setObjectName("muted")
        self.algorithm_hint.setWordWrap(True)
        process_layout.addWidget(self.algorithm_hint)
        common = QGroupBox("公共预处理")
        common_layout = QVBoxLayout(common)
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("标准灰度输入", "gray")
        self.channel_combo.addItem("绿色通道输入", "green")
        self.channel_combo.currentIndexChanged.connect(lambda: self._set_param("input_channel", self.channel_combo.currentData()))
        common_layout.addWidget(self.channel_combo)
        self.filter_combo = QComboBox()
        for label, key in (("不去噪", "none"), ("高斯滤波", "gaussian"), ("自适应中值滤波", "adaptive_median")):
            self.filter_combo.addItem(label, key)
        self.filter_combo.currentIndexChanged.connect(lambda: self._set_param("denoise", self.filter_combo.currentData()))
        common_layout.addWidget(self.filter_combo)
        self._row(common_layout, "gaussian_kernel", "高斯核大小", 1, 31, step=2)
        self._row(common_layout, "gaussian_sigma", "高斯 σ（0 为自动）", 0, 100, factor=10)
        self._row(common_layout, "adaptive_median_max_kernel", "自适应中值最大窗口", 3, 15, step=2)
        self.algorithm_group = QGroupBox("算法参数")
        algo_layout = QVBoxLayout(self.algorithm_group)
        self.otsu_invert = QCheckBox("前景反相（暗区域为白色）")
        self.otsu_invert.toggled.connect(lambda value: self._set_param("otsu_invert", value))
        algo_layout.addWidget(self.otsu_invert)
        self._row(algo_layout, "watershed_sigma", "分水岭平滑 σ", 0, 50, factor=10)
        self._row(algo_layout, "watershed_foreground", "局部暗结构阈值", 1, 99, suffix="%")
        self._row(algo_layout, "watershed_min_distance", "标记最小间距", 1, 50, suffix=" px")
        self._row(algo_layout, "edge_kernel", "梯度核大小", 1, 7, step=2)
        self._row(algo_layout, "edge_threshold", "边缘阈值（0 为连续强度）", 0, 255)
        self._row(algo_layout, "canny_low", "Canny 低阈值", 0, 254)
        self._row(algo_layout, "canny_high", "Canny 高阈值", 1, 255)
        process_layout.addWidget(self.algorithm_group)
        process_layout.addWidget(common)
        display = QGroupBox("显示调节")
        display_layout = QVBoxLayout(display)
        self._row(display_layout, "gray_mix", "灰度预览", 0, 100, suffix="%")
        self._row(display_layout, "window_width", "窗宽 WW", 1, 255)
        self._row(display_layout, "window_level", "窗位 WL", 0, 255)
        self.binary_note = QLabel("二值掩膜保持黑白标签，显示调节不改变算法输入。")
        self.binary_note.setObjectName("muted")
        self.binary_note.setWordWrap(True)
        display_layout.addWidget(self.binary_note)
        process_layout.addWidget(display)
        process_layout.addStretch()
        for name, label, lo, hi, default, suffix in (
            ("zoom", "同步缩放", 25, 400, 100, "%"),
            ("rotation", "同步旋转", -180, 180, 0, "°"),
        ):
            row = SliderRow(label, lo, hi, default, suffix=suffix)
            row.changed.connect(lambda value, key=name: self.view_changed.emit(key, value))
            row.drag_started.connect(lambda key=name: self.interaction_started.emit(key))
            row.drag_finished.connect(lambda key=name: self.interaction_finished.emit(key))
            self.view_rows[name] = row
            geometry_layout.addWidget(row)
        self.scale_note = QLabel("倍率相对于适应窗口；选择 1:1 后按原图像素缩放。")
        self.scale_note.setWordWrap(True)
        self.scale_note.setObjectName("muted")
        geometry_layout.addWidget(self.scale_note)
        crop = QGroupBox("非破坏式裁剪 / 边距")
        crop_layout = QVBoxLayout(crop)
        for name, label in (("left", "左边距"), ("right", "右边距"), ("top", "上边距"), ("bottom", "下边距")):
            row = SliderRow(label, 0, 1, 0, suffix=" px")
            row.changed.connect(lambda value, key=name: self.view_changed.emit(key, value))
            self.view_rows[name] = row
            crop_layout.addWidget(row)
        clear = QPushButton("清除裁剪")
        clear.clicked.connect(lambda: self.view_changed.emit("clear_roi", None))
        crop_layout.addWidget(clear)
        geometry_layout.addWidget(crop)
        geometry_layout.addStretch()
        reset = QPushButton("处理参数恢复默认")
        reset.clicked.connect(self.restore_requested)
        layout.addWidget(reset)
        self.set_params(self.params)
        self.setEnabled(False)

    def _scroll_tab(self, title):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("scrollBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(10)
        scroll.setWidget(body)
        self.tabs.addTab(scroll, title)
        return body, layout

    def _row(self, layout, key, label, lo, hi, step=1, factor=1, suffix=""):
        row = SliderRow(label, lo, hi, getattr(self.params, key), step, factor, suffix)
        row.changed.connect(lambda value: self._set_param(key, value))
        row.drag_started.connect(lambda: self.interaction_started.emit(key))
        row.drag_finished.connect(lambda: self.interaction_finished.emit(key))
        layout.addWidget(row)
        self.rows[key] = row

    def _set_param(self, key, value):
        updates = {key: value}
        if key == "canny_low" and value >= self.params.canny_high:
            updates["canny_high"] = value + 1
        if key == "canny_high" and value <= self.params.canny_low:
            updates["canny_low"] = value - 1
        self.params = replace(self.params, **updates)
        for changed_key, changed_value in updates.items():
            if changed_key != key:
                self.rows[changed_key].set_value(changed_value)
        if key in ("denoise", "edge_threshold"):
            self._update_visibility()
        self.params_changed.emit(self.params, key)

    def set_params(self, params):
        self.params = params
        self.algorithm_title.setText(f"{ALGORITHM_NAMES[params.algorithm]}参数")
        for combo, key in ((self.channel_combo, "input_channel"), (self.filter_combo, "denoise")):
            previous = combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(getattr(params, key)))
            combo.blockSignals(previous)
        previous = self.otsu_invert.blockSignals(True)
        self.otsu_invert.setChecked(params.otsu_invert)
        self.otsu_invert.blockSignals(previous)
        for key, row in self.rows.items():
            row.set_value(getattr(params, key))
        self._update_visibility()

    def _update_visibility(self):
        algorithm = self.params.algorithm
        visible = {
            "gaussian_kernel": self.params.denoise == "gaussian",
            "gaussian_sigma": self.params.denoise == "gaussian",
            "adaptive_median_max_kernel": self.params.denoise == "adaptive_median",
            "watershed_sigma": algorithm == "watershed",
            "watershed_foreground": algorithm == "watershed",
            "watershed_min_distance": algorithm == "watershed",
            "edge_kernel": algorithm in ("sobel", "laplacian"),
            "edge_threshold": algorithm in ("sobel", "prewitt", "roberts", "laplacian", "scharr"),
            "canny_low": algorithm == "canny", "canny_high": algorithm == "canny",
        }
        for key, show in visible.items():
            self.rows[key].setVisible(show)
        self.otsu_invert.setVisible(algorithm == "otsu")
        self.algorithm_group.setVisible(algorithm != "preview")
        self.rows["gray_mix"].setVisible(algorithm == "preview")
        binary = algorithm in ("otsu", "watershed", "canny") or (algorithm != "preview" and self.params.edge_threshold > 0)
        for key in ("window_width", "window_level"):
            self.rows[key].setEnabled(not binary)
        self.binary_note.setVisible(binary)
        hints = {
            "preview": "调整灰度、去噪与显示范围，左侧保持原始颜色。",
            "otsu": "自动选择全局阈值；运行后显示实际阈值。",
            "watershed": "局部暗结构增强、自适应候选与距离峰标记。",
            "canny": "双阈值检测；高阈值始终大于低阈值。",
        }
        self.algorithm_hint.setText(hints.get(algorithm, "输出归一化边缘强度；阈值大于 0 时输出二值图。"))

    def set_image_size(self, width, height):
        self._image_size = width, height

    def sync_view(self, state):
        self.view_rows["zoom"].set_value(round(state.zoom * 100))
        self.view_rows["rotation"].set_value(state.rotation_deg)
        width, height = self._image_size
        if not width:
            return
        x, y, w, h = state.roi or (0, 0, width, height)
        margins = {"left": x, "right": width - x - w, "top": y, "bottom": height - y - h}
        opposites = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        for key, value in margins.items():
            dimension = width if key in ("left", "right") else height
            self.view_rows[key].set_maximum(dimension - margins[opposites[key]] - 1)
            self.view_rows[key].set_value(value)
        self.scale_note.setText("当前倍率：适应窗口 × 缩放比例" if state.fit else "当前倍率：原图像素 × 缩放比例")

