# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
                             QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from retinal_vessel_system.config import ALGORITHM_NAMES, DATA_DIR, OUTPUT_DIR
from retinal_vessel_system.core.controller import Controller
from retinal_vessel_system.core.state import ProcessParams
from retinal_vessel_system.gui.comparison_panel import ComparisonPanel
from retinal_vessel_system.gui.image_view import ImageView
from retinal_vessel_system.gui.parameter_panel import ParameterPanel
from retinal_vessel_system.gui.synced_view import SyncedViews
from retinal_vessel_system.processing.pipeline import display_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('视网膜血管图像分割系统')
        self.resize(1440, 900)
        self.setMinimumSize(1200, 760)
        self.setAcceptDrops(True)
        self.controller = Controller(self)
        self._exporting = self._mouse_dragging = self._preview_dirty = False
        self._displayed_result = None
        self._last_directory = str(DATA_DIR)
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)
        self._preview_timer.timeout.connect(self._preview_tick)
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(16)
        self._display_timer.setSingleShot(True)
        self._display_timer.timeout.connect(self._render_display)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(100)
        self._settle.timeout.connect(self._finish_processing_interaction)
        self._build_ui()
        self.sync = SyncedViews(self.left_view, self.right_view, self.controller.view, self)
        self.sync.changed.connect(self.parameters.sync_view)
        self.sync.changed.connect(self._update_view_status)
        self.controller.image_loaded.connect(self._on_image_loaded)
        self.controller.result_ready.connect(self._on_result)
        self.controller.preview_ready.connect(self._on_preview)
        self.controller.comparisons_ready.connect(self._on_comparisons)
        self.controller.busy_changed.connect(lambda _: self._update_actions())
        self.controller.status.connect(self.status_label.setText)
        self.controller.error.connect(self._show_error)
        self.controller.exported.connect(self._on_exported)
        self.parameters.params_changed.connect(self._on_params)
        self.parameters.view_changed.connect(self._on_view_parameter)
        self.parameters.interaction_started.connect(self._interaction_started)
        self.parameters.interaction_finished.connect(self._interaction_finished)
        self.parameters.restore_requested.connect(self.restore_parameters)
        self.comparison.selected.connect(self.select_algorithm)
        self._update_actions()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 10)
        root.setSpacing(12)
        toolbar = QHBoxLayout()
        brand = QLabel('视网膜血管图像分割系统')
        brand.setObjectName('brand')
        toolbar.addWidget(brand)
        toolbar.addStretch()
        self.upload_button = self._button('上传图像', self.open_image, 'primary')
        self.execute_button = self._button('执行处理', self.execute)
        self.batch_button = self._button('更新全部对比', self.batch_compare)
        self.save_button = self._button('保存结果', self.save_results)
        self.reset_button = self._button('全部重置', self.reset_all)
        for button in (self.upload_button, self.execute_button, self.batch_button, self.save_button, self.reset_button):
            toolbar.addWidget(button)
        root.addLayout(toolbar)
        workspace = QHBoxLayout()
        workspace.setSpacing(12)
        self.navigation = QFrame()
        self.navigation.setObjectName('sidebar')
        self.navigation.setFixedWidth(165)
        side = QVBoxLayout(self.navigation)
        side.setContentsMargins(10, 10, 10, 10)
        side.setSpacing(4)
        self.algorithm_buttons = {}
        self.algorithm_group = QButtonGroup(self)
        for category, algorithms in (('基础与增强', ('preview',)), ('图像分割', ('otsu', 'watershed')),
                                     ('边缘检测', ('sobel', 'prewitt', 'roberts', 'canny', 'laplacian', 'scharr'))):
            heading = QLabel(category)
            heading.setObjectName('navHeading')
            heading.setFixedHeight(25)
            side.addWidget(heading)
            for key in algorithms:
                label = '基础预览' if key == 'preview' else 'Otsu 自动阈值' if key == 'otsu' else ALGORITHM_NAMES[key]
                button = QPushButton(label)
                button.setObjectName('navButton')
                button.setCheckable(True)
                button.setFixedHeight(30)
                button.clicked.connect(lambda checked=False, algorithm=key: self.select_algorithm(algorithm))
                self.algorithm_group.addButton(button)
                self.algorithm_buttons[key] = button
                side.addWidget(button)
        self.algorithm_buttons['preview'].setChecked(True)
        side.addStretch()
        self.image_info = QLabel('尚未上传图像')
        self.image_info.setObjectName('muted')
        self.image_info.setWordWrap(True)
        side.addWidget(self.image_info)
        self.sample_button = self._button('加载示例眼底图', self.load_example)
        side.addWidget(self.sample_button)
        workspace.addWidget(self.navigation)
        center = QVBoxLayout()
        center.setSpacing(8)
        images = QHBoxLayout()
        images.setSpacing(12)
        self.left_view = ImageView('上传或拖入一张眼底图像\n\n也可加载左侧示例图')
        self.right_view = ImageView('选择左侧算法并调整右侧参数')
        self.left_title, self.right_title = QLabel('原始图像'), QLabel('处理结果')
        for title, view in ((self.left_title, self.left_view), (self.right_title, self.right_view)):
            card = QFrame()
            card.setObjectName('imageCard')
            layout = QVBoxLayout(card)
            layout.setContentsMargins(10, 10, 10, 10)
            title.setObjectName('sectionTitle')
            layout.addWidget(title)
            layout.addWidget(view, 1)
            images.addWidget(card, 1)
        center.addLayout(images, 1)
        tools = QHBoxLayout()
        self.fit_button = self._button('适应窗口', lambda: self.sync.fit())
        self.actual_button = self._button('1:1', lambda: self.sync.actual_size())
        self.crop_button = self._button('框选 ROI', self._toggle_crop)
        self.crop_button.setCheckable(True)
        self.crop_button.setToolTip('在任一图像框选，原图坐标的共同 ROI 同步显示')
        self.view_reset_button = self._button('重置视图', lambda: self.sync.reset())
        tools.addWidget(QLabel('双图同步'))
        tools.addStretch()
        for button in (self.fit_button, self.actual_button, self.crop_button, self.view_reset_button):
            tools.addWidget(button)
        center.addLayout(tools)
        self.view_status = QLabel('滚轮缩放 · 拖动平移 · 两图共享几何视图')
        self.view_status.setObjectName('muted')
        center.addWidget(self.view_status)
        self.result_info = QLabel('上传后自动生成八种算法结果')
        self.result_info.setObjectName('muted')
        center.addWidget(self.result_info)
        workspace.addLayout(center, 1)
        self.parameters = ParameterPanel()
        workspace.addWidget(self.parameters)
        root.addLayout(workspace, 1)
        self.comparison = ComparisonPanel()
        root.addWidget(self.comparison)
        self.setCentralWidget(central)
        self.status_label = QLabel('就绪 · 上传一张眼底图像开始')
        self.statusBar().addWidget(self.status_label, 1)
        notice = QLabel('仅供教学与研究')
        notice.setObjectName('muted')
        self.statusBar().addPermanentWidget(notice)
        for label, shortcut, handler in (('打开图像', 'Ctrl+O', self.open_image),
                                         ('保存结果', 'Ctrl+S', self.save_results),
                                         ('执行处理', 'Ctrl+Return', self.execute)):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(handler)
            self.addAction(action)

    @staticmethod
    def _button(text, callback, name=None):
        button = QPushButton(text)
        if name:
            button.setObjectName(name)
        button.clicked.connect(callback)
        return button

    def open_image(self):
        if self._exporting:
            return
        filename, _ = QFileDialog.getOpenFileName(self, '上传眼底图像', self._last_directory,
                                                 '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)')
        if filename:
            self.load_image(filename)

    def _stop_interaction(self):
        self._preview_timer.stop()
        self._settle.stop()
        self._display_timer.stop()
        self._preview_dirty = self._mouse_dragging = False

    def load_image(self, path):
        if self._exporting:
            return
        self._stop_interaction()
        if self.controller.load_image(path):
            self._last_directory = str(Path(path).parent)

    def load_example(self):
        self.load_image(DATA_DIR / 'retina.jpg')

    def _on_image_loaded(self):
        self._displayed_result = None
        rgb = self.controller.state.original
        h, w = rgb.shape[:2]
        self.left_view.set_image(rgb)
        self.right_view.set_image(rgb)
        self.parameters.setEnabled(True)
        self.parameters.set_params(self.controller.params)
        self.parameters.set_image_size(w, h)
        self.sync.set_size(w, h)
        self.image_info.setText(f'{w} × {h} px\nRGB · 8 位')
        self.image_info.setToolTip(str(self.controller.state.source_path))
        self.crop_button.setChecked(False)
        self._toggle_crop(False)
        self._update_selection()
        self._refresh_comparisons()
        self._update_actions()

    def _on_params(self, params, key=''):
        changed = self.controller.update_params(params)
        if changed:
            if not self.controller.dragging:
                self.controller.begin_drag()
            self._preview_dirty = True
            if not self._preview_timer.isActive():
                self._preview_timer.start()
            if not self._mouse_dragging:
                self._settle.start()
            self.result_info.setText('正在预览 · 松开后计算原始分辨率')
            self._refresh_comparisons()
        else:
            # 合并显示调节，但不重新执行算法或改变视图。
            if not self._display_timer.isActive():
                self._display_timer.start()
        self._update_actions()

    def _preview_tick(self):
        if self._preview_dirty:
            self._preview_dirty = False
            self.controller.request_preview()

    def _interaction_started(self, key):
        if key == 'rotation':
            self.sync.begin_rotation()
        if key in self.parameters.rows and key not in ('window_width', 'window_level'):
            self._mouse_dragging = True
            self._settle.stop()
            self.controller.begin_drag()

    def _interaction_finished(self, key):
        if key == 'rotation':
            self.sync.end_rotation()
        if key in self.parameters.rows and key not in ('window_width', 'window_level'):
            self._finish_processing_interaction()

    def _finish_processing_interaction(self):
        self._stop_interaction()
        self.controller.end_drag()
        self.result_info.setText('正在计算最终结果…' if not self.controller.result_is_current() else self.result_info.text())
        self._update_actions()

    def execute(self):
        if self._exporting:
            return
        self._stop_interaction()
        self.controller.end_drag(schedule=False)
        self.controller.request_processing(batch=True)

    def batch_compare(self):
        self.execute()

    def _on_result(self, result):
        self._displayed_result = result
        self._render_display()
        self._update_actions()

    def _on_preview(self, result):
        self._displayed_result = result
        self._render_display()

    def _render_display(self):
        result = self._displayed_result
        if result is None:
            return
        image = display_image(result, self.controller.params)
        logical = result.metadata.get('source_size')
        self.right_view.set_image(image, self.controller.view.roi, logical_size=logical)
        self.right_title.setText(f'处理结果 · {ALGORITHM_NAMES[result.algorithm]}')
        detail = ''
        if 'threshold' in result.metadata:
            detail = f" · 阈值 {result.metadata['threshold']:.1f}"
        elif 'marker_count' in result.metadata:
            detail = f" · 标记 {result.metadata['marker_count']} 个"
        quality = '快速预览' if result.metadata.get('preview') else '原始分辨率'
        self.result_info.setText(f'{ALGORITHM_NAMES[result.algorithm]} · {result.elapsed_ms:.1f} ms{detail} · {quality}')

    def _refresh_comparisons(self):
        loaded = self.controller.state.original is not None
        keys = {k for k in self.comparison.cards if self.controller.comparison_is_current(k)}
        self.comparison.set_results(self.controller.state.results, keys, loaded)
        self.comparison.set_selected(self.controller.params.algorithm)

    def _on_comparisons(self, results):
        self._refresh_comparisons()
        self._update_actions()

    def _update_selection(self):
        self.algorithm_buttons[self.controller.params.algorithm].setChecked(True)
        self.comparison.set_selected(self.controller.params.algorithm)

    def select_algorithm(self, algorithm):
        if self._exporting:
            return
        self._stop_interaction()
        self.controller.set_algorithm(algorithm)
        self.parameters.set_params(self.controller.params)
        self.parameters.tabs.setCurrentIndex(0)
        self._update_selection()
        self._refresh_comparisons()

    def _on_view_parameter(self, key, value):
        if key == 'zoom':
            self.sync.set_zoom(value)
        elif key == 'rotation':
            self.sync.set_rotation(value)
        elif key == 'clear_roi':
            self.sync.set_roi(None)
        else:
            width, height = self.parameters._image_size
            margins = {k: self.parameters.view_rows[k].value() for k in ('left', 'right', 'top', 'bottom')}
            margins[key] = value
            self.sync.set_roi((margins['left'], margins['top'], width - margins['left'] - margins['right'],
                               height - margins['top'] - margins['bottom']))

    def _update_view_status(self, state):
        crop = f'ROI {state.roi}' if state.roi else '完整图像'
        self.view_status.setText(f'同步视图 · {round(state.zoom * 100)}% · {state.rotation_deg:g}° · {crop}')

    def _toggle_crop(self, checked):
        for view in (self.left_view, self.right_view):
            view.set_crop_mode(checked)
        if checked:
            self.parameters.tabs.setCurrentIndex(1)

    def restore_parameters(self):
        params = ProcessParams(algorithm=self.controller.params.algorithm)
        self.parameters.set_params(params)
        self._on_params(params)
        if not self.controller.dragging:
            self.execute()

    def reset_all(self):
        if self.controller.state.original is None or self._exporting:
            return
        self._stop_interaction()
        self.controller.reset()
        self.parameters.set_params(self.controller.params)
        self.sync.reset()
        self.crop_button.setChecked(False)
        self._toggle_crop(False)
        self._update_selection()

    def _update_actions(self):
        loaded = self.controller.state.original is not None
        for button in (self.execute_button, self.batch_button, self.reset_button,
                       self.fit_button, self.actual_button, self.crop_button, self.view_reset_button):
            button.setEnabled(loaded and not self._exporting)
        for button in self.algorithm_buttons.values():
            button.setEnabled(loaded and not self._exporting)
        self.save_button.setEnabled(loaded and self.controller.result_is_current() and not self.controller.busy and not self._exporting)

    def save_results(self):
        if not self.controller.result_is_current() or self.controller.busy:
            self.status_label.setText('请等待当前参数处理完成后再保存')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('保存结果')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('保存原图、当前算法结果、同视角对比图和参数记录。'))
        note = QLabel('算法结果保持原始分辨率；对比图应用 ROI 与旋转。')
        note.setObjectName('muted')
        layout.addWidget(note)
        all_algorithms = QCheckBox('同时保存 8 种算法及算法对比图')
        all_algorithms.setEnabled(self.controller.comparisons_are_current())
        all_algorithms.setChecked(self.controller.comparisons_are_current())
        if not all_algorithms.isEnabled():
            all_algorithms.setToolTip('全部算法正在更新，完成后可导出完整对比')
        layout.addWidget(all_algorithms)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        directory = QFileDialog.getExistingDirectory(self, '选择导出目录', str(OUTPUT_DIR))
        if directory:
            self._exporting = True
            self.parameters.setEnabled(False)
            for widget in (self.navigation, self.upload_button, self.sample_button, self.comparison):
                widget.setEnabled(False)
            self._update_actions()
            self.controller.export(directory, all_algorithms.isChecked())

    def _end_export(self):
        self._exporting = False
        self.parameters.setEnabled(self.controller.state.original is not None)
        for widget in (self.navigation, self.upload_button, self.sample_button, self.comparison):
            widget.setEnabled(True)
        self._update_actions()

    def _on_exported(self, path):
        self._end_export()
        self.result_info.setText(f'导出完成：{path}')

    def _show_error(self, message):
        if self._exporting:
            self._end_export()
        QMessageBox.warning(self, '操作提示', message)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self._exporting:
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not self._exporting:
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    self.load_image(url.toLocalFile())
                    break
            event.acceptProposedAction()

    def closeEvent(self, event):
        self._stop_interaction()
        self.controller.shutdown()
        event.accept()
