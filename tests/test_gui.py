# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace
from threading import Event
from time import sleep

import numpy as np
import cv2
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QProgressBar, QScrollArea

from retinal_vessel_system.core.state import ProcessParams
from retinal_vessel_system.gui.main_window import MainWindow
from retinal_vessel_system.gui.parameter_panel import ParameterPanel
from retinal_vessel_system.processing.pipeline import run_algorithm
from retinal_vessel_system.utils.image_io import save_image, read_image
from conftest import wait_until


@pytest.fixture
def window(app, tmp_path, fundus):
    window = MainWindow()
    window.show()
    app.processEvents()
    path = tmp_path / "测试眼底.png"
    save_image(path, fundus)
    window.load_image(path)
    wait_until(app, lambda: window.controller.state.current is not None and not window.controller.busy)
    yield window
    window.close()
    app.processEvents()


def test_no_image_disables_actions_and_loading_preserves_original(app):
    window = MainWindow()
    assert not window.execute_button.isEnabled()
    assert not window.save_button.isEnabled()
    assert not window.parameters.isEnabled()
    window.close()


def test_shared_geometry_and_crop_in_both_views(app, window):
    window.sync.set_zoom(180)
    window.sync.set_rotation(35)
    window.sync.pan_by(13, -7)
    app.processEvents()
    assert window.left_view.transform() == window.right_view.transform()
    point = QPoint(65, 40)
    left_position = window.left_view.mapFromScene(point.x(), point.y())
    right_position = window.right_view.mapFromScene(point.x(), point.y())
    # 双视口中心可能因奇数宽度有一像素取整误差，不能产生累计偏移。
    left_relative = left_position - window.left_view.viewport().rect().center()
    right_relative = right_position - window.right_view.viewport().rect().center()
    assert abs(left_relative.x() - right_relative.x()) <= 1
    assert abs(left_relative.y() - right_relative.y()) <= 1
    window.sync.set_roi((15, 10, 70, 60))
    assert window.left_view.item.pos() == window.right_view.item.pos()
    assert window.left_view.item.pixmap().size() == window.right_view.item.pixmap().size()
    assert window.parameters.view_rows["right"].value() == 43
    original = window.controller.state.original.copy()
    window.sync.reset()
    assert window.controller.view.roi is None
    np.testing.assert_array_equal(window.controller.state.original, original)


def test_mouse_pan_and_roi_with_rotated_view(app, window):
    window.sync.set_rotation(25)
    viewport = window.left_view.viewport()
    center = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(viewport, center + QPoint(18, 11))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=center + QPoint(18, 11))
    assert window.controller.view.pan_x != 0
    assert window.left_view.transform() == window.right_view.transform()
    window.sync.fit()
    window.crop_button.setChecked(True)
    window._toggle_crop(True)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=center - QPoint(30, 30))
    QTest.mouseMove(viewport, center + QPoint(30, 30))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=center + QPoint(30, 30))
    roi = window.controller.view.roi
    assert roi is not None
    x, y, w, h = roi
    assert 0 <= x < x + w <= 128 and 0 <= y < y + h <= 96
    assert window.left_view.item.pos() == window.right_view.item.pos()


def test_canny_and_odd_kernel_slider_constraints(app):
    panel = ParameterPanel()
    panel.set_params(ProcessParams(algorithm="canny"))
    panel.rows["canny_low"].slider.setValue(220)
    assert panel.params.canny_high == 221
    panel.rows["canny_high"].slider.setValue(1)
    assert panel.params.canny_low < panel.params.canny_high
    panel.rows["gaussian_kernel"].slider.setValue(7)
    assert panel.params.gaussian_kernel == 15
    panel.deleteLater()


def test_batch_thumbnail_switch_and_default_restore(app, window):
    window.batch_compare()
    wait_until(app, lambda: window.controller.comparisons_are_current() and not window.controller.busy)
    assert all(card.isEnabled() for card in window.comparison.cards.values())
    assert window.controller.result_is_current()  # 预览下批量对比同样可保存
    window.comparison.cards["otsu"].click()
    assert window.controller.params.algorithm == "otsu"
    assert "Otsu" in window.right_title.text()
    assert "阈值" in window.result_info.text()
    window.reset_all()
    wait_until(app, lambda: not window.controller.busy and window.controller.result_is_current())
    assert window.controller.params == ProcessParams()
    wait_until(app, window.controller.comparisons_are_current)
    assert len(window.controller.state.results) == 8


def test_late_background_result_cannot_overwrite_new_parameters(app, window):
    controller = window.controller
    controller.invalidate()
    wait_until(app, lambda: controller._full_active is None)
    rgb = controller.state.original
    entered = Event()
    old = ProcessParams(algorithm="otsu", otsu_invert=True)
    controller.update_params(old)
    def slow_task(cancel):
        entered.set()
        sleep(0.08)
        # 即使第三方算法无法及时取消，任务序号仍拒绝旧结果。
        return run_algorithm(rgb, old)
    controller._full_active = controller._start_job("process", slow_task, old, controller._generation)
    wait_until(app, entered.is_set)
    new = replace(old, algorithm="canny", canny_low=75)
    controller.update_params(new)
    controller.request_processing()
    wait_until(app, lambda: not controller.busy and controller.result_is_current())
    assert controller.state.current.algorithm == "canny"
    assert controller.state.current.params.canny_low == 75
    if "otsu" in controller.state.results:
        assert not controller.state.results["otsu"].params.otsu_invert


def test_changed_preprocessing_invalidates_old_batch(app, window):
    window.batch_compare()
    wait_until(app, lambda: not window.controller.busy and window.controller.comparisons_are_current())
    previous = dict(window.controller.state.results)
    window._on_params(replace(window.controller.params, denoise="gaussian", gaussian_kernel=9))
    assert all(window.controller.state.results[k] is value for k, value in previous.items())
    assert not window.controller.comparisons_are_current()
    assert all(card.detail.text() == "待更新" for card in window.comparison.cards.values())
    assert not window.save_button.isEnabled()
    wait_until(app, lambda: not window._settle.isActive() and not window.controller.busy and window.controller.result_is_current())
    assert window.controller.state.current.params.gaussian_kernel == 9


@pytest.mark.parametrize("size", [(1200, 760), (1440, 900)])
def test_navigation_and_all_cards_fit_without_scrollbars(app, window, size):
    window.resize(*size)
    app.processEvents()
    assert window.size().width() == size[0] and window.size().height() == size[1]
    assert not window.findChildren(QProgressBar)
    assert not hasattr(window.parameters, "algorithm_combo")
    assert not window.navigation.findChildren(QScrollArea)
    assert not window.comparison.findChildren(QScrollArea)
    for button in window.algorithm_buttons.values():
        assert button.isVisible()
        assert window.navigation.rect().contains(button.geometry())
    for card in window.comparison.cards.values():
        assert card.isVisible()
        assert window.comparison.rect().contains(card.geometry())
        assert card.thumbnail.geometry().bottom() < card.title.geometry().top()
        assert card.title.geometry().bottom() < card.detail.geometry().top()


def test_algorithm_parameters_survive_navigation_and_thumbnail_selection(app, window):
    wait_until(app, window.controller.comparisons_are_current)
    window.select_algorithm("canny")
    window.parameters._set_param("canny_low", 73)
    window.parameters._set_param("denoise", "gaussian")
    window.select_algorithm("prewitt")
    window.parameters._set_param("edge_threshold", 92)
    window.select_algorithm("sobel")
    assert window.controller.params.edge_threshold == 0
    window.comparison.cards["prewitt"].click()
    assert window.controller.params.edge_threshold == 92
    window.comparison.cards["canny"].click()
    assert window.controller.params.canny_low == 73
    assert window.controller.params.denoise == "gaussian"
    assert window.algorithm_buttons["canny"].isChecked()
    assert "Canny" in window.parameters.algorithm_title.text()
    wait_until(app, window.controller.result_is_current)


def test_gray_drag_shows_frames_before_release_and_saves_exact_final(app, window):
    previews = []
    window.controller.preview_ready.connect(previews.append)
    window._interaction_started("gray_mix")
    window.parameters.rows["gray_mix"].slider.setValue(35)
    wait_until(app, lambda: len(previews) >= 1)
    window.parameters.rows["gray_mix"].slider.setValue(80)
    wait_until(app, lambda: len(previews) >= 2)
    assert window.controller.dragging
    assert not window.controller.result_is_current()
    assert window.parameters.rows["gray_mix"].value() == 80
    assert window.right_view.logical_size == window.left_view.logical_size
    window._interaction_finished("gray_mix")
    wait_until(app, window.controller.result_is_current)
    expected = run_algorithm(window.controller.state.original, window.controller.params)
    np.testing.assert_array_equal(window.controller.state.current.image, expected.image)
    assert not window.controller.state.current.metadata.get("preview")


def test_returning_slider_to_cached_value_replaces_intermediate_preview(app, window):
    original_result = window.controller.state.current
    window._interaction_started("gray_mix")
    window.parameters.rows["gray_mix"].slider.setValue(65)
    wait_until(app, lambda: window._displayed_result.metadata.get("preview"))
    window.parameters.rows["gray_mix"].slider.setValue(0)
    window._interaction_finished("gray_mix")
    assert window.controller.state.current is original_result
    assert window._displayed_result is original_result
    np.testing.assert_array_equal(window.right_view.image, original_result.image)


def test_display_adjustment_preserves_both_view_transforms_and_centers(app, window):
    window.sync.set_zoom(160)
    window.sync.set_rotation(23)
    window.sync.pan_by(17, -9)
    app.processEvents()
    transforms = [view.transform() for view in window.sync.views]
    centers = [view.mapToScene(view.viewport().rect().center()) for view in window.sync.views]
    result = window.controller.state.current
    window.parameters._set_param("window_width", 130)
    wait_until(app, lambda: not window._display_timer.isActive())
    assert window.controller.state.current is result
    for index, view in enumerate(window.sync.views):
        assert view.transform() == transforms[index]
        assert view.mapToScene(view.viewport().rect().center()) == centers[index]


def test_rotation_does_not_change_scale_during_drag(app, window):
    initial = window.left_view.transform()
    scale = np.hypot(initial.m11(), initial.m12())
    window._interaction_started("rotation")
    window.sync.set_rotation(37)
    transform = window.left_view.transform()
    assert np.hypot(transform.m11(), transform.m12()) == pytest.approx(scale)
    window._interaction_finished("rotation")
    window.sync.set_rotation(60)
    transform = window.left_view.transform()
    assert np.hypot(transform.m11(), transform.m12()) == pytest.approx(scale)
    assert window.left_view.transform() == window.right_view.transform()


def test_preview_work_is_bounded_to_one_running_and_latest_pending(app, window):
    controller = window.controller
    entered = Event()
    original_run = controller.preview_processor.run
    def slow_run(*args, **kwargs):
        entered.set()
        sleep(0.05)
        return original_run(*args, **kwargs)
    controller.preview_processor.run = slow_run
    controller.begin_drag()
    controller.request_preview()
    wait_until(app, entered.is_set)
    for value in range(10, 90):
        controller.update_params(replace(controller.params, gray_mix=value))
        controller.request_preview()
    assert sum(job["kind"] == "preview" for job in controller._jobs.values()) == 1
    assert controller._preview_pending[0].gray_mix == 89
    controller.end_drag()
    wait_until(app, controller.result_is_current)
    assert controller.state.current.params.gray_mix == 89


def test_upload_automatically_populates_all_eight_incrementally(app, window):
    sizes = []
    window.controller.comparisons_ready.connect(lambda results: sizes.append(len(results)))
    window.reset_all()
    wait_until(app, window.controller.comparisons_are_current)
    assert any(0 < size < 8 for size in sizes)
    assert sizes[-1] == 8


def test_small_preview_keeps_original_roi_coordinates_and_cannot_be_exported(app, window, tmp_path, fundus):
    large = cv2.resize(fundus, (960, 720))
    source = tmp_path / "大图.png"
    save_image(source, large)
    window.load_image(source)
    wait_until(app, window.controller.result_is_current)
    window.sync.set_roi((120, 80, 600, 500))
    centers = [view.mapToScene(view.viewport().rect().center()) for view in window.sync.views]
    window._interaction_started("gray_mix")
    window.parameters.rows["gray_mix"].slider.setValue(67)
    wait_until(app, lambda: window._displayed_result.metadata.get("preview"))
    assert window.right_view.image.shape[1] == 640
    assert window.right_view.logical_size == (960, 720)
    assert window.right_view.clip.rect() == window.left_view.clip.rect()
    projected = window.right_view.item.mapToScene(320, 240)
    assert projected.x() == pytest.approx(480) and projected.y() == pytest.approx(360)
    assert [view.mapToScene(view.viewport().rect().center()) for view in window.sync.views] == centers
    controller = window.controller
    controller.error.disconnect(window._show_error)
    errors = []
    controller.error.connect(errors.append)
    controller.export(tmp_path / "导出")
    assert errors and "原始分辨率" in errors[-1]
    assert not (tmp_path / "导出").exists()
    window._interaction_finished("gray_mix")
    wait_until(app, controller.result_is_current)
    assert controller.state.current.image.shape == large.shape
    exported = []
    controller.exported.connect(exported.append)
    controller.export(tmp_path / "导出")
    wait_until(app, lambda: bool(exported))
    from pathlib import Path
    raw = read_image(Path(exported[-1]) / "processed_current.png")
    np.testing.assert_array_equal(raw, controller.state.current.image)

