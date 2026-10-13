# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace

import cv2
import numpy as np
import pytest
from scipy.ndimage import median_filter

from retinal_vessel_system.config import COMPARISON_ALGORITHMS
from retinal_vessel_system.core.state import ProcessParams
from retinal_vessel_system.processing.basic import gray_preview, window_image
from retinal_vessel_system.processing.enhancement import adaptive_median_filter
from retinal_vessel_system.processing.pipeline import compare_algorithms, display_image, run_algorithm
from retinal_vessel_system.processing.pipeline import ImageProcessor
from retinal_vessel_system.processing.segmentation import otsu_segment, watershed_segment


@pytest.mark.parametrize("algorithm", COMPARISON_ALGORITHMS)
def test_algorithm_preserves_coordinates_and_original(fundus, algorithm):
    original = fundus.copy()
    result = run_algorithm(fundus, ProcessParams(algorithm=algorithm))
    assert result.image.shape == fundus.shape[:2]
    assert result.image.dtype == np.uint8
    assert not result.image.flags.writeable
    assert np.count_nonzero(result.image) > 0
    np.testing.assert_array_equal(fundus, original)


def test_otsu_threshold_and_inversion():
    gray = np.array([[20, 20, 20, 180, 180, 180]], np.uint8)
    mask, metadata = otsu_segment(gray)
    np.testing.assert_array_equal(mask, [[0, 0, 0, 255, 255, 255]])
    assert metadata["threshold"] == 20
    inverse, _ = otsu_segment(gray, True)
    np.testing.assert_array_equal(inverse, 255 - mask)


def test_adaptive_median_expands_window_and_preserves_non_noise():
    # 3×3 脉冲块必须扩窗；局部固定 3×3 中值无法修复中心。
    image = np.full((11, 11), 100, np.uint8)
    image[4:7, 4:7] = 255
    result = adaptive_median_filter(image, 7)
    assert result[5, 5] == 100
    assert median_filter(image, size=3)[5, 5] == 255
    assert result[1, 1] == 100
    rng = np.random.default_rng(9)
    noisy = rng.integers(0, 256, (7, 8), dtype=np.uint8)
    # 独立的逐像素算法核对全部两阶段语义与 reflect 边界。
    padded = np.pad(noisy, 3, mode="symmetric")
    expected = noisy.copy()
    for y in range(7):
        for x in range(8):
            center = int(noisy[y, x])
            for size in (3, 5, 7):
                radius = size // 2
                window = padded[y + 3 - radius:y + 4 + radius, x + 3 - radius:x + 4 + radius]
                lo, mid, hi = int(window.min()), int(np.median(window)), int(window.max())
                if lo < mid < hi:
                    expected[y, x] = center if lo < center < hi else mid
                    break
                if size == 7:
                    expected[y, x] = mid
    np.testing.assert_array_equal(adaptive_median_filter(noisy, 7), expected)


def test_gray_blending_and_window_do_not_change_algorithm_input(fundus):
    np.testing.assert_array_equal(gray_preview(fundus, 0), fundus)
    full_gray = gray_preview(fundus, 100)
    np.testing.assert_array_equal(full_gray[:, :, 0], full_gray[:, :, 2])
    params = ProcessParams(algorithm="otsu")
    changed = replace(params, gray_mix=70, window_width=80, window_level=190)
    original = run_algorithm(fundus, params)
    other = run_algorithm(fundus, changed)
    np.testing.assert_array_equal(original.image, other.image)
    np.testing.assert_array_equal(display_image(other, changed), other.image)
    assert params.computation_key() == changed.computation_key()
    np.testing.assert_array_equal(window_image(np.array([0, 50, 100, 150, 200], np.uint8), 100, 100), [0, 0, 127, 255, 255])


def test_batch_uses_fixed_preprocessing(fundus):
    params = ProcessParams(denoise="gaussian", input_channel="green", gaussian_kernel=7)
    results = compare_algorithms(fundus, params)
    assert tuple(results) == COMPARISON_ALGORITHMS
    for key, result in results.items():
        expected = run_algorithm(fundus, replace(params, algorithm=key))
        np.testing.assert_array_equal(result.image, expected.image)
        assert result.params.input_channel == "green"


def test_watershed_handles_uniform_images_and_generates_markers(fundus):
    empty, metadata = watershed_segment(np.zeros((25, 30), np.uint8))
    assert not empty.any() and metadata["marker_count"] == 0
    gray = cv2.cvtColor(fundus, cv2.COLOR_RGB2GRAY)
    mask, metadata = watershed_segment(gray)
    assert metadata["marker_count"] > 0
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[42, 60] == 255  # 暗线附近产生前景
    _, fewer = watershed_segment(gray, foreground=99)
    assert fewer["contrast_threshold"] > metadata["contrast_threshold"]


@pytest.mark.parametrize("changes", [{"canny_low": 200, "canny_high": 100}, {"gaussian_kernel": 4},
                                    {"adaptive_median_max_kernel": 2}, {"window_width": 0},
                                    {"algorithm": "unknown"}])
def test_invalid_parameters_fail_clearly(changes):
    with pytest.raises(ValueError):
        ProcessParams(**changes)


def test_gray_drag_reuses_preprocessed_input(monkeypatch, fundus):
    from retinal_vessel_system.processing import pipeline
    calls = []
    prepare = pipeline.prepare_input
    def record(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(pipeline, "prepare_input", record)
    processor = ImageProcessor(fundus)
    for value in (10, 35, 60, 85):
        result = processor.run(ProcessParams(gray_mix=value, denoise="adaptive_median"))
        expected = run_algorithm(fundus, ProcessParams(gray_mix=value, denoise="adaptive_median"))
        np.testing.assert_array_equal(result.image, expected.image)
    # 四次独立参考计算加一次缓存输入构建。
    assert len(calls) == 5

