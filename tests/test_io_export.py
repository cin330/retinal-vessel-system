# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import json

import cv2
import numpy as np
import pytest

from retinal_vessel_system.core.state import ImageState, ProcessParams, ViewState
from retinal_vessel_system.processing.pipeline import compare_algorithms
from retinal_vessel_system.utils.comparison import view_orientation
from retinal_vessel_system.utils.export import export_results
from retinal_vessel_system.utils.image_io import read_image, save_image


def test_chinese_path_and_invalid_image(tmp_path, fundus):
    path = tmp_path / "眼底图像.png"
    save_image(path, fundus)
    np.testing.assert_array_equal(read_image(path), fundus)
    bad = tmp_path / "损坏.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="有效"):
        read_image(bad)
    state = ImageState()
    state.load(fundus, path)
    assert not state.original.flags.writeable
    fundus[:] = 0
    assert state.original.any()


def test_export_all_preserves_raw_result_and_common_view(tmp_path, fundus):
    params = ProcessParams(algorithm="canny", window_width=60)
    results = compare_algorithms(fundus, params)
    view = ViewState(zoom=3, rotation_deg=90, pan_x=10, pan_y=5, roi=(10, 15, 60, 40))
    folder = export_results(tmp_path, fundus, tmp_path / "输入.png", results["canny"], params, view, results)
    raw = cv2.imdecode(np.fromfile(folder / "processed_current.png", np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(raw, results["canny"].image)
    comparison = read_image(folder / "compare_before_after.png")
    assert comparison.shape == (100, 80, 3)  # 旋转后高 60，标题 40；宽 40×2
    assert len(list((folder / "algorithms").glob("*.png"))) == 8
    assert read_image(folder / "compare_algorithms.png").shape == (930, 1080, 3)
    metadata = json.loads((folder / "parameters.json").read_text())
    assert metadata["view"]["roi"] == [10, 15, 60, 40]
    assert metadata["parameters"]["canny_high"] == 150
    assert len(metadata["algorithms"]) == 8
    assert metadata["created_at"][-6] in ("+", "-")
    assert not list(tmp_path.glob(".export-*"))


def test_view_rotation_matches_clockwise_direction():
    image = np.zeros((10, 10), np.uint8)
    image[2, 5] = 255
    rotated = view_orientation(image, ViewState(rotation_deg=90), binary=True)
    assert rotated[5, 8] == 255

