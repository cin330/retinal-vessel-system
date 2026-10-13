# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""在真实示例图上验证算法、异步 GUI、导出，生成可复核的本地记录。"""
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import PySide6
import scipy
import skimage
from PySide6.QtWidgets import QApplication

from retinal_vessel_system.config import STYLE
from retinal_vessel_system.config import COMPARISON_ALGORITHMS
from retinal_vessel_system.core.state import ProcessParams
from retinal_vessel_system.gui.main_window import MainWindow
from retinal_vessel_system.processing.pipeline import run_algorithm
from retinal_vessel_system.utils.export import export_results


def wait(application, predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("真实示例验证超时")


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    errors = []
    # 验证失败应抛出异常，不能被模态对话框掩盖。
    window.controller.error.disconnect(window._show_error)
    window.controller.error.connect(errors.append)
    window.show()
    try:
        window.load_example()
        wait(app, lambda: bool(errors) or (window.controller.result_is_current() and not window.controller.busy))
        if errors:
            raise RuntimeError(errors[-1])
        original = window.controller.state.original
        unchanged = original.copy()
        frames = []
        window.controller.preview_ready.connect(lambda result: frames.append(time.monotonic()))
        window._interaction_started("gray_mix")
        drag_started = time.monotonic()
        for value in range(2, 100, 2):
            window.parameters.rows["gray_mix"].slider.setValue(value)
            deadline = time.monotonic() + 0.02
            while time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.002)
        frames_before_release = len(frames)
        window._interaction_finished("gray_mix")
        wait(app, lambda: window.controller.result_is_current())
        expected = run_algorithm(original, window.controller.params)
        np.testing.assert_array_equal(window.controller.state.current.image, expected.image)
        interaction = {"drag_duration_ms": round((time.monotonic() - drag_started) * 1000, 2),
                       "preview_frames_before_release": frames_before_release,
                       "first_preview_latency_ms": round((frames[0] - drag_started) * 1000, 2) if frames else None,
                       "median_preview_interval_ms": round(float(np.median(np.diff(frames))) * 1000, 2) if len(frames) > 1 else None,
                       "final_gray_mix": window.controller.params.gray_mix,
                       "final_matches_full_resolution_reference": True}
        assert frames_before_release >= 5
        enhancement = {}
        for method in ("gaussian", "adaptive_median"):
            result = run_algorithm(original, ProcessParams(denoise=method, gray_mix=100))
            enhancement[method] = {"elapsed_ms": round(result.elapsed_ms, 2), "shape": list(result.image.shape)}
        window.select_algorithm("watershed")
        window.parameters.set_params(replace(window.controller.params, input_channel="green", denoise="gaussian"))
        window._on_params(window.parameters.params)
        window.batch_compare()
        wait(app, lambda: bool(errors) or (window.controller.comparisons_are_current() and not window.controller.busy))
        if errors:
            raise RuntimeError(errors[-1])
        results = {k: window.controller.state.results[k] for k in COMPARISON_ALGORITHMS}
        assert len(results) == 8
        for key, result in results.items():
            assert result.image.shape == original.shape[:2], key
            assert result.image.dtype == np.uint8, key
            assert result.image.any(), key
        np.testing.assert_array_equal(original, unchanged)
        docs = ROOT / "docs"
        docs.mkdir(exist_ok=True)
        app.processEvents()
        assert window.grab().save(str(docs / "ui-preview.png"))
        window.resize(1200, 760)
        app.processEvents()
        assert window.size().width() == 1200 and window.size().height() == 760
        assert window.grab().save(str(docs / "ui-minimum.png"))
        window.resize(1440, 900)
        app.processEvents()
        # 当前掩膜保持原始尺寸；对比图采用共同 ROI 与旋转。
        window.sync.set_roi((120, 120, 1100, 1100))
        window.sync.set_rotation(15)
        folder = export_results(ROOT / "outputs", original, window.controller.state.source_path,
                                window.controller.state.current, window.controller.params,
                                window.controller.view, results)
        versions = {"python": sys.version.split()[0], "PySide6": PySide6.__version__,
                    "opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                    "scikit-image": skimage.__version__}
        report = {"validated_at": datetime.now().astimezone().isoformat(), "environment": "yolo",
                  "versions": versions, "sample": "data/retina.jpg", "image_shape": list(original.shape),
                  "enhancement": enhancement,
                  "interaction": interaction,
                  "algorithms": {key: {**result.description(), "nonzero_pixels": int(np.count_nonzero(result.image))}
                                 for key, result in results.items()},
                  "original_unchanged": True, "export_directory": str(folder.relative_to(ROOT)),
                  "screenshot": "docs/ui-preview.png"}
        (docs / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "versions": versions, "enhancement": enhancement, "interaction": interaction,
                          "export": str(folder), "screenshot": str(docs / "ui-preview.png")}, ensure_ascii=False, indent=2))
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
