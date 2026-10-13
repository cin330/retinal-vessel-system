# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from retinal_vessel_system.core.state import ProcessParams, Result, ViewState
from retinal_vessel_system.processing.pipeline import display_image
from retinal_vessel_system.utils.comparison import algorithm_grid, before_after
from retinal_vessel_system.utils.image_io import save_image


def export_results(directory: str | Path, original, source_path: Path,
                   current: Result, params: ProcessParams, view: ViewState, results=None, cancel=None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    created = datetime.now().astimezone()
    name = created.strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex[:6]
    target = directory / name
    # 全部文件写好后一次重命名，避免把不完整导出标记为成功。
    with TemporaryDirectory(prefix=".export-", dir=directory) as temp:
        folder = Path(temp) / name
        folder.mkdir()
        if cancel:
            cancel()
        save_image(folder / "original.png", original)
        save_image(folder / "processed_current.png", current.image)
        displayed = display_image(current, params)
        save_image(folder / "compare_before_after.png", before_after(original, displayed, view, current.binary))
        if results:
            (folder / "algorithms").mkdir()
            for key, result in results.items():
                if cancel:
                    cancel()
                save_image(folder / "algorithms" / f"{key}.png", result.image)
            save_image(folder / "compare_algorithms.png", algorithm_grid(original, results, view))
        metadata = {
            "version": "1.0.0", "created_at": created.isoformat(), "input_name": source_path.name,
            "input_path": str(source_path), "image_size": [original.shape[1], original.shape[0]],
            "parameters": asdict(params), "view": asdict(view), "current": current.description(),
            "algorithms": {k: r.description() for k, r in (results or {}).items()},
            "export_semantics": {"processed_current": "原始分辨率算法像素；不应用窗宽窗位和视图变换",
                                 "compare_before_after": "共享 ROI 与顺时针旋转；右侧应用显示窗宽窗位，二值掩膜除外",
                                 "compare_algorithms": "固定公共预处理，原图加算法原始结果；共享 ROI 与旋转",
                                 "zoom_pan": "仅影响 GUI 观察，不改变导出尺寸"},
        }
        (folder / "parameters.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if cancel:
            cancel()
        folder.rename(target)
    return target

