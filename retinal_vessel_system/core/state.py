# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retinal_vessel_system.config import ALGORITHM_NAMES

ALGORITHM_FIELDS = {
    "preview": ("gray_mix",), "otsu": ("otsu_invert",),
    "watershed": ("watershed_sigma", "watershed_foreground", "watershed_min_distance"),
    "canny": ("canny_low", "canny_high"),
    "sobel": ("edge_kernel", "edge_threshold"), "laplacian": ("edge_kernel", "edge_threshold"),
    "prewitt": ("edge_threshold",), "roberts": ("edge_threshold",), "scharr": ("edge_threshold",),
}


@dataclass
class ViewState:
    zoom: float = 1.0
    rotation_deg: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    roi: tuple[int, int, int, int] | None = None  # x, y, width, height，原图坐标
    fit: bool = True  # zoom 是相对适应窗口的倍率；False 时为像素倍率


@dataclass(frozen=True)
class ProcessParams:
    gray_mix: int = 0
    window_width: int = 255
    window_level: int = 128
    input_channel: str = "gray"
    denoise: str = "none"
    gaussian_kernel: int = 5
    gaussian_sigma: float = 0.0
    adaptive_median_max_kernel: int = 7
    otsu_invert: bool = False
    watershed_sigma: float = 1.5
    watershed_foreground: int = 35
    watershed_min_distance: int = 12
    edge_kernel: int = 3
    edge_threshold: int = 0
    canny_low: int = 50
    canny_high: int = 150
    algorithm: str = "preview"

    def __post_init__(self):
        limits = {
            "gray_mix": (0, 100), "window_width": (1, 255), "window_level": (0, 255),
            "gaussian_kernel": (1, 31), "gaussian_sigma": (0, 10),
            "adaptive_median_max_kernel": (3, 15), "watershed_sigma": (0, 5),
            "watershed_foreground": (1, 99), "watershed_min_distance": (1, 50),
            "edge_kernel": (1, 7), "edge_threshold": (0, 255),
            "canny_low": (0, 254), "canny_high": (1, 255),
        }
        for name, (lo, hi) in limits.items():
            value = getattr(self, name)
            if not lo <= value <= hi:
                raise ValueError(f"{name} 应在 {lo}–{hi} 范围内")
            if name not in ("gaussian_sigma", "watershed_sigma") and not isinstance(value, int):
                raise ValueError(f"{name} 必须为整数")
        for name in ("gaussian_kernel", "adaptive_median_max_kernel", "edge_kernel"):
            if getattr(self, name) % 2 != 1:
                raise ValueError(f"{name} 必须为奇数")
        if self.canny_high <= self.canny_low:
            raise ValueError("Canny 高阈值必须大于低阈值")
        if self.algorithm not in ALGORITHM_NAMES:
            raise ValueError("未知算法")
        if self.input_channel not in ("gray", "green") or self.denoise not in ("none", "gaussian", "adaptive_median"):
            raise ValueError("未知输入通道或去噪方法")

    def computation_key(self, algorithm: str | None = None) -> tuple:
        """仅包含真正影响该算法像素的参数，窗宽/窗位不进入算法缓存键。"""
        algorithm = algorithm or self.algorithm
        shared = [algorithm, self.input_channel, self.denoise]
        if self.denoise == "gaussian":
            shared += [self.gaussian_kernel, self.gaussian_sigma]
        elif self.denoise == "adaptive_median":
            shared += [self.adaptive_median_max_kernel]
        relevant = {
            "preview": ("gray_mix",), "otsu": ("otsu_invert",),
            "watershed": ("watershed_sigma", "watershed_foreground", "watershed_min_distance"),
            "canny": ("canny_low", "canny_high"),
            "sobel": ("edge_kernel", "edge_threshold"),
            "laplacian": ("edge_kernel", "edge_threshold"),
            "prewitt": ("edge_threshold",), "roberts": ("edge_threshold",),
            "scharr": ("edge_threshold",),
        }
        return tuple(shared + [getattr(self, k) for k in relevant[algorithm]])

    def preprocessing_key(self) -> tuple:
        return self.computation_key("otsu")[:-1][1:]


@dataclass(frozen=True)
class Result:
    image: np.ndarray
    algorithm: str
    params: ProcessParams
    elapsed_ms: float
    binary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def description(self) -> dict:
        return {"algorithm": self.algorithm, "parameters": asdict(self.params),
                "elapsed_ms": round(self.elapsed_ms, 3), "binary": self.binary,
                "metadata": self.metadata}


@dataclass
class ImageState:
    original: np.ndarray | None = None  # 只读 RGB uint8
    source_path: Path | None = None
    current: Result | None = None
    results: dict[str, Result] = field(default_factory=dict)

    def load(self, image: np.ndarray, path: Path):
        original = np.ascontiguousarray(image).copy()
        original.setflags(write=False)
        self.original, self.source_path = original, path
        self.current = None
        self.results.clear()

