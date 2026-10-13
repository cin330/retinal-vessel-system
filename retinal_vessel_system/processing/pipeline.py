# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace
from time import perf_counter
from collections import OrderedDict

import cv2
import numpy as np

from retinal_vessel_system.config import COMPARISON_ALGORITHMS
from retinal_vessel_system.core.state import ProcessParams, Result
from retinal_vessel_system.processing.basic import algorithm_input, window_image
from retinal_vessel_system.processing.edges import detect_edges
from retinal_vessel_system.processing.enhancement import adaptive_median_filter, gaussian_filter
from retinal_vessel_system.processing.segmentation import otsu_segment, watershed_segment


def prepare_input(rgb: np.ndarray, params: ProcessParams, cancel=None) -> np.ndarray:
    gray = algorithm_input(rgb, params.input_channel)
    if params.denoise == "gaussian":
        gray = gaussian_filter(gray, params.gaussian_kernel, params.gaussian_sigma)
    elif params.denoise == "adaptive_median":
        gray = adaptive_median_filter(gray, params.adaptive_median_max_kernel, cancel)
    return gray


def run_algorithm(rgb: np.ndarray, params: ProcessParams, prepared=None, cancel=None) -> Result:
    started = perf_counter()
    if cancel:
        cancel()
    gray = prepare_input(rgb, params, cancel) if prepared is None else prepared
    metadata = {"input_channel": params.input_channel, "denoise": params.denoise}
    binary = False
    if params.algorithm == "preview":
        if params.denoise == "none":
            color = rgb
        elif params.denoise == "gaussian":
            color = gaussian_filter(rgb, params.gaussian_kernel, params.gaussian_sigma)
        else:
            # 自适应中值按单通道处理；色彩模式保留色度，替换亮度。
            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = gray
            color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = cv2.addWeighted(color, 1 - params.gray_mix / 100,
                               cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), params.gray_mix / 100, 0)
    elif params.algorithm == "otsu":
        image, info = otsu_segment(gray, params.otsu_invert)
        metadata.update(info)
        binary = True
    elif params.algorithm == "watershed":
        image, info = watershed_segment(gray, params.watershed_sigma, params.watershed_foreground,
                                       params.watershed_min_distance, cancel)
        metadata.update(info)
        binary = True
    else:
        image = detect_edges(gray, params.algorithm, params.edge_kernel, params.edge_threshold,
                             params.canny_low, params.canny_high)
        binary = params.algorithm == "canny" or params.edge_threshold > 0
    if cancel:
        cancel()
    image = np.ascontiguousarray(image)
    image.setflags(write=False)
    return Result(image, params.algorithm, params, (perf_counter() - started) * 1000, binary, metadata)


def compare_algorithms(rgb: np.ndarray, params: ProcessParams, cancel=None) -> dict[str, Result]:
    start = perf_counter()
    prepared = prepare_input(rgb, params, cancel)
    preparation_ms = (perf_counter() - start) * 1000
    results = {}
    for algorithm in COMPARISON_ALGORITHMS:
        if cancel:
            cancel()
        result = run_algorithm(rgb, replace(params, algorithm=algorithm), prepared, cancel)
        results[algorithm] = replace(result, metadata={**result.metadata, "shared_preprocessing_ms": round(preparation_ms, 3)})
    return results


def display_image(result: Result, params: ProcessParams) -> np.ndarray:
    return result.image if result.binary else window_image(result.image, params.window_width, params.window_level)


class ImageProcessor:
    """每个计算通道独立缓存输入，灰度调节复用滤波与灰度底图。"""
    def __init__(self, rgb):
        self.rgb = rgb
        self.inputs, self.colors = OrderedDict(), OrderedDict()

    @staticmethod
    def _remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 3:
            cache.popitem(last=False)

    def run(self, params, cancel=None, spatial_scale=1.0):
        started = perf_counter()
        key = params.preprocessing_key()
        cache_hit = key in self.inputs
        if cancel:
            cancel()
        if key not in self.inputs:
            self._remember(self.inputs, key, prepare_input(self.rgb, params, cancel))
        gray = self.inputs[key]
        def finished(result):
            return replace(result, elapsed_ms=(perf_counter() - started) * 1000,
                           metadata={**result.metadata, "input_channel": params.input_channel,
                                     "denoise": params.denoise, "preprocessing_cache_hit": cache_hit})
        if params.algorithm != "preview":
            if spatial_scale == 1 or params.algorithm != "watershed":
                return finished(run_algorithm(self.rgb, params, prepared=gray, cancel=cancel))
            image, info = watershed_segment(gray, params.watershed_sigma, params.watershed_foreground,
                                            params.watershed_min_distance, cancel, spatial_scale)
            if cancel:
                cancel()
            image.setflags(write=False)
            return finished(Result(image, "watershed", params, 0, True, info))
        if key not in self.colors:
            if params.denoise == "none":
                color = self.rgb
            elif params.denoise == "gaussian":
                color = gaussian_filter(self.rgb, params.gaussian_kernel, params.gaussian_sigma)
            else:
                ycrcb = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = gray
                color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            self._remember(self.colors, key, (color, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))
        color, gray_rgb = self.colors[key]
        image = cv2.addWeighted(color, 1 - params.gray_mix / 100, gray_rgb, params.gray_mix / 100, 0)
        if cancel:
            cancel()
        image.setflags(write=False)
        return finished(Result(image, "preview", params, 0))


def scaled_preview_params(params, scale):
    def odd(value, minimum=1):
        return max(minimum, (round(value * scale) // 2) * 2 + 1)
    return replace(params, gaussian_kernel=min(31, odd(params.gaussian_kernel)),
                   gaussian_sigma=params.gaussian_sigma * scale,
                   adaptive_median_max_kernel=min(15, odd(params.adaptive_median_max_kernel, 3)),
                   edge_kernel=min(7, odd(params.edge_kernel)),
                   watershed_sigma=params.watershed_sigma * scale,
                   watershed_min_distance=max(1, round(params.watershed_min_distance * scale)))

