# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace
from pathlib import Path
from threading import Event

import cv2
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from retinal_vessel_system.config import ALGORITHM_NAMES, COMPARISON_ALGORITHMS
from retinal_vessel_system.core.state import ALGORITHM_FIELDS, ImageState, ProcessParams, ViewState
from retinal_vessel_system.core.worker import Worker
from retinal_vessel_system.processing.pipeline import ImageProcessor, scaled_preview_params
from retinal_vessel_system.utils.export import export_results
from retinal_vessel_system.utils.image_io import read_image


class Controller(QObject):
    image_loaded = Signal()
    result_ready = Signal(object)
    preview_ready = Signal(object)
    comparisons_ready = Signal(object)
    busy_changed = Signal(bool)
    status = Signal(str)
    error = Signal(str)
    exported = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state, self.params, self.view = ImageState(), ProcessParams(), ViewState()
        self.algorithm_settings = {k: ProcessParams(algorithm=k) for k in ALGORITHM_NAMES}
        self.pool, self.preview_pool = QThreadPool(self), QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.preview_pool.setMaxThreadCount(1)
        self._jobs = {}
        self._serial = self._generation = self._preview_epoch = 0
        self._full_active = self._preview_active = None
        self._full_pending = self._preview_pending = None
        self._compare_requested = False
        self._export_pending = None
        self.dragging = self.busy = False
        self.processor = self.preview_processor = None

    def params_for(self, algorithm):
        saved = self.algorithm_settings[algorithm]
        return replace(self.params, algorithm=algorithm,
                       **{f: getattr(saved, f) for f in ALGORITHM_FIELDS[algorithm]})

    def comparison_is_current(self, algorithm):
        result = self.state.results.get(algorithm)
        return result is not None and result.params.computation_key() == self.params_for(algorithm).computation_key()

    def result_is_current(self):
        result = self.state.current
        return result is not None and result.params.computation_key() == self.params.computation_key()

    def comparisons_are_current(self):
        return all(self.comparison_is_current(k) for k in COMPARISON_ALGORITHMS)

    def _set_busy(self):
        busy = self._export_pending is not None or any(j['kind'] == 'export' for j in self._jobs.values())
        busy = busy or (self.state.original is not None and not self.result_is_current())
        if busy != self.busy:
            self.busy = busy
            self.busy_changed.emit(busy)

    def _cancel(self, token):
        if token in self._jobs:
            self._jobs[token]['event'].set()

    def invalidate(self):
        for job in self._jobs.values():
            job['event'].set()
        self._preview_epoch += 1
        self._full_pending = self._preview_pending = None
        self._compare_requested = False

    def load_image(self, path):
        try:
            rgb = read_image(path)
        except (ValueError, OSError) as exc:
            self.error.emit(str(exc))
            return False
        self.invalidate()
        self._generation += 1
        self.dragging = False
        self.state.load(rgb, Path(path).resolve())
        self.params = ProcessParams()
        self.algorithm_settings = {k: ProcessParams(algorithm=k) for k in ALGORITHM_NAMES}
        self.processor = ImageProcessor(self.state.original)
        h, w = rgb.shape[:2]
        self.preview_scale = min(1.0, 640 / max(h, w))
        small = cv2.resize(rgb, (max(1, round(w * self.preview_scale)), max(1, round(h * self.preview_scale))),
                           interpolation=cv2.INTER_AREA) if self.preview_scale < 1 else rgb
        self.preview_processor = ImageProcessor(small)
        self.image_loaded.emit()
        self.request_processing(batch=True)
        return True

    def update_params(self, params):
        changed = self.params.computation_key() != params.computation_key()
        self.params = params
        self.algorithm_settings[params.algorithm] = params
        if changed:
            self._cancel(self._full_active)
            self._full_pending = None
        self._set_busy()
        return changed

    def set_algorithm(self, algorithm):
        if algorithm not in ALGORITHM_NAMES or self.state.original is None:
            return
        self.end_drag(schedule=False)
        self.algorithm_settings[self.params.algorithm] = self.params
        self.update_params(self.params_for(algorithm))
        self._preview_epoch += 1
        self._cancel(self._preview_active)
        self._preview_pending = None
        self.request_processing(batch=True)

    def select_result(self, algorithm):
        self.set_algorithm(algorithm)

    def begin_drag(self):
        self.dragging = True
        self._preview_epoch += 1
        self._cancel(self._full_active)
        self._full_pending = None

    def end_drag(self, schedule=True):
        self.dragging = False
        self._preview_epoch += 1
        self._cancel(self._preview_active)
        self._preview_pending = None
        if schedule and self.state.original is not None:
            self.request_processing(batch=True)

    def request_preview(self):
        if self.state.original is None:
            return
        self._preview_pending = (self.params, self._preview_epoch, self._generation)
        self._start_preview()

    def _start_preview(self):
        if self._preview_active is not None or self._preview_pending is None:
            return
        params, epoch, generation = self._preview_pending
        self._preview_pending = None
        processor, scale = self.preview_processor, self.preview_scale
        source_size = [self.state.original.shape[1], self.state.original.shape[0]]
        adapted = scaled_preview_params(params, scale)
        def compute(cancel):
            result = processor.run(adapted, cancel, scale)
            return replace(result, params=params, metadata={**result.metadata, 'preview': True,
                           'source_size': source_size})
        self._preview_active = self._start_job('preview', compute, params, generation, epoch, self.preview_pool)

    def request_processing(self, batch=False):
        if self.state.original is None:
            self.error.emit('请先上传图像')
            return
        self._compare_requested = self._compare_requested or batch
        cached = self.state.results.get(self.params.algorithm)
        if cached and cached.params.computation_key() == self.params.computation_key():
            self.state.current = cached
            self.result_ready.emit(cached)
        elif self.result_is_current():
            # 拖回已有完整结果的参数值时，必须替换屏幕上的中间预览。
            self.result_ready.emit(self.state.current)
        elif not self.result_is_current():
            self._full_pending = self.params
            self._cancel(self._full_active)
            self.status.emit(f'{ALGORITHM_NAMES[self.params.algorithm]} · 更新中')
        self._set_busy()
        self._start_full()

    def _start_job(self, kind, function, params=None, generation=None, epoch=None, pool=None):
        self._serial += 1
        token, event = self._serial, Event()
        worker = Worker(token, function, event)
        worker.signals.finished.connect(self._finished)
        self._jobs[token] = {'kind': kind, 'params': params, 'generation': generation,
                             'epoch': epoch, 'event': event, 'worker': worker}
        (pool or self.pool).start(worker)
        return token

    def _start_full(self):
        if self._full_active is not None or self.dragging or self.processor is None:
            return
        if self._export_pending is not None:
            function = self._export_pending
            self._export_pending = None
            self._full_active = self._start_job('export', function, generation=self._generation)
            return
        params = self._full_pending
        self._full_pending = None
        kind = 'process'
        if params is None and self._compare_requested:
            order = [k for k in COMPARISON_ALGORITHMS if k != 'watershed'] + ['watershed']
            missing = [k for k in order if not self.comparison_is_current(k)]
            if missing:
                params, kind = self.params_for(missing[0]), 'comparison'
            else:
                self._compare_requested = False
                self.status.emit('全部算法已更新')
        if params is not None:
            processor = self.processor
            self._full_active = self._start_job(kind, lambda cancel: processor.run(params, cancel),
                                                params, self._generation)

    def export(self, directory, include_all=False):
        if not self.result_is_current():
            self.error.emit('当前参数尚未完成原始分辨率计算，请稍后保存')
            return
        if include_all and not self.comparisons_are_current():
            self.error.emit('全部算法正在更新，请完成后再保存完整对比')
            return
        rgb, source, current = self.state.original, self.state.source_path, self.state.current
        params, view = self.params, replace(self.view)
        results = {k: self.state.results[k] for k in COMPARISON_ALGORITHMS} if include_all else None
        self._export_pending = lambda cancel: export_results(directory, rgb, source, current, params, view, results, cancel)
        self._cancel(self._full_active)
        self._set_busy()
        self._start_full()

    @Slot(int, object, str)
    def _finished(self, token, payload, error):
        job = self._jobs.pop(token, None)
        if job is None:
            return
        if token == self._full_active:
            self._full_active = None
        if token == self._preview_active:
            self._preview_active = None
        valid = not job['event'].is_set() and job['generation'] == self._generation
        kind = job['kind']
        if valid and error:
            self.error.emit(error)
            if kind == 'comparison':
                self._compare_requested = False
        if valid and payload is not None:
            if kind == 'preview':
                if self.dragging and job['epoch'] == self._preview_epoch and payload.algorithm == self.params.algorithm:
                    self.preview_ready.emit(payload)
            elif kind == 'export':
                self.exported.emit(str(payload))
                self.status.emit(f'已保存：{payload}')
            else:
                wanted = self.params_for(payload.algorithm)
                if wanted.computation_key() == payload.params.computation_key():
                    if payload.algorithm != 'preview':
                        self.state.results[payload.algorithm] = payload
                        self.comparisons_ready.emit(dict(self.state.results))
                    if self.params.computation_key() == payload.params.computation_key():
                        self.state.current = payload
                        self.result_ready.emit(payload)
                        self.status.emit(f'{ALGORITHM_NAMES[payload.algorithm]} · {payload.elapsed_ms:.1f} ms')
        self._set_busy()
        self._start_preview()
        self._start_full()

    def reset(self):
        self.invalidate()
        self.dragging = False
        self.params = ProcessParams()
        self.algorithm_settings = {k: ProcessParams(algorithm=k) for k in ALGORITHM_NAMES}
        self.state.current = None
        self.state.results.clear()
        self.comparisons_ready.emit({})
        self.request_processing(batch=True)

    def shutdown(self):
        self.invalidate()
        self.pool.waitForDone()
        self.preview_pool.waitForDone()
