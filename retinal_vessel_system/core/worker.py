# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from threading import Event

from PySide6.QtCore import QObject, QRunnable, Signal


class CancelledTask(Exception):
    pass


class WorkerSignals(QObject):
    finished = Signal(int, object, str)


class Worker(QRunnable):
    def __init__(self, token: int, function, event: Event):
        super().__init__()
        self.token, self.function, self.event = token, function, event
        self.signals = WorkerSignals()

    def check_cancelled(self):
        if self.event.is_set():
            raise CancelledTask()

    def run(self):
        try:
            self.check_cancelled()
            payload = self.function(self.check_cancelled)
            self.check_cancelled()
            self.signals.finished.emit(self.token, payload, "")
        except CancelledTask:
            self.signals.finished.emit(self.token, None, "")
        except Exception as exc:
            self.signals.finished.emit(self.token, None, f"{type(exc).__name__}: {exc}")

