"""JSONL rotating log, one record per line (design doc section 18)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("motocam")
    root.setLevel(level)
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "motocam.jsonl", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonlFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console_handler)

    return root


class ControlRoomLogHandler(logging.Handler):
    def __init__(self, link: Any, level: int = logging.NOTSET):
        super().__init__(level)
        self.link = link
        self._exception_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self._exception_formatter.formatException(record.exc_info)}"
            if self.link is None:
                return
            self.link.send_log_event(record.levelname, message, record.name)
        except Exception:
            # Logging handlers must never destabilize the live moto app.
            # Swallow the error and keep going so startup / Hailo testing can continue.
            try:
                self.handleError(record)
            except Exception:
                pass


class StartupLogBuffer(logging.Handler):
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.records: list[tuple[str, str, str]] = []
        self._exception_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self._exception_formatter.formatException(record.exc_info)}"
            self.records.append((record.levelname, message, record.name))
        except Exception:
            self.handleError(record)

    def replay_to(self, link: Any) -> None:
        for level, message, module in self.records:
            link.send_log_event(level, message, module)


def install_control_room_log_forwarder(logger: logging.Logger, link: Any) -> ControlRoomLogHandler:
    for handler in logger.handlers:
        if isinstance(handler, ControlRoomLogHandler):
            handler.link = link
            return handler
    handler = ControlRoomLogHandler(link)
    logger.addHandler(handler)
    return handler


def install_crash_guard(logger: logging.Logger) -> None:
    """PyQt6 aborts the whole process (SIGABRT) if a Python exception
    escapes a slot invoked from C++ -- e.g. any QTimer.timeout callback,
    which is most of this app's control loop. That's not acceptable for a
    live control unit: one bad GPS sentence or a None frame shouldn't kill
    the whole camera/gimbal session mid-ride. Installing our own
    sys.excepthook makes PyQt6 log-and-continue instead of aborting.
    """

    def _excepthook(exc_type, exc_value, exc_tb) -> None:
        logger.error("Unhandled exception in Qt slot", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook
