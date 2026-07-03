"""JSONL rotating log, one record per line (design doc section 18)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path


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
