import logging

from motocam.core.logging_setup import StartupLogBuffer, install_control_room_log_forwarder


class FakeLink:
    def __init__(self):
        self.events: list[tuple[str, str, str]] = []

    def send_log_event(self, level: str, message: str, module: str = "") -> None:
        self.events.append((level, message, module))


def test_motocam_logs_are_forwarded_to_control_room():
    link = FakeLink()
    logger = logging.getLogger("motocam.test.forward")
    root = logging.getLogger("motocam")
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.propagate = False
    try:
        install_control_room_log_forwarder(root, link)
        logger.warning("BLE disconnected")
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)
        root.setLevel(old_level)
        root.propagate = old_propagate

    assert link.events == [("WARNING", "BLE disconnected", "motocam.test.forward")]


def test_startup_log_buffer_replays_early_logs_to_control_room():
    link = FakeLink()
    logger = logging.getLogger("motocam.test.startup")
    root = logging.getLogger("motocam")
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.propagate = False
    buffer = StartupLogBuffer()
    try:
        root.addHandler(buffer)
        logger.info("boot phase one")
        logger.warning("boot phase two")
        buffer.replay_to(link)
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)
        root.setLevel(old_level)
        root.propagate = old_propagate

    assert link.events == [
        ("INFO", "boot phase one", "motocam.test.startup"),
        ("WARNING", "boot phase two", "motocam.test.startup"),
    ]
