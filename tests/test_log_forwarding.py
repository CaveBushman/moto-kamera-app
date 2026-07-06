import logging

from motocam.core.logging_setup import install_control_room_log_forwarder


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
