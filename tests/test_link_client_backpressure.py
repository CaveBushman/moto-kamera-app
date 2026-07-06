import asyncio
import base64

from motocam.core.protocol import Envelope, GpsTelemetry, Telemetry
from motocam.network.link_client import LinkClient


class FakeWs:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_link_client_start_accepts_configured_loop_before_it_is_running():
    loop = asyncio.new_event_loop()
    previous_loop = None
    try:
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        asyncio.set_event_loop(loop)
        client = LinkClient("ws://control-room", "moto-1")
        client.start()

        assert client._loop is loop
        assert client._task is not None
        client._task.cancel()
    finally:
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
        asyncio.set_event_loop(previous_loop)


def test_link_client_rejects_invalid_control_room_urls():
    client = LinkClient("http://127.0.0.1:8765", "moto-1")

    try:
        client._validate_url()
    except ValueError as exc:
        assert "scheme" in str(exc)
    else:
        raise AssertionError("invalid scheme accepted")

    client.url = "ws://:8765"
    try:
        client._validate_url()
    except ValueError as exc:
        assert "host is empty" in str(exc)
    else:
        raise AssertionError("empty host accepted")

    client.url = "ws://127.0.0.1"
    try:
        client._validate_url()
    except ValueError as exc:
        assert "port is missing" in str(exc)
    else:
        raise AssertionError("missing port accepted")


def test_preview_send_keeps_only_latest_frame_under_backpressure():
    asyncio.run(_preview_backpressure_case())


async def _preview_backpressure_case():
    client = LinkClient("ws://control-room", "moto-1")
    sent = []
    release = asyncio.Event()

    async def fake_send(envelope):
        sent.append(envelope)
        await release.wait()

    client._send = fake_send  # type: ignore[method-assign]
    client.send_preview_frame(b"old")
    await asyncio.sleep(0)
    client.send_preview_frame(b"newer")
    client.send_preview_frame(b"latest")

    assert len(sent) == 1
    release.set()
    for _ in range(4):
        await asyncio.sleep(0)

    assert len(sent) == 2
    assert base64.b64decode(sent[1].payload["jpeg_b64"]) == b"latest"


def test_telemetry_send_keeps_only_latest_payload_under_backpressure():
    asyncio.run(_telemetry_backpressure_case())


async def _telemetry_backpressure_case():
    client = LinkClient("ws://control-room", "moto-1")
    sent = []
    release = asyncio.Event()

    async def fake_send(envelope):
        sent.append(envelope)
        await release.wait()

    client._send = fake_send  # type: ignore[method-assign]
    client.send_telemetry(Telemetry(gps=GpsTelemetry(speed_kmh=10.0)))
    await asyncio.sleep(0)
    client.send_telemetry(Telemetry(gps=GpsTelemetry(speed_kmh=20.0)))
    client.send_telemetry(Telemetry(gps=GpsTelemetry(speed_kmh=30.0)))

    assert len(sent) == 1
    release.set()
    for _ in range(4):
        await asyncio.sleep(0)

    assert len(sent) == 2
    assert sent[1].payload["gps"]["speed_kmh"] == 30.0


def test_log_event_send_includes_timestamp():
    asyncio.run(_log_event_case())


async def _log_event_case():
    client = LinkClient("ws://control-room", "moto-1")
    ws = FakeWs()
    client._ws = ws
    client._set_connected(True)
    client.send_log_event("WARNING", "BLE disconnected", "motocam.gimbal")
    await asyncio.sleep(0)

    assert not client._log_queue
    sent = Envelope.from_json(ws.sent[0])
    assert sent.payload["level"] == "WARNING"
    assert sent.payload["message"] == "BLE disconnected"
    assert sent.payload["module"] == "motocam.gimbal"
    assert isinstance(sent.payload["ts"], float)


def test_log_event_send_uses_single_drain_worker_for_burst():
    asyncio.run(_log_event_burst_case())


async def _log_event_burst_case():
    client = LinkClient("ws://control-room", "moto-1")
    ws = FakeWs()
    release = asyncio.Event()

    async def fake_send(payload):
        ws.sent.append(payload)
        await release.wait()

    ws.send = fake_send  # type: ignore[method-assign]
    client._ws = ws
    client._set_connected(True)
    client.send_log_event("INFO", "one", "motocam.test")
    await asyncio.sleep(0)
    first_task = client._log_send_task
    client.send_log_event("INFO", "two", "motocam.test")
    client.send_log_event("INFO", "three", "motocam.test")

    assert client._log_send_task is first_task
    assert len(ws.sent) == 1
    release.set()
    await first_task

    assert [Envelope.from_json(item).payload["message"] for item in ws.sent] == ["one", "two", "three"]


def test_log_event_queue_is_bounded_under_backpressure():
    asyncio.run(_log_event_queue_bound_case())


async def _log_event_queue_bound_case():
    client = LinkClient("ws://control-room", "moto-1")
    client._ws = FakeWs()
    client._set_connected(True)
    release = asyncio.Event()

    async def fake_send(_payload):
        await release.wait()

    client._ws.send = fake_send  # type: ignore[method-assign]
    client.send_log_event("INFO", "first", "motocam.test")
    await asyncio.sleep(0)
    for index in range(5050):
        client.send_log_event("INFO", f"msg-{index}", "motocam.test")

    assert len(client._log_queue) == 5000
    assert client._log_queue[0].payload["message"] == "msg-50"
    release.set()
    await client._log_send_task


def test_log_events_buffer_until_control_room_connects():
    asyncio.run(_log_event_offline_buffer_case())


async def _log_event_offline_buffer_case():
    client = LinkClient("ws://control-room", "moto-1")
    client.send_log_event("INFO", "boot one", "motocam.main")
    client.send_log_event("WARNING", "boot two", "motocam.main")
    await asyncio.sleep(0)

    assert [item.payload["message"] for item in client._log_queue] == ["boot one", "boot two"]
    assert client._log_send_task is None

    ws = FakeWs()
    client._ws = ws
    client._set_connected(True)
    await client._log_send_task

    assert not client._log_queue
    assert [Envelope.from_json(item).payload["message"] for item in ws.sent] == ["boot one", "boot two"]
