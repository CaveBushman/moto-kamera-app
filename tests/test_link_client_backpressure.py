import asyncio
import base64

from motocam.core.protocol import GpsTelemetry, Telemetry
from motocam.network.link_client import LinkClient


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
    sent = []

    async def fake_send(envelope):
        sent.append(envelope)

    client._send = fake_send  # type: ignore[method-assign]
    client.send_log_event("WARNING", "BLE disconnected", "motocam.gimbal")
    await asyncio.sleep(0)

    assert sent[0].payload["level"] == "WARNING"
    assert sent[0].payload["message"] == "BLE disconnected"
    assert sent[0].payload["module"] == "motocam.gimbal"
    assert isinstance(sent[0].payload["ts"], float)


def test_log_event_send_uses_single_drain_worker_for_burst():
    asyncio.run(_log_event_burst_case())


async def _log_event_burst_case():
    client = LinkClient("ws://control-room", "moto-1")
    sent = []
    release = asyncio.Event()

    async def fake_send(envelope):
        sent.append(envelope)
        await release.wait()

    client._send = fake_send  # type: ignore[method-assign]
    client.send_log_event("INFO", "one", "motocam.test")
    await asyncio.sleep(0)
    first_task = client._log_send_task
    client.send_log_event("INFO", "two", "motocam.test")
    client.send_log_event("INFO", "three", "motocam.test")

    assert client._log_send_task is first_task
    assert len(sent) == 1
    release.set()
    await first_task

    assert [item.payload["message"] for item in sent] == ["one", "two", "three"]


def test_log_event_queue_is_bounded_under_backpressure():
    asyncio.run(_log_event_queue_bound_case())


async def _log_event_queue_bound_case():
    client = LinkClient("ws://control-room", "moto-1")
    release = asyncio.Event()

    async def fake_send(_envelope):
        await release.wait()

    client._send = fake_send  # type: ignore[method-assign]
    client.send_log_event("INFO", "first", "motocam.test")
    await asyncio.sleep(0)
    for index in range(250):
        client.send_log_event("INFO", f"msg-{index}", "motocam.test")

    assert len(client._log_queue) == 200
    assert client._log_queue[0].payload["message"] == "msg-50"
    release.set()
    await client._log_send_task
