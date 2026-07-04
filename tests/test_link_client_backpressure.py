import asyncio
import base64

from motocam.core.protocol import GpsTelemetry, Telemetry
from motocam.network.link_client import LinkClient


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
