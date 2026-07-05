"""BLE transport logic for the RS 4 Pro: GATT characteristic selection
(the write/notify pairing that must land on fff5/fff4 on real hardware),
configured-UUID handling, the pairing heuristics, MTU chunking on send,
and the notify->queue path. All exercised with fake GATT objects -- no
bleak, no adapter."""
from __future__ import annotations

import asyncio

import pytest

from motocam.gimbal.dji_rs4pro import BleEndpoint, BleTransport

# Real RS 4 Pro GATT profile (from the app log: write=fff5 notify=fff4 on
# the fff0 service, write-without-response).
FFF0 = "0000fff0-0000-1000-8000-00805f9b34fb"
FFF4 = "0000fff4-0000-1000-8000-00805f9b34fb"  # notify
FFF5 = "0000fff5-0000-1000-8000-00805f9b34fb"  # write-without-response


class FakeChar:
    def __init__(self, uuid: str, properties: list[str]):
        self.uuid = uuid
        self.properties = properties


class FakeService:
    def __init__(self, uuid: str, characteristics: list[FakeChar]):
        self.uuid = uuid
        self.characteristics = characteristics


class FakeServiceCollection:
    """Mimics bleak's BleakGATTServiceCollection, whose `.services` is a
    dict keyed by handle -- exercises the _iter_services dict branch."""

    def __init__(self, services: list[FakeService]):
        self.services = {i: svc for i, svc in enumerate(services)}


def _rs4_profile() -> list[FakeService]:
    return [
        FakeService(FFF0, [
            FakeChar(FFF4, ["notify"]),
            FakeChar(FFF5, ["write-without-response", "write"]),
        ]),
    ]


def _select(services, service_uuid=None, tx=None, rx=None) -> BleEndpoint:
    return BleTransport._select_characteristics(
        services, service_uuid=service_uuid, tx_char_uuid=tx, rx_char_uuid=rx
    )


# -- real-hardware profile ------------------------------------------------
def test_selects_fff5_write_and_fff4_notify_on_real_profile():
    endpoint = _select(_rs4_profile())
    assert endpoint.tx_uuid == FFF5
    assert endpoint.rx_uuid == FFF4
    # fff5 advertises write-without-response -> response=False (matches log)
    assert endpoint.write_with_response is False


def test_works_when_services_is_a_dict_collection():
    endpoint = _select(FakeServiceCollection(_rs4_profile()))
    assert endpoint.tx_uuid == FFF5 and endpoint.rx_uuid == FFF4


# -- configured UUIDs -----------------------------------------------------
def test_configured_uuids_are_honored_over_discovery():
    # add a decoy writable char that discovery might otherwise pick
    services = [FakeService(FFF0, [
        FakeChar(FFF4, ["notify"]),
        FakeChar(FFF5, ["write-without-response"]),
        FakeChar("0000aaaa-0000-1000-8000-00805f9b34fb", ["write"]),
    ])]
    endpoint = _select(services, tx=FFF5, rx=FFF4)
    assert endpoint.tx_uuid == FFF5 and endpoint.rx_uuid == FFF4


def test_missing_configured_write_uuid_raises():
    with pytest.raises(RuntimeError, match="write characteristic not found"):
        _select(_rs4_profile(), tx="0000dead-0000-1000-8000-00805f9b34fb")


def test_missing_configured_notify_uuid_raises():
    with pytest.raises(RuntimeError, match="notify characteristic not found"):
        _select(_rs4_profile(), rx="0000dead-0000-1000-8000-00805f9b34fb")


# -- pairing heuristics ---------------------------------------------------
def test_best_pair_prefers_write_and_notify_on_the_same_service():
    services = [
        FakeService(FFF0, [FakeChar(FFF5, ["write"]), FakeChar(FFF4, ["notify"])]),
        FakeService("0000ee00-0000-1000-8000-00805f9b34fb",
                    [FakeChar("0000ee04-0000-1000-8000-00805f9b34fb", ["notify"])]),
    ]
    endpoint = _select(services)
    assert endpoint.tx_uuid == FFF5 and endpoint.rx_uuid == FFF4  # same-service pair wins


def test_service_uuid_filter_ignores_other_services():
    services = [
        FakeService(FFF0, [FakeChar(FFF4, ["notify"]), FakeChar(FFF5, ["write"])]),
        FakeService("0000ee00-0000-1000-8000-00805f9b34fb",
                    [FakeChar("0000ee05-0000-1000-8000-00805f9b34fb", ["write"])]),
    ]
    endpoint = _select(services, service_uuid=FFF0)
    assert endpoint.tx_uuid == FFF5 and endpoint.rx_uuid == FFF4


def test_plain_write_char_yields_write_with_response_true():
    services = [FakeService(FFF0, [FakeChar(FFF4, ["notify"]), FakeChar(FFF5, ["write"])])]
    endpoint = _select(services)
    assert endpoint.write_with_response is True


def test_no_write_or_notify_characteristic_raises():
    services = [FakeService(FFF0, [FakeChar(FFF4, ["read"])])]
    with pytest.raises(RuntimeError, match="Could not find BLE write/notify"):
        _select(services)


# -- send() MTU chunking + notify queue -----------------------------------
class FakeClient:
    def __init__(self):
        self.writes: list[tuple[str, bytes, bool]] = []

    async def write_gatt_char(self, uuid, data, response):
        self.writes.append((uuid, bytes(data), response))


def test_send_splits_frame_into_mtu_sized_writes():
    transport = BleTransport(address="x", mtu_payload_bytes=8)
    client = FakeClient()
    transport._client = client
    transport._endpoint = BleEndpoint(tx_uuid=FFF5, rx_uuid=FFF4, write_with_response=False)

    frame = bytes(range(20))  # 20 bytes -> 8 + 8 + 4
    asyncio.run(transport.send(frame))

    sizes = [len(data) for _uuid, data, _resp in client.writes]
    assert sizes == [8, 8, 4]
    assert b"".join(data for _u, data, _r in client.writes) == frame
    assert all(uuid == FFF5 and resp is False for uuid, _d, resp in client.writes)


def test_send_without_open_raises():
    transport = BleTransport(address="x")
    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(transport.send(b"\x01\x02"))


def test_on_notify_enqueues_bytes():
    transport = BleTransport(address="x")
    transport._queue = asyncio.Queue()
    transport._on_notify(None, bytearray(b"\x55\x12\x04"))
    assert transport._queue.get_nowait() == b"\x55\x12\x04"


def test_drain_returns_all_queued_and_empties():
    transport = BleTransport(address="x")
    transport._queue = asyncio.Queue()
    transport._on_notify(None, b"\xaa")
    transport._on_notify(None, b"\xbb\xcc")
    assert asyncio.run(transport.drain()) == b"\xaa\xbb\xcc"
    assert asyncio.run(transport.drain()) == b""  # emptied
