# RS 4 Pro BLE — reverse-engineering findings

Evidence from a real capture (`scripts/rs4_ble_capture.py` →
`scripts/rs4_ble_analyze.py`), 615 vendor frames over BLE characteristic
`fff4` (notify). Implemented in `motocam/gimbal/dji_duml.py`.

## Transport: DJI DUML (confirmed)

The RS 4 Pro's BLE stream is **not** the public R SDK 0xAA framing used on
the RSA UART/CAN port (`rsdk_protocol.py`). It is DJI **DUML**:

| off | field |
|----:|-------|
| 0 | SOF `0x55` |
| 1 | length low 8 bits (total frame length) |
| 2 | length high 2 bits \| version<<2 (version = 1) |
| 3 | CRC-8 of bytes[0:3], init `0x77` |
| 4 | sender id |
| 5 | receiver id |
| 6–7 | sequence number (LE) |
| 8 | cmd type (`0x00` request, `0x20` ack) |
| 9 | cmd set (`0x04` = gimbal) |
| 10 | cmd id |
| 11… | payload |
| −2 | CRC-16 of bytes[0:−2], LE, init `0x3692`, reflected poly `0x8408` |

**Validation:** across all 615 frames — SOF `0x55` 615/615, `byte[1] ==
len` 615/615, header **CRC-8 615/615**, frame **CRC-16 615/615**. Our
`build_duml_frame()` reproduces captured frames byte-for-byte, so frames
we build carry checksums the gimbal accepts.

## Command sets / ids seen (all pushed *from* the gimbal)

| cmd_set | cmd_id | n | note |
|--------:|-------:|--:|------|
| 0x04 (gimbal) | 0x27 | 231 | dominant gimbal push |
| 0x1c | 0x01 | 115 | |
| 0x02 | 0x80 | 115 | |
| 0x04 | 0x05 / 0x38 / 0x1c | 23 each | gimbal |
| 0x04 | 0x64 | 23 | ack |
| 0x00 | 0xf1 | 23 | (general — version?) |
| 0x0d | 0x02 | 23 | |

sender→receiver pairs: `04→02`, `e5→02`, `e4→02`, `27→02`.

## What we can and cannot do yet

**Solved — the transport.** We can frame, checksum, parse and build DUML.
This replaces the (incorrect for BLE) 0xAA path in the BLE backend.

**Not solved — telemetry mapping.** In this capture the gimbal was
essentially stationary: within each `(cmd_set, cmd_id)` the payload bytes
are constant; only the sequence number and CRC change. So we cannot yet
map payload offsets to yaw / roll / pitch.
→ **Need a capture while deliberately moving the gimbal** (pan full
left→right, tilt up→down, roll) so the angle bytes sweep and can be
correlated. Then `get_orientation()` for the BLE backend can be written.

**Not solved — control commands.** The capture is only `fff4`
*notifications* (gimbal → app). It contains no app→gimbal writes, so the
set-speed / set-position command ids and payloads are unknown.
→ **Need a write-side capture** (the DJI Ronin app's BLE writes — e.g. an
Android HCI snoop log, or a BLE sniffer) to learn the control cmd_set/id
and payload layout. The write characteristic is most likely `fff5`
(paired with the `fff4` notify), to be confirmed.

## Next steps

1. Capture with the gimbal moving through known extents → decode attitude
   → implement DUML `get_orientation()`.
2. Capture the Ronin app's writes → decode the speed/position command →
   implement DUML control, replacing the 0xAA R SDK frames on the BLE
   transport in `dji_rs4pro.py`.
3. Keep CAN/UART on the existing R SDK (0xAA) path — that transport is
   unchanged; only BLE is DUML.
