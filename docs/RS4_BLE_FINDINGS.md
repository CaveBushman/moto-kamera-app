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

## Control commands: SOLVED and live-confirmed

A PacketLogger capture of the DJI Ronin app's own BLE writes (iOS
Bluetooth logging profile + PacketLogger.app, parsed with
`scripts/pklg_parse.py` — see that script's docstring for the `.pklg`
record format) revealed both control commands actually used by the app.
The write characteristic is confirmed **`fff5`** (write-without-response),
paired with the `fff4` notify, matching `BleTransport`'s existing GATT
auto-discovery.

### Joystick velocity (cmd_set 0x04, cmd_id 0x01)

Sent continuously (~15–20 Hz) while any axis is off-center, sender
`0x00` → receiver `0x04`, cmd_type `0x00`. 9-byte payload: three uint16
LE channels, each centered at `1024` (rest/zero):

```
[chA uint16 LE] [chB uint16 LE] [chC uint16 LE] [00 00] [02]
```

The capture showed each moved channel sweeping smoothly across roughly
`364..1683`/`364..1680` (held at each extreme) — symmetric `1024 ± ~660`,
which is the *empirically proven-safe* deflection used by
`build_joystick_frame()` (not the theoretical 0..2047 11-bit full range,
which was never observed). The third channel stayed at `1024` (untouched)
throughout the capture.

**Live-confirmed against real hardware** (RS 4 Pro, moto-1 unit):
- `chA` = **tilt** (positive = up, negative = down)
- `chC` = **pan** (negative = left, positive = right)
- `chB` = unconfirmed (likely roll), never exercised in any capture

Implemented as `DjiRs4ProBackend.set_velocity()` for the DUML/BLE
protocol, normalizing `pan_deg_s`/`tilt_deg_s` by the backend's
`max_pan_speed`/`max_tilt_speed` into the `±660` channel range.

### Recenter / HOME (cmd_set 0x04, cmd_id 0x4c)

A one-shot command (unlike the continuous joystick stream), sender
`0x02` → receiver `0x04`, cmd_type `0x40`, 2-byte payload `fe 01`.

An earlier capture had suggested `cmd_set 0x00 / cmd_id 0x34` (payload
`01 01 00 00 00 00`) as the recenter command — it looked plausible (fired
once, right as the button was pressed) but **live-tested as a no-op**:
sent to the real gimbal twice, including once with the gimbal
deliberately misaligned first, with zero motion. It doesn't reappear in
later captures at all, so it was evidently an unrelated command that
happened to fire near that button press.

`cmd_set 0x04 / cmd_id 0x4c` is the real one: it lives under the same
command set as the joystick (0x04 = gimbal, vs. 0x00 = general/system),
fires exactly once per press, and the surrounding traffic in the capture
is just the same ambient chatter (`0x06/0x48`, `0x00/0x0e`, `0x04/0x12`,
`0x07/0x30`, `0x00/0x00`, `0x00/0x01`) seen in every capture regardless
of what button was pressed. **Live-confirmed**: sent to the real gimbal
with it deliberately off-center — it visibly recentered.

Implemented as `DjiRs4ProBackend.go_home()` for the DUML/BLE protocol via
`build_recenter_frame()`.

### Ambient/background traffic (not yet decoded, not needed so far)

Every capture — regardless of what the operator did — also shows
`cmd_set 0x06/id 0x48`, `cmd_set 0x00/id 0x0e`, `cmd_set 0x00/id 0x00`,
`cmd_set 0x00/id 0x01`, `cmd_set 0x07/id 0x30`, `cmd_set 0x0d/id 0x01`,
and `cmd_set 0x04/id 0x12` (the last with an always-constant ~27-byte
payload) at roughly 1–10 Hz. These look like periodic status/heartbeat
polling unrelated to any specific action — joystick and recenter both
worked when sent standalone (no need to replicate this ambient traffic),
so it's left alone for now.

## Still open

**Telemetry mapping (orientation).** In the read-only capture (gimbal
essentially stationary or moved without isolating one axis at a time),
payload bytes were either constant or too aliased (the attitude push,
`cmd_set 0x04/id 0x05`, is only ~1 Hz) to reliably assign offsets to
yaw/roll/pitch. `get_orientation()` over BLE remains un-implemented
(returns the last known value, never queried/faked). Would need a
capture holding each axis at known angles (0°, ±90°) for several seconds
each to get clean, unaliased samples.

## Method note: getting a write-side capture (worked)

1. Install "Additional Tools for Xcode" (has `PacketLogger.app`) and the
   iOS Bluetooth logging profile from developer.apple.com.
2. Connect the iPhone via USB, `PacketLogger.app` → File → New iOS Trace.
3. Operate the DJI Ronin app (joystick / recenter button) while it
   records.
4. Save as `.pklg`, parse with `scripts/pklg_parse.py <file.pklg>` — it
   extracts every ATT-Send DUML frame, grouped by `(cmd_set, cmd_id)`,
   with full untruncated payloads and a timeline.
5. **The phone and any of our own BLE test scripts cannot hold the BLE
   link at the same time** — only one central can connect to the gimbal.
   Disconnect one before using the other, or captures come back empty or
   drop after ~1s.
