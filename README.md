# MotoCam

Motorcycle AI camera control unit for Raspberry Pi 5 + AI HAT+ + Blackmagic
PYXIS + DJI RS 4 Pro. See `Technicky_navrh_motocyklova_AI_kamera_RPi5_PYXIS_RS4Pro.md`
for the full technical design this implements.

Talks to the director's dashboard in the sibling
`../08 livestream road cycling control room` project over a WebSocket link
(design doc section 6.4) -- telemetry/preview/rider PTT out, stage info
and director talkback PTT in.

## Status

This is a working MVP1-level app: live preview (UVC or synthetic test
pattern), tap-to-select + OpenCV CSRT tracking, **real Hailo AI HAT+
detection with FULL-AI auto-acquire**, DJI R SDK gimbal control,
**real PYXIS camera control**, simulated/real GPS, and a real
bidirectional link to the control room.

- `motocam/ai/hailo_detector.py` -- REAL: YOLO object detection on the
  Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS) via the HailoRT Python API
  (`hailo_platform`). Loads a compiled `.hef`, runs per-frame inference,
  parses the HAILO_NMS output and maps boxes back to source pixels
  (letterbox-aware). In **FULL AI** mode the highest-confidence
  detection of the configured `ai.target_class` auto-acquires the CSRT
  tracker from the detector's own box (design doc 8.2/8.3); **AI ASSIST**
  keeps the rider tapping, **MANUAL** never auto-acquires. Enabled with
  `ai.type: hailo` + an `ai.model` HEF path; the coordinate maths is
  pure and unit-tested, the runtime is import-guarded, and a missing
  runtime/HEF degrades to `NullDetector` (tap-to-select still works) --
  never fake detections. Off the Pi, leave `ai.type` unset for
  NullDetector. Model compilation (ONNX -> HEF via the Hailo Model Zoo)
  is a one-time offline step; `ai.labels` in config gives the compiled
  model's class order (omit for a stock COCO YOLOv8/v11 HEF).

- `motocam/camera/pyxis_camera.py` -- REAL: Blackmagic Camera Control
  REST API over Ethernet (`http://<ip>/control/api/v1`, the officially
  documented control surface for PYXIS). Record start/stop, ISO, white
  balance, shutter (speed or angle), iris, autofocus, servo zoom
  (velocity synthesized from position nudges -- REST has no velocity
  endpoint), media-remaining and frame-rate readouts. Auto-reconnects
  with a throttled probe (one `/system` GET per ~5 s while unreachable),
  degrades per-field on endpoints a given firmware/lens doesn't serve,
  and never blocks the UI (stdlib urllib in an executor). Selected via
  `camera.type: blackmagic_pyxis` in config; `camera.type: mock` for
  development without the camera. Field names should be spot-checked
  against the firmware's own OpenAPI files on first contact with real
  hardware; the IP is retargetable live from Settings -> Camera & Lens.
- `motocam/gimbal/dji_rs4pro.py` -- REAL (pending one hardware smoke
  test): DJI R SDK control over BLE (`gimbal.connection: ble`) or via
  the gimbal's RSA port over CAN (`connection: can`, e.g. an MCP2515 HAT
  / USB-CAN adapter on the Pi at 1 Mbps, IDs 0x223/0x222) or UART
  (`connection: uart`).
  Frame codec + commands (velocity control for the PID loop, absolute
  position for HOME, position readout) live in `gimbal/rsdk_protocol.py`;
  all protocol constants are verified against the open-source
  ConstantRobotics/DJIR_SDK implementation of DJI's official R SDK
  protocol v2.2 (run by its authors against real RS-series hardware).
  The backend only reports connected after the gimbal answers a
  GET_POSITION with a CRC-valid frame, so any RS 4 Pro difference shows
  as DISCONNECTED, never as fake control. DJI's R SDK PDF v2.5 does not
  publish BLE GATT UUIDs, so BLE supports Settings -> GIMBAL CONTROL ->
  SCAN BLE DEVICES, best-effort write/notify auto-discovery, plus
  explicit service/write/notify UUID fields in Settings/config for the
  first real-hardware pairing. The shipped config defaults to BLE for
  Raspberry Pi field use; `connection: mock` keeps the simulated gimbal
  for development. `python-can` is required only on the
  Pi (`pip3 install python-can`); `bleak` is used for BLE and is already
  in `requirements.txt`.

Beyond MVP1, also implemented:

- **Race cockpit UI pass**: rebuilt top status bar, stage banner, camera/
  gimbal metric panels, bottom mode bar, and preview overlays into a more
  polished broadcast/touch interface with clearer LIVE/PREVIEW/REC states.
- **Manual gimbal control**: a big semi-transparent on-screen joystick
  (`ui/widgets/virtual_joystick.py`), bottom-right of the preview, glove-sized,
  only live in MANUAL mode. Holding it at an offset keeps panning/tilting,
  not just discrete drag events.
- **Zoom control**: a horizontal spring-back rocker above the joystick
  (`ui/widgets/zoom_rocker.py`); left/right sends zoom-out/zoom-in velocity
  and release sends zero.
- **AF control**: a compact autofocus button beside the smaller REC button
  in the camera panel; wired through the camera controller/mock backend.
- **Settings dialog** (`ui/widgets/settings_dialog.py`): unit ID (MOTO 1-4)
  and control-room IP/port with live reconnect, PYXIS IP/port, gimbal
  transport (mock/BLE/CAN/UART) with BLE UUID overrides, camera ISO/WB/
  shutter/iris, PTT input/output audio devices, and AI tracking params
  (target class, confidence, dead zone, max pan/tilt speed) -- all editable
  at runtime, no restart needed.
  `LinkClient.reconfigure()` handles the live control-room reconnect.
- **Unit identity on screen**: a large "MOTO N" badge in the top bar (not
  just the window title, which isn't visible on a kiosk touchscreen).
- **Director talkback**: plays PTT audio sent from the control room and
  shows a prominent "CONTROL ROOM TALKBACK" overlay on the live preview.
- **ATEM awareness**: shows LIVE/PREVIEW on the preview image when the
  control room reports this moto as ATEM program/preview.
- **Startup identity**: splash screen with the federation logo before the
  rider UI appears.
- Crash-hardening: `core.logging_setup.install_crash_guard()` stops a Qt
  slot exception from taking down the whole process (PyQt6's default is to
  abort) -- logs a full traceback instead.
- Dark "glassmorphism-lite" theme (`ui/theme.py` + `ui/effects.py`):
  translucent cards with real drop shadows for panels/dialogs, kept opaque
  and high-contrast on the mode bar and status chips on purpose (11.1
  glanceable-in-sunlight, glove-usable requirement -- glass panels only
  where legibility isn't safety-critical).
- Federation logo (`ui/widgets/logo.py`) in the top bar, always
  `scaledToHeight()` so the wide wordmark never gets stretched/distorted.
- **Full-bleed preview + collapsible HUD**: the live feed now fills nearly
  the whole window instead of sharing it with a fixed CAMERA/GIMBAL row;
  those readouts float as a semi-transparent glass HUD over the top of the
  feed, off by default, brought back with a "CAM / GIMBAL" pill (mirrored
  by a "SETTINGS" pill on the opposite corner) -- see
  `ui/widgets/preview_view.py::set_hud_widget`.
- **PTT / ZOOM / joystick spacing**: the three bottom thumb controls sit in
  three separate columns (far-left/center/far-right) instead of stacked
  close together, so a gloved thumb can't easily slip from one to another
  on a moving bike.
- **Video source selector** (Settings -> Video Source): lists detected
  UVC/V4L2 capture devices and hot-swaps `VideoEngine`'s device at runtime
  (`video/devices.py`, `VideoEngine.set_device`), no restart needed.
- **Preview relay bandwidth cap** (`config/config.yaml -> preview_relay`):
  the dashboard JPEG preview is downscaled and rate-limited separately
  from the PYXIS SRT/program path, so weak uplinks are not consumed by
  control-room thumbnails.
- **Telemetry source reporting**: every telemetry packet now carries
  whether GPS/video/camera/gimbal/AI is real hardware or fallback
  (`GPS REAL/SIMULATED`, `VID REAL/SYNTHETIC`, `GMB DJI-RSDK-BLE/MOCK`,
  etc.). The control room displays this directly, so field testing on
  the Pi cannot accidentally mistake a simulated subsystem for real data.
  The rider top bar also marks fallback sources as warnings (`GPS SIM`,
  `GIMBAL MOCK`, `CAM SYNTH`, `AI NULL`) instead of letting them look
  like normal ready states.
- **GPS auto-detect**: `gps.device: auto` scans common USB serial paths
  and only accepts a port after seeing an NMEA sentence; otherwise it
  falls back to simulated GPS with the source marked `SIMULATED`.
- **Ride-safe lockout** (Settings -> Safety, off by default): above a
  configurable GPS speed, SETTINGS and the CAM/GIMBAL HUD gray out so a
  gloved thumb isn't tempted into fiddly ISO/shutter/lens taps while
  riding -- PTT, the joystick and zoom always stay live.
- **Cancel tracking**: a "✕ CANCEL TRACKING" pill appears over the preview
  whenever a tapped target is locked/weak, dropping just that target
  (`TrackingEngine.clear()`) without touching gimbal mode or position --
  previously the only way to let go of a target was the big RESET button
  (which also re-homes the gimbal) or cycling modes away and back.

## Setup

```bash
cd "07 moto kamera app"
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt   # use pip3/python3, not the bare
                                    # `python`/`pip` names -- some shells
                                    # alias those to the system Python,
                                    # which silently installs outside the venv
```

## Running

Run as a module from this directory (the project root), not by `cd`-ing
into `motocam/` and running `main.py` directly -- the app uses absolute
imports like `from motocam.video...` that need the project root on
`sys.path`:

```bash
python3 -m motocam.main
# or with a specific config:
python3 -m motocam.main --config config/config.yaml
```

With no capture device attached it falls back to a synthetic test pattern
and simulated GPS, so the full UI/tracking/networking loop runs on a plain
dev laptop with no hardware.

By default it tries to reach the control room at `ws://127.0.0.1:8765`
(see `config/config.yaml` -> `telemetry.control_room_url`). Start the
control room app first (or the moto app will just keep retrying every 3s
in the background -- it degrades gracefully with NET shown as DOWN).

## Layout

```
motocam/
  core/       config loading, JSONL logging, wire protocol (protocol.py),
              crash guard (logging_setup.install_crash_guard)
  video/      UVC capture / synthetic fallback (video_engine.py)
  ai/         Hailo YOLO integration point (currently a NullDetector stub)
  tracking/   tap-to-select CSRT tracker + PID pan/tilt regulator
  gimbal/     abstract controller + mock backend + DJI RS 4 Pro backend
  camera/     abstract controller + mock backend + PYXIS REST backend
  gps/        NMEA serial reader with simulated-fix fallback
  audio/      rider microphone PTT capture + director talkback playback +
              PortAudio device listing for Settings
  network/    WebSocket client to the control room (live reconfigure)
  watchdog/   CPU/RAM/temp health sampling
  ui/         PyQt6 widgets (glassmorphism-lite dark theme, touch-friendly
              per design doc 11.1): main_window, theme, effects, and
              widgets/ (top_bar, preview_view, virtual_joystick, camera_panel,
              gimbal_panel, mode_bar, settings_dialog, service_screen, logo)
```

## Known limitations (be aware, not blockers)

- `psutil` on some macOS betas throws on `virtual_memory()`/`cpu_percent()`
  (mach struct mismatch) -- handled gracefully (`watchdog/health.py` catches
  it and reports the field as unavailable), just don't expect real numbers
  on an affected dev machine.
- `DjiRs4ProBackend` is implemented for BLE/CAN/UART, but still needs a
  real RS 4 Pro smoke test. BLE may require filling the service/write/
  notify UUIDs in Settings if auto-discovery picks the wrong GATT pair.
  (`PyxisCameraBackend` is now a real REST client; its remaining caveat
  is spot-checking field names against the firmware's OpenAPI files on
  first contact with actual hardware.)
- **macOS beta native crash**: on some macOS 26/27 betas, PyQt6/Qt6 can
  segfault inside its own Cocoa backing-store code
  (`QPaintDevice::devicePixelRatio()` null deref during a window flush --
  not a Python exception, so `install_crash_guard()` can't catch it; a
  native SIGSEGV takes the whole process down unconditionally). This is a
  Qt/Cocoa compatibility bug against the beta OS, not an app bug -- it
  won't occur on the real deployment target (Raspberry Pi OS uses Qt's
  `xcb`/`eglfs` backend, not `cocoa`). If it hits your dev Mac, try
  `QT_ENABLE_HIGHDPI_SCALING=0 python3 main.py` (known workaround for this
  crash signature on early macOS betas) and/or
  `pip3 install --upgrade PyQt6 PyQt6-Qt6`.
  After a few crashes, macOS starts showing its own "reopen windows after
  a crash?" recovery dialog at next launch -- and rendering *that* dialog
  hits the same Qt bug, crashing again before your code even runs, ad
  infinitum. Break the loop with:
  ```bash
  defaults write org.python.python NSQuitAlwaysKeepsWindows -bool false
  rm -rf ~/Library/Saved\ Application\ State/org.python.python.savedState
  ```
