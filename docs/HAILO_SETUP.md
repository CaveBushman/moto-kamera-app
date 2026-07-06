# Getting Hailo running (Raspberry Pi AI HAT+)

The status bar shows **AI NO HEF**, **AI NO HAILO**, **AI AI ERR** or
**AI NULL** whenever the app is running the `NullDetector` — i.e. it
could not build a real Hailo detector, so only tap-to-select tracking
works and FULL AI can't auto-acquire. Physically plugging in the AI HAT+
is **not** enough; three things must all be true:

1. **Driver + runtime** installed (`hailortcli`, kernel driver).
2. **`hailo_platform` Python binding** importable in the *app's* venv.
3. A compiled **`.hef` model** at the path in `config/config.yaml`.

Run this any time to see which piece is missing:

```bash
python scripts/hailo_check.py     # use the app's venv python
```

And this to do the setup:

```bash
scripts/setup_hailo.sh            # apt + verify + fetch/locate a model
```

---

## 1. Driver + runtime

On Raspberry Pi OS (Bookworm) for the Pi 5 + AI HAT+:

```bash
sudo apt update
sudo apt install -y hailo-all      # driver + HailoRT + python package
sudo reboot                        # required before the device is usable
```

Verify the device is seen:

```bash
hailortcli scan
hailortcli fw-control identify      # shows the Hailo-8 firmware
```

If `scan` finds nothing: reseat the HAT+, confirm the PCIe FFC cable, and
make sure you rebooted after installing.

## 2. The venv must see `hailo_platform`

`hailo-all` installs the Python package into the **system** interpreter.
A normal virtualenv is isolated and won't see it — that's the most common
cause of "device works in `hailortcli` but the app still says AI NULL".

Recreate the venv so it can:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import hailo_platform; print('binding OK')"
```

## 3. A compiled model (`.hef`)

The HAT+ contains no model — you supply a YOLO model compiled to a Hailo
Executable Format (`.hef`) for the **Hailo-8**. Models are large and are
git-ignored (see `models/README.md`); each machine fetches its own.

**Fastest start — a stock YOLOv8n (COCO):**

```bash
# if you already have a HEF:
scripts/setup_hailo.sh /path/to/yolov8n.hef

# or fetch via the Raspberry Pi Hailo examples:
git clone https://github.com/hailo-ai/hailo-rpi5-examples
cd hailo-rpi5-examples && ./download_resources.sh
scripts/setup_hailo.sh ./resources/yolov8n.hef    # path may vary
```

Then set in `config/config.yaml`:

```yaml
ai:
  type: hailo
  model: models/yolov8n.hef     # or an absolute path
  target_class: bicycle         # COCO has no "cyclist" -- use bicycle/person
  labels: null                  # null = default COCO order
```

> A stock COCO model detects `person`/`bicycle`/`motorcycle`, **not**
> `cyclist`. FULL AI auto-acquire only fires when `target_class` matches a
> class the model actually outputs, so set it to `bicycle` (or `person`).

**Best for cycling — a custom `cyclist` model:** train/convert to ONNX,
compile to HEF for `hailo8` with the Hailo Model Zoo / Dataflow Compiler,
drop it in `models/`, then `target_class: cyclist` and list its classes
under `ai.labels`.

---

## Confirming it works

```bash
python scripts/hailo_check.py     # all three -> [ok]
```

Start the app: the **AI** chip should leave `NULL` and show the tracking
state. With detections flowing, FULL AI auto-acquires the configured
class and the peloton tracker (ByteTrack) maintains stable rider IDs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hailortcli` not found | runtime not installed | step 1 |
| `scan` finds no device | not rebooted / HAT not seated | reboot, reseat |
| `import hailo_platform` fails in venv | venv isolated from system pkg | step 2 |
| `AI NO HEF` | no `.hef` at `ai.model` | step 3 |
| `AI NO HAILO` | `hailo_platform` unavailable in the app venv | step 2 |
| `AI AI ERR` | Hailo initialized but model/runtime failed | check `logs/motocam.jsonl` |
| AI leaves NULL but FULL AI never locks | `target_class` not in model classes | set `bicycle`/`person` |

The app logs the exact fallback reason — check it:

```bash
grep -i hailo logs/motocam.jsonl | tail
```
