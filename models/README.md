# AI models (Hailo `.hef`)

Compiled Hailo-8 detection models live here. They are **large binaries and
are git-ignored** (`*.hef`) — each machine downloads its own with
`scripts/setup_hailo.sh`, so the repo stays lean and nobody commits a
200 MB blob.

## What goes here

A YOLO model compiled to a Hailo Executable Format (`.hef`) for the
**Hailo-8** on the Raspberry Pi AI HAT+ (26 TOPS). The default the app
looks for is set by `ai.model` in `config/config.yaml`.

- **Getting started:** a stock **YOLOv8n** HEF (COCO classes — detects
  `person`, `bicycle`, `motorcycle`, …). `scripts/setup_hailo.sh` fetches
  one here as `models/yolov8n.hef`. With a stock model, set
  `ai.target_class` to `bicycle` or `person` (COCO has no `cyclist`).
- **Best for cycling:** a custom `cyclist`-trained model compiled to HEF via
  the Hailo Model Zoo / Dataflow Compiler. Then `ai.target_class: cyclist`
  and list its classes under `ai.labels`.

See `docs/HAILO_SETUP.md` for the full procedure.

## Development marker

`models/dev_cyclist.hef` is intentionally **not** a real Hailo executable.
It is a tiny MotoCam marker file used on a dev machine to enable the
synthetic `DEV HEF` detector, so AI ASSIST / FULL AI / ByteTrack can be
tested before a compiled model is available. Do not use it for a race.
