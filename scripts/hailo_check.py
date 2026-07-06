#!/usr/bin/env python3
"""Hailo readiness check -- tells you exactly why the AI chip shows NULL.

"AI NULL" means build_detector() fell back to NullDetector. That needs
THREE things, and this reports which are missing:

  1. the hailo_platform Python binding importable in THIS interpreter
     (installing the runtime system-wide is not enough if the app runs in
     a venv without --system-site-packages);
  2. a Hailo device the driver can see (hailortcli);
  3. a compiled .hef model at the path in config/config.yaml (ai.model).

Run it with the SAME python as the app (its venv):

    python scripts/hailo_check.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config" / "config.yaml"
DEV_HEF_MAGIC = b"MOTOCAM_DEV_HEF\n"


def _load_ai_config() -> dict:
    try:
        import yaml

        with open(CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("ai", {})
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read {CONFIG_PATH}: {exc}")
        return {}


def check_binding() -> bool:
    import importlib

    try:
        importlib.import_module("hailo_platform")
        print("  [ok]   hailo_platform importable in this interpreter")
        return True
    except ImportError as exc:
        print(f"  [MISS] hailo_platform NOT importable here ({exc})")
        print("         -> sudo apt install hailo-all, and make the venv see it:")
        print("            recreate with `python -m venv --system-site-packages .venv`")
        print("            then `source .venv/bin/activate && pip install -r requirements.txt`")
        print("            (or pip install the hailort wheel into the venv)")
        return False


def check_device() -> bool:
    cli = shutil.which("hailortcli")
    if cli is None:
        print("  [MISS] hailortcli not found -> runtime not installed (apt install hailo-all)")
        return False
    try:
        out = subprocess.run([cli, "scan"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  [MISS] hailortcli scan failed to run ({exc})")
        return False
    text = (out.stdout + out.stderr).strip()
    if "Hailo" in out.stdout and "not found" not in out.stdout.lower():
        print("  [ok]   Hailo device visible to the driver:")
        for line in out.stdout.splitlines():
            if line.strip():
                print(f"           {line.strip()}")
        return True
    print("  [MISS] no Hailo device detected by hailortcli scan")
    if text:
        print(f"           {text.splitlines()[0]}")
    print("         -> check the HAT is seated, reboot after installing hailo-all")
    return False


def check_model(ai_cfg: dict) -> str | None:
    if str(ai_cfg.get("type", "")).lower() != "hailo":
        print(f"  [MISS] config ai.type is '{ai_cfg.get('type')}', not 'hailo' -> detector disabled by config")
        return None
    model = ai_cfg.get("model")
    if not model:
        print("  [MISS] config ai.model is empty -> nothing to load")
        return None
    path = Path(model)
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    if path.is_file():
        size_mb = path.stat().st_size / 1e6
        if _is_dev_hef(path):
            print(f"  [dev]  MotoCam dev HEF marker present: {path} ({size_mb:.1f} MB)")
            print("         -> synthetic DEV HEF detector will run; replace with a real Hailo-8 HEF for race use")
            return "dev"
        print(f"  [ok]   real model candidate present: {path} ({size_mb:.1f} MB)")
        return "real"
    print(f"  [MISS] model file not found: {path}")
    print("         -> run scripts/setup_hailo.sh (downloads a stock YOLOv8n HEF)")
    return None


def _is_dev_hef(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(len(DEV_HEF_MAGIC)) == DEV_HEF_MAGIC
    except OSError:
        return False


def main() -> int:
    print(f"== Hailo readiness check ==\n(interpreter: {sys.executable})\n")
    ai_cfg = _load_ai_config()

    print("1) Python binding")
    ok_binding = check_binding()
    print("\n2) Device / driver")
    ok_device = check_device()
    print("\n3) Model file")
    model_kind = check_model(ai_cfg)

    print("\n== verdict ==")
    if model_kind == "dev":
        print("  DEV HEF active -> AI tracking pipeline can be tested without HailoRT.")
        print("  This is not real accelerator inference; install HailoRT + real HEF before deployment.")
        return 0
    if ok_binding and ok_device and model_kind == "real":
        print("  All three present -> the AI chip should leave NULL and run real inference.")
        return 0
    print("  AI stays NULL until every [MISS] above is resolved.")
    print("  Details: docs/HAILO_SETUP.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
