#!/usr/bin/env bash
# Set up Hailo (Raspberry Pi AI HAT+, Hailo-8) for MotoCam.
#
# Deliberately NOT `set -e`: it walks all four steps and reports what is
# missing rather than aborting on the first gap. Re-run it after fixing a
# step -- it is idempotent.
#
# Usage:
#   scripts/setup_hailo.sh                 # apt + verify + auto-find a model
#   scripts/setup_hailo.sh /path/to.hef    # use a local HEF you already have
#   scripts/setup_hailo.sh https://.../x.hef   # download a HEF
set -uo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script is intended for Raspberry Pi OS (Linux) with a Hailo AI HAT+."
  echo "Current OS: $(uname -s)"
  exit 1
fi

if ! command -v apt >/dev/null 2>&1; then
  echo "apt not found; run this script on Raspberry Pi OS / Debian-based Linux."
  exit 1
fi

if [[ -f /proc/device-tree/model ]] && ! grep -qi "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
  echo "This does not look like a Raspberry Pi. Hailo AI HAT+ setup needs a Raspberry Pi."
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$REPO/models"
DEST="$MODELS/yolov8n.hef"
PY="${PYTHON:-python3}"
SRC="${1:-}"
mkdir -p "$MODELS"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

say "1/4  Runtime + driver (hailo-all)"
if command -v hailortcli >/dev/null 2>&1; then
  echo "hailortcli already installed -- skipping apt."
else
  echo "Installing hailo-all (needs sudo; a reboot is required afterwards)..."
  if sudo apt update && sudo apt install -y hailo-all; then
    echo "Installed. REBOOT before the device is usable:  sudo reboot"
  else
    echo "apt install failed -- install 'hailo-all' manually, then re-run this script."
  fi
fi

say "2/4  Is the device visible?"
if command -v hailortcli >/dev/null 2>&1; then
  hailortcli scan || echo "scan failed -- reboot after install and check the HAT is seated."
else
  echo "hailortcli missing -- runtime not installed yet (see step 1)."
fi

say "3/4  Python binding in the app venv"
if "$PY" -c 'import hailo_platform' 2>/dev/null; then
  echo "hailo_platform importable with $PY -- good."
else
  echo "hailo_platform NOT importable with $PY."
  echo "  The runtime installs into the system Python; a plain venv can't see it."
  echo "  Recreate the venv with system packages:"
  echo "     python -m venv --system-site-packages .venv"
  echo "     source .venv/bin/activate && pip install -r requirements.txt"
fi

say "4/4  Model (.hef)"
if [ -f "$DEST" ]; then
  echo "Model already present: $DEST"
elif [ -n "$SRC" ] && [ -f "$SRC" ]; then
  cp "$SRC" "$DEST" && echo "Copied $SRC -> $DEST"
elif [ -n "$SRC" ]; then
  echo "Downloading $SRC ..."
  curl -fL "$SRC" -o "$DEST" && echo "Saved $DEST" || echo "download failed -- pass a local .hef path instead."
else
  FOUND="$(find /usr/share /opt -iname 'yolov8*.hef' 2>/dev/null | head -1)"
  if [ -n "$FOUND" ]; then
    cp "$FOUND" "$DEST" && echo "Copied already-installed model $FOUND -> $DEST"
  else
    echo "No .hef supplied and none found under /usr/share or /opt."
    echo "Obtain a Hailo-8 YOLOv8n HEF, e.g. via the Raspberry Pi Hailo examples:"
    echo "    git clone https://github.com/hailo-ai/hailo-rpi5-examples"
    echo "    cd hailo-rpi5-examples && ./download_resources.sh"
    echo "  then re-run:  scripts/setup_hailo.sh /path/to/yolov8n.hef"
  fi
fi

say "Config to set (config/config.yaml)"
cat <<'CFG'
    ai:
      type: hailo
      model: models/yolov8n.hef      # or an absolute path
      target_class: bicycle          # a stock COCO model has no 'cyclist'
CFG

echo
echo "Verify everything:  $PY scripts/hailo_check.py"
