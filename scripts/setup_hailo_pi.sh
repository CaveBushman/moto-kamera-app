#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended for Raspberry Pi OS on a Raspberry Pi."
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Cannot detect OS release. Run on Raspberry Pi OS Bookworm or similar."
  exit 1
fi

if ! grep -qiE 'raspbian|debian' /etc/os-release; then
  echo "This script targets Raspberry Pi OS / Debian-based Linux."
  exit 1
fi

echo "== Hailo Pi setup =="
echo "1/6 Installing Hailo runtime"
sudo apt update
sudo apt install -y hailo-all

echo "2/6 Reboot reminder"
echo "A reboot is required before the Hailo device becomes usable."
read -r -p "Reboot now? [y/N] " answer
case "$answer" in
  [Yy]|[Yy][Ee][Ss])
    sudo reboot
    exit 0
    ;;
  *)
    echo "Reboot later with: sudo reboot"
    ;;
esac

echo "3/6 Verifying driver"
hailortcli scan || true
hailortcli fw-control identify || true

echo "4/6 Recreating venv with system packages"
if [[ -d .venv ]]; then
  echo ".venv exists; recreating it so system packages are visible to the app"
  rm -rf .venv
fi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import hailo_platform; print('binding OK')"

echo "5/6 Ensuring HEF model exists"
if [[ -f /usr/share/hailo-models/yolov8s_h8.hef ]]; then
  echo "Using HEF: /usr/share/hailo-models/yolov8s_h8.hef"
elif [[ -f models/yolov8n.hef ]]; then
  echo "Model already present: models/yolov8n.hef"
else
  bash scripts/setup_hailo.sh
fi

echo "6/6 Activating real Hailo mode"
python3 - <<'PY'
from pathlib import Path
import re

path = Path('config/config.yaml')
text = path.read_text()
text = re.sub(r'(\s*type:\s*)hailo_canary', r'\1hailo', text, count=1)
if 'type: hailo' not in text:
    text = text.replace('ai:\n', 'ai:\n  type: hailo\n', 1)
path.write_text(text)
PY

echo "Running readiness check"
python scripts/hailo_check.py
