#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_cyclist_hef.sh /path/to/your_model.hef"
  echo "Copies the supplied Hailo HEF to models/cyclist.hef and updates config/config.yaml."
  exit 1
fi

SRC="$1"
if [[ ! -f "$SRC" ]]; then
  echo "HEF file not found: $SRC"
  exit 1
fi

mkdir -p /motocam/ai/models
cp "$SRC" /motocam/ai/models/cyclist.hef

echo "Installed HEF: /motocam/ai/models/cyclist.hef"

python3 - <<'PY'
from pathlib import Path
import re

path = Path('config/config.yaml')
text = path.read_text()
text = re.sub(r'(\s*model:\s*).+', r'\1/motocam/ai/models/cyclist.hef', text, count=1)
text = re.sub(r'(\s*target_class:\s*).+', r'\1cyclist', text, count=1)
path.write_text(text)
PY

echo "Updated config/config.yaml to use the cyclist HEF."
echo "Next run: python scripts/hailo_check.py"
