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

DEST="$REPO/models/cyclist.hef"
mkdir -p "$REPO/models"
cp "$SRC" "$DEST"

echo "Installed HEF: $DEST"

python3 - "$DEST" <<'PY'
from pathlib import Path
import re
import sys

dest = sys.argv[1]
path = Path('config/config.yaml')
text = path.read_text()
# Absolute path, not a repo-relative one: resolve_hef_path() only resolves
# a relative model path against config.yaml's own directory, and this app
# is normally launched with cwd=motocam/ (one level below the repo root
# models/ actually lives in) -- a relative path here would silently
# resolve to the wrong, nonexistent location. Scoped to the ai: block's
# own model:/target_class: lines (indented under it), not the first match
# anywhere in the file, so an unrelated top-level key can never be hit.
text = re.sub(r'(?m)^(ai:\n(?:  .*\n)*?  model:\s*).+$', rf'\1{dest}', text, count=1)
text = re.sub(r'(?m)^(ai:\n(?:  .*\n)*?  target_class:\s*).+$', r'\1cyclist', text, count=1)
path.write_text(text)
PY

echo "Updated config/config.yaml to use the cyclist HEF."
echo "Next run: python scripts/hailo_check.py"
