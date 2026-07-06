# Hailo setup pro Raspberry Pi

Tento dokument shrnuje nejrychlejší postup, jak aktivovat Hailo pro MotoCam na Raspberry Pi s AI HAT+.

## 1. Na Raspberry Pi

```bash
sudo apt update
sudo apt install -y hailo-all
sudo reboot
```

Po restartu ověř:

```bash
hailortcli scan
hailortcli fw-control identify
```

## 2. V env aplikace

```bash
cd /path/to/motocam
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import hailo_platform; print('binding OK')"
```

## 3. Model HEF

V repo je očekáván model v:

```yaml
ai:
  type: hailo_canary
  model: models/yolov8n.hef
```

Pokud model chybí:

```bash
./scripts/setup_hailo.sh
```

Nebo zadat vlastní cestu:

```bash
./scripts/setup_hailo.sh /path/to/your_model.hef
```

## 4. Ověření

```bash
python scripts/hailo_check.py
```

Při úspěchu by měl skript ukázat všechny tři položky jako `[ok]`.
