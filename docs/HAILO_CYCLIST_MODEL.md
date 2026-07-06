# Hailo model pro silničního cyklistu

Pro MotoCam je nejlepší začít s reálným Hailo HEF modelem, který umí detekovat cyklisty nebo alespoň osoby/bicykly. V praxi doporučuji jednu z těchto cest:

## Varianta A: rychlý start se stock COCO HEF

Tento model neobsahuje `cyclist`, ale detekuje `bicycle` a `person`.

1. Na Raspberry Pi:

```bash
sudo apt update
sudo apt install -y hailo-all
sudo reboot
```

2. Po restartu:

```bash
cd /path/to/motocam
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/setup_hailo.sh
```

3. V [config/config.yaml](config/config.yaml) použij:

```yaml
ai:
  type: hailo_canary
  model: models/yolov8n.hef
  target_class: bicycle
```

## Varianta B: vlastní cyclist model

If you already have a trained detector or a converted Hailo-compatible model, place it as `.hef` and run:

```bash
bash scripts/prepare_cyclist_hef.sh /path/to/your_model.hef
```

Potom v [config/config.yaml](config/config.yaml) zůstane:

```yaml
ai:
  type: hailo_canary
  model: models/cyclist.hef
  target_class: cyclist
```

## Jak získat vlastní model

1. Trénink/konverze v ONNX.
2. Kompilace do Hailo HEF pomocí Hailo Dataflow Compiler / Hailo Model Zoo.
3. Výstup uložit jako `.hef`.

## Poznámka

Na tomto stroji zatím nejde otestovat skutečný Hailo runtime, protože je to macOS a ne Raspberry Pi s AI HAT+.
