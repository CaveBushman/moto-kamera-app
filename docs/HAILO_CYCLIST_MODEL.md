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

## Jak nasbírat vlastní trénovací data

MotoCam umí za jízdy sbírat trénovací příklady přímo z toho, co operátor
sám potvrdí -- ne syrové detekce stock modelu (ten stejně zná jen
`bicycle`/`person`, ne `cyclist`), ale box, na který jsi klepnul
(tap-to-select) nebo který FULL AI automaticky zamklo a appka ho dál
sleduje. Tohle je ground truth od člověka, ne hádanka modelu.

1. V Settings -> AI TRACKING zapni "Collect AI training data while
   tracking" (nebo `ai.training_capture.enabled: true` v `config.yaml`).
2. Jeď, sleduj cyklisty (AI ASSIST s klepnutím, nebo FULL AI). Appka
   ukládá snímek + anotaci vždy, když je tracker ve stavu LOCKED (aktivně
   potvrzený, ne dobíhající přes výpadek), max jednou za
   `ai.training_capture.interval_s` (výchozí 2 s), aby data nezaplnila
   kartu skoro identickými snímky.
3. Výsledek je ve formátu Ultralytics YOLO v `data/training_capture/`
   (`<timestamp>.jpg` + `<timestamp>.txt` + `classes.txt`) -- připravené
   rovnou k tréninku, žádná ruční konverze.
4. Přenes `data/training_capture/` na stroj s GPU pro trénink (viz níže).

### Peloton jako druhá třída

Na startu závodu je cíl skoro vždy peloton (skupina), ne jeden jezdec --
appka to zvládne stejným mechanismem, jen s jinou třídou:

1. V Settings -> AI TRACKING přepni "Target class" na `peloton` ještě
   před tap-to-select na startu (combo box je editovatelný, `peloton` je
   už v seznamu). Klepnutím označíš box kolem celé skupiny.
2. Dokud je tracker LOCKED, capture ukládá snímky s třídou podle
   aktuálně vybraného Target class -- tedy `peloton`, ne `cyclist`.
   `classes.txt` roste o nový řádek při prvním výskytu nové třídy a
   pořadí (index 0, 1, ...) se nikdy nepřepisuje, i když si řádky ručně
   doplníš.
3. Po rozjetí závodu (rozpad pelotonu, únik) přepni Target class zpět na
   `cyclist` a pokračuj ve sběru dat pro jednotlivce.
4. Výsledný `data/training_capture/` tak obsahuje obě třídy pohromadě --
   trénink pak žene jeden YOLO model, co pozná `cyclist` i `peloton`
   zvlášť.

Fotky nahrané ručně (ne appkou) do stejné složky musí použít stejné
indexy tříd jako `classes.txt` už obsahuje -- zkontroluj ho, než je
doanotuješ.

## Jak získat vlastní model

1. Trénink/doladění (fine-tuning) YOLOv8/v11 na nasbíraných datech --
   na stroji s GPU, ne na Pi (na CPU by to trvalo dny až týdny).
2. Export do ONNX.
3. Kompilace do Hailo HEF pomocí Hailo Dataflow Compiler / Hailo Model Zoo
   -- taky mimo Pi, DFC běží na x86_64.
4. Výstup uložit jako `.hef` a nainstalovat přes
   `bash scripts/prepare_cyclist_hef.sh /path/to/your_model.hef`.

AI HAT+ (Hailo-8) sám o sobě neumí trénovat -- je to čistě inferenční
akcelerátor (rychlý dopředný průchod hotovou sítí), ne trénovací
hardware. Trénink vždy probíhá mimo Pi.

## Poznámka

Na tomto stroji zatím nejde otestovat skutečný Hailo runtime, protože je to macOS a ne Raspberry Pi s AI HAT+.
