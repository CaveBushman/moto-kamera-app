"""Dark, glove-friendly touch UI theme (design doc 11.1) -- "glassmorphism
lite": translucent frosted-looking cards, soft gradients and glow borders
for secondary chrome (panels, dialogs, the joystick), real depth added via
effects.py's drop shadows since Qt stylesheets can't do backdrop blur.

Deliberately NOT applied to the two places where it would fight the
design doc's own requirements (11.1 glanceable in sunlight, gloves, 24.4
readability risk): the mode bar's big action buttons and the status chips
stay high-contrast and effectively opaque, because those are read/tapped
at a glance while riding, not admired at a desk.
"""

ACCENT = "#0077ff"
ACCENT_2 = "#00c2ff"

DARK_STYLESHEET = f"""
QWidget {{
    background-color: #090b10;
    color: #eef1f6;
    font-family: "DejaVu Sans", "Segoe UI", sans-serif;
    font-size: 16px;
}}

QMainWindow, QDialog {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #12151b, stop:0.55 #090b10, stop:1 #050609);
}}

QWidget#topBar {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(18, 19, 23, 252), stop:0.52 rgba(10, 12, 17, 252), stop:1 rgba(21, 20, 25, 252));
    border-bottom: 1px solid rgba(255, 255, 255, 36);
}}

QLabel#unitBadge {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #005eff, stop:1 #00a6ff);
    border: 1px solid rgba(255, 255, 255, 120);
    border-radius: 10px;
    padding: 7px 16px;
}}

QLabel.statusChip {{
    background-color: rgba(17, 20, 26, 238);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 8px;
    padding: 5px 10px;
    font-weight: bold;
}}

QLabel.statusOk {{
    border: 1px solid rgba(61, 220, 132, 150);
    background-color: rgba(20, 48, 34, 235);
}}
QLabel.statusWarn {{
    border: 1px solid rgba(255, 176, 32, 160);
    background-color: rgba(61, 43, 16, 235);
}}
QLabel.statusBad {{
    border: 1px solid rgba(255, 77, 77, 170);
    background-color: rgba(75, 22, 28, 238);
}}
QLabel.statusIdle {{
    border: 1px solid rgba(255, 255, 255, 42);
    background-color: rgba(17, 20, 26, 230);
}}

/* Regular buttons (settings, diag, panel controls): soft gradient +
   glass border, rounded. Kept high-enough-contrast to stay legible. */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(39, 43, 52, 245), stop:1 rgba(21, 24, 30, 245));
    border: 1px solid rgba(255, 255, 255, 54);
    border-radius: 8px;
    padding: 13px 16px;
    font-size: 18px;
    font-weight: 850;
    min-height: 40px;
}}

QPushButton:hover {{
    border: 1px solid rgba(255, 255, 255, 70);
}}

QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(30, 33, 40, 235), stop:1 rgba(24, 26, 32, 235));
}}

QPushButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT_2});
    border: 1px solid rgba(255, 255, 255, 140);
    color: white;
}}

QPushButton#recordButton {{
    min-height: 42px;
    color: #eef1f6;
}}

QPushButton#recordButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #c1121f, stop:1 #ff4d4d);
    border: 1px solid rgba(255, 255, 255, 150);
}}

QPushButton#autofocusButton {{
    min-height: 42px;
    color: #dff8ff;
    border: 1px solid rgba(0, 194, 255, 110);
}}

QPushButton#autofocusButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #004a8f, stop:1 #0087b8);
}}

QPushButton#manualOverride {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #b91c1c, stop:1 #7f1212);
    border: 1px solid rgba(255, 120, 120, 150);
    color: white;
}}

QPushButton#manualOverride:hover {{
    border: 1px solid rgba(255, 160, 160, 180);
}}

/* Frosted glass cards for grouped controls */
QGroupBox {{
    background-color: rgba(15, 18, 24, 232);
    border: 1px solid rgba(255, 255, 255, 42);
    border-top: 2px solid rgba(0, 194, 255, 130);
    border-radius: 8px;
    margin-top: 16px;
    font-weight: 900;
    padding: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 2px 9px;
    background-color: rgba(9, 11, 16, 215);
    border-radius: 5px;
    color: #9aa6b8;
    font-size: 13px;
}}

QFrame#previewFrame {{
    background-color: black;
    border: 1px solid rgba(255, 255, 255, 56);
    border-radius: 0px;
}}

QWidget#previewHud {{
    background: transparent;
}}

/* CAMERA/GIMBAL readouts float directly over the live feed (see
   MainWindow._build_ui / PreviewView.set_hud_widget) so the video stays
   the centerpiece instead of being squeezed into a strip above a solid
   panel row. Meaningfully more see-through than the standard QGroupBox
   so the feed actually reads behind them, not just a darker video. */
QGroupBox#cameraPanel, QGroupBox#gimbalPanel {{
    background-color: rgba(15, 18, 24, 145);
}}

/* Always-visible pill that brings the CAM/GIMBAL HUD back -- it's off by
   default so the feed stays unobstructed, so this needs to read clearly
   against live video without looking like part of the chrome below it. */
QPushButton#hudToggle {{
    background-color: rgba(10, 12, 17, 190);
    border: 1px solid rgba(255, 255, 255, 90);
    border-radius: 16px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 800;
    min-height: 0px;
}}

QPushButton#hudToggle:checked {{
    background: rgba(0, 119, 255, 210);
    border: 1px solid rgba(255, 255, 255, 140);
    color: white;
}}

/* Only shown while a target is actually locked/weak -- warm/alert tint
   so it reads as "this will drop your target", distinct from the neutral
   CAM/GIMBAL and SETTINGS pills. */
QPushButton#cancelTrackButton {{
    background-color: rgba(146, 32, 32, 215);
    border: 1px solid rgba(255, 180, 180, 150);
    border-radius: 16px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 800;
    min-height: 0px;
    color: white;
}}

QPushButton#cancelTrackButton:hover {{
    background-color: rgba(180, 40, 40, 230);
}}

QFrame#metricTile {{
    background-color: rgba(6, 8, 12, 205);
    border: 1px solid rgba(255, 255, 255, 34);
    border-radius: 7px;
}}

QLabel.metricTitle {{
    color: #8d97a8;
    font-size: 10px;
    font-weight: 900;
}}

QLabel.metricValue {{
    color: #f5f7fb;
    font-size: 18px;
    font-weight: 950;
}}

QLabel.connectionBadge {{
    border-radius: 7px;
    padding: 10px 12px;
    font-weight: 900;
    qproperty-alignment: AlignCenter;
}}

QLabel.ok {{
    background-color: rgba(26, 78, 47, 230);
    border: 1px solid rgba(61, 220, 132, 150);
    color: #dfffee;
}}

QLabel.bad {{
    background-color: rgba(83, 25, 30, 230);
    border: 1px solid rgba(255, 77, 77, 150);
    color: #ffe5e5;
}}

QWidget#modeBar {{
    background-color: #07090d;
    border-top: 1px solid rgba(255, 255, 255, 36);
}}

QWidget#modeBar QPushButton {{
    min-height: 48px;
    border-radius: 8px;
    font-size: 18px;
}}

QWidget#modeBar QPushButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #006cff, stop:1 #00c2ff);
}}

/* Native macOS/Fusion styles don't reliably inherit QWidget's color on
   editable input widgets, which was leaving IP/port entry effectively
   invisible (dark-on-dark) in Settings -- style them explicitly. */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: rgba(20, 22, 28, 230);
    color: #eef1f6;
    border: 1px solid rgba(255, 255, 255, 35);
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: {ACCENT};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT_2};
}}

QComboBox QAbstractItemView {{
    background-color: #181b22;
    color: #eef1f6;
    selection-background-color: {ACCENT};
    outline: none;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
}}

QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 50);
    border-radius: 6px;
    min-height: 24px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
