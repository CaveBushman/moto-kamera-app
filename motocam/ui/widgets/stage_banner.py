"""Race stage heading (e.g. "STAGE 12: PRAHA - JESTED"), set by the
director in the control room and pushed down over the link -- the moto
app never originates this, only displays whatever it was last told
(handled in MainWindow via MessageType.STAGE_INFO)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel


class StageBanner(QLabel):
    def __init__(self):
        super().__init__("NO STAGE SET")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(42)
        self.setStyleSheet(
            "background-color: rgba(7, 9, 13, 244); color: #8f98a8; "
            "font-size: 17px; font-weight: 900; padding: 8px 14px; "
            "border-bottom: 1px solid rgba(255, 255, 255, 32);"
        )

    def set_stage(self, stage_number: int, stage_name: str, laps: int = 0) -> None:
        # Stage 0 means "no number set" (a test/untitled session) -- showing
        # "STAGE 0: FOO" would read as a real stage number, so the prefix
        # is dropped and only the name (+ laps, if any) shows.
        laps_text = f" -- {laps} LAPS" if laps > 0 else ""
        name = stage_name.upper()
        text = f"STAGE {stage_number}: {name}{laps_text}" if stage_number > 0 else f"{name}{laps_text}"
        self.setText(text)
        self.setStyleSheet(
            "background-color: rgba(7, 9, 13, 248); color: #f5f7fb; "
            "font-size: 20px; font-weight: 950; padding: 8px 14px; "
            "border-bottom: 1px solid rgba(255, 255, 255, 38); "
            "border-top: 2px solid rgba(61, 220, 132, 170);"
        )
