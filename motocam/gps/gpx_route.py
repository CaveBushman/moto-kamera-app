"""Finish-line extraction from a race-route GPX file.

The finish-zone peel-off rule (gps/finish_zone.py) needs the finish
line's coordinates. Typing them into Settings by hand proved to be the
wrong interface for race day -- the route already exists as a GPX file
(the same one the control room plans the stage with), so the moto unit
reads the finish straight from it: drop the stage's .gpx into `data/`
and the newest one is picked up at app start. No manual entry, no
"operator armed Null Island by accident" class of mistakes.

Finish resolution, in priority order:
1. A waypoint (<wpt>) whose name says it's the finish ("finish", "cíl",
   "cil", "arrivee", "goal") -- explicit beats implicit.
2. The last trackpoint of the last track segment (<trk>) -- a stage
   route's track ends at the finish line by definition.
3. The last routepoint (<rte>) -- same logic for route-typed GPX.

Parsing matches on local tag names (namespace-stripped) so GPX 1.0, 1.1
and no-namespace exports all work; a malformed file degrades to "no
finish from GPX" with a warning, never an exception into startup. A
director STAGE_INFO push still overrides whatever the GPX said -- the
control room knows about a mid-race finish change before the file does.
"""
from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("motocam.gps.gpx")

DEFAULT_GPX_DIR = "data"
FINISH_WAYPOINT_NAMES = {"finish", "cíl", "cil", "arrivee", "arrivée", "goal", "finish line", "finishline"}


@dataclass(frozen=True)
class GpxFinish:
    lat: float
    lon: float
    source: str  # human-readable: which file + which rule matched


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _point_coords(element: ET.Element) -> tuple[float, float] | None:
    try:
        lat, lon = float(element.get("lat", "")), float(element.get("lon", ""))
    except ValueError:
        return None
    if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def parse_finish(gpx_path: str | Path) -> GpxFinish | None:
    """Extract the finish coordinates from one GPX file, or None if the
    file is unreadable/malformed or holds no usable points."""
    path = Path(gpx_path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        logger.warning("GPX file %s could not be parsed: %s", path, exc)
        return None

    finish_waypoint: tuple[float, float] | None = None
    last_trackpoint: tuple[float, float] | None = None
    last_routepoint: tuple[float, float] | None = None

    for element in root.iter():
        name = _local_name(element.tag)
        if name == "wpt":
            wpt_name = ""
            for child in element:
                if _local_name(child.tag) == "name" and child.text:
                    wpt_name = child.text.strip().lower()
                    break
            if wpt_name in FINISH_WAYPOINT_NAMES:
                coords = _point_coords(element)
                if coords is not None:
                    finish_waypoint = coords
        elif name == "trkpt":
            coords = _point_coords(element)
            if coords is not None:
                last_trackpoint = coords
        elif name == "rtept":
            coords = _point_coords(element)
            if coords is not None:
                last_routepoint = coords

    if finish_waypoint is not None:
        return GpxFinish(*finish_waypoint, source=f"{path.name} (finish waypoint)")
    if last_trackpoint is not None:
        return GpxFinish(*last_trackpoint, source=f"{path.name} (last trackpoint)")
    if last_routepoint is not None:
        return GpxFinish(*last_routepoint, source=f"{path.name} (last routepoint)")
    logger.warning("GPX file %s contains no usable finish point", path)
    return None


def find_route_gpx(gpx_dir: str | Path = DEFAULT_GPX_DIR) -> Path | None:
    """Newest *.gpx directly in `gpx_dir` or one subdirectory level down
    -- 'newest' by mtime, so copying today's stage file over yesterday's
    just works without any cleanup ritual.

    Deliberately NOT a recursive rglob: the default gpx_dir is `data/`,
    whose training_capture/ subtree grows by ~1800 files per ride-hour
    when capture is enabled -- an unbounded recursive walk + stat of the
    whole tree ran synchronously in MainWindow.__init__, i.e. seconds of
    black screen at startup on race morning. One level of subdirectories
    still supports a tidy `data/routes/` layout without paying for the
    capture archive."""
    directory = Path(gpx_dir)
    if not directory.is_dir():
        return None
    candidates = list(directory.glob("*.gpx")) + list(directory.glob("*/*.gpx"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_finish_from_gpx(gpx_dir: str | Path = DEFAULT_GPX_DIR) -> GpxFinish | None:
    """Find the newest route GPX and pull the finish out of it. Returns
    None (with a log line, not an exception) when there's no file or no
    usable point -- startup must proceed either way."""
    path = find_route_gpx(gpx_dir)
    if path is None:
        logger.info("No route GPX found in %s -- finish zone stays unset until a STAGE_INFO push", gpx_dir)
        return None
    finish = parse_finish(path)
    if finish is not None:
        logger.info("Finish line loaded from %s: %.6f, %.6f", finish.source, finish.lat, finish.lon)
    return finish
