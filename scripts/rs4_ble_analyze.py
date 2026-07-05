#!/usr/bin/env python3
"""Analyze captures produced by scripts/rs4_ble_capture.py."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_events(paths: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{path}:{line_no}: bad JSON: {exc}", file=sys.stderr)
                    continue
                event["_file"] = str(path)
                events.append(event)
    return events


def varying_positions(frames: list[bytes]) -> list[tuple[int, list[int]]]:
    if len(frames) < 2:
        return []
    min_len = min(len(frame) for frame in frames)
    positions: list[tuple[int, list[int]]] = []
    for index in range(min_len):
        values = sorted({frame[index] for frame in frames})
        if len(values) > 1:
            positions.append((index, values[:12]))
    return positions


def pct(value: int, total: int) -> str:
    return f"{(100.0 * value / total):.1f}%" if total else "0.0%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", help="JSONL capture file(s)")
    parser.add_argument("--top", type=int, default=12, help="Rows per section")
    args = parser.parse_args()

    paths = [Path(item) for item in args.captures]
    events = load_events(paths)
    notifications = [event for event in events if event.get("event") == "notification"]
    frame_events = [event for event in events if event.get("event") == "vendor_frame"]

    print("== RS 4 Pro BLE capture analysis ==")
    print(f"files: {len(paths)}")
    print(f"events: {len(events)}")
    print(f"notifications: {len(notifications)}")
    print(f"vendor frames: {len(frame_events)}")

    by_char = Counter(str(event.get("short_uuid") or event.get("char")) for event in notifications)
    if by_char:
        print("\nNotifications by characteristic:")
        for key, count in by_char.most_common(args.top):
            print(f"  {key:<8} {count:>6}  {pct(count, len(notifications))}")

    by_signature = Counter(str(event.get("signature")) for event in frame_events)
    by_len = Counter(int(event.get("len", 0)) for event in frame_events)
    if by_len:
        print("\nFrame lengths:")
        for key, count in by_len.most_common(args.top):
            print(f"  {key:>3} bytes  {count:>6}  {pct(count, len(frame_events))}")

    if by_signature:
        print("\nFrame signatures:")
        for sig, count in by_signature.most_common(args.top):
            sample = next(event for event in frame_events if event.get("signature") == sig)
            print(
                f"  {sig:<10} count={count:<6} len={sample.get('len')} "
                f"first_payload={sample.get('first_payload_hex')}"
            )

    grouped: dict[str, list[bytes]] = defaultdict(list)
    for event in frame_events:
        try:
            raw = bytes.fromhex(str(event.get("hex", "")))
        except ValueError:
            continue
        grouped[str(event.get("signature"))].append(raw)

    if grouped:
        print("\nVarying byte positions by signature:")
        for sig, frames in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[: args.top]:
            variations = varying_positions(frames)
            if not variations:
                print(f"  {sig:<10} constant across {len(frames)} frame(s)")
                continue
            rendered = ", ".join(
                f"{index}:[" + ",".join(f"{value:02x}" for value in values[:8]) + "]"
                for index, values in variations[:16]
            )
            extra = "" if len(variations) <= 16 else f" +{len(variations) - 16} more"
            print(f"  {sig:<10} frames={len(frames):<6} {rendered}{extra}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
