"""Telemetry source labels shared by the rider UI and wire packets."""
from __future__ import annotations

FALLBACK_SOURCE_VALUES = frozenset({"unknown", "mock", "simulated", "synthetic", "null"})
SOURCE_VALUE_LABELS = {
    "unknown": "UNK",
    "simulated": "SIM",
    "synthetic": "SYN",
    "null": "NULL",
    "null_disabled": "OFF",
    "null_model": "NO HEF",
    "null_runtime": "NO HAILO",
    "null_error": "AI ERR",
    "dev_hef": "DEV HEF",
    "mock": "MOCK",
}


def normalise_source(value: str | None) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def source_value_display(value: str | None) -> str:
    normalized = normalise_source(value)
    return SOURCE_VALUE_LABELS.get(normalized, normalized.upper())


def is_fallback_source(value: str | None) -> bool:
    normalized = normalise_source(value)
    return normalized in FALLBACK_SOURCE_VALUES or normalized.startswith("null_")
