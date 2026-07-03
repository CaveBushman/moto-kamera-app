"""Mirror of controlroom's tests/test_protocol_parity.py, run from this
project so either repo's test suite catches wire-format drift on its
own -- motocam/core/protocol.py and the sibling control room project's
core/protocol.py are two independently maintained copies of the same
wire format (no shared Python package between the projects).

Skips (rather than fails) if the sibling project isn't checked out next
to this one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

MOTOCAM_PROTOCOL = Path(__file__).resolve().parents[1] / "motocam" / "core" / "protocol.py"
CONTROLROOM_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "08 livestream road cycling control room" / "controlroom" / "core" / "protocol.py"
)


def _dataclass_fields(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text())
    result: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
            if fields:
                result[node.name] = fields
    return result


@pytest.mark.skipif(
    not CONTROLROOM_PROTOCOL.is_file(), reason="sibling control room project not checked out alongside this one"
)
def test_wire_protocol_dataclasses_match_sibling_project():
    motocam_fields = _dataclass_fields(MOTOCAM_PROTOCOL)
    controlroom_fields = _dataclass_fields(CONTROLROOM_PROTOCOL)

    common_classes = set(motocam_fields) & set(controlroom_fields)
    assert common_classes, "expected at least one dataclass name shared between both protocol.py files"

    mismatches = {
        name: (motocam_fields[name], controlroom_fields[name])
        for name in common_classes
        if motocam_fields[name] != controlroom_fields[name]
    }
    assert not mismatches, f"protocol.py drift between the two projects: {mismatches}"
