from __future__ import annotations

import pytest

from motocam.core.config import load_config, resolve_config_path, save_config


def test_resolve_config_path_returns_explicit_path_when_given(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("unit_id: moto-2\n", encoding="utf-8")
    assert resolve_config_path(config_file) == config_file


def test_resolve_config_path_raises_when_nothing_found(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        resolve_config_path(missing)


def test_load_config_parses_yaml_into_a_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("unit_id: moto-3\ncamera:\n  ip: 192.168.9.20\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg["unit_id"] == "moto-3"
    assert cfg["camera"]["ip"] == "192.168.9.20"


def test_load_config_returns_empty_dict_for_empty_file(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")
    assert load_config(config_file) == {}


def test_save_config_then_load_config_round_trips(tmp_path):
    config_file = tmp_path / "nested" / "config.yaml"
    original = {"unit_id": "moto-4", "video": {"device": 0, "fps": 30}}
    save_config(original, config_file)
    assert config_file.is_file()
    assert load_config(config_file) == original


def test_save_config_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "config.yaml"
    save_config({"unit_id": "moto-1"}, target)
    assert target.is_file()
