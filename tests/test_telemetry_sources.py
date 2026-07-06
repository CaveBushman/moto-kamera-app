from motocam.core.telemetry_sources import is_fallback_source, source_value_display


def test_source_value_display_compacts_fallbacks():
    assert source_value_display("simulated") == "SIM"
    assert source_value_display("synthetic") == "SYN"
    assert source_value_display("mock") == "MOCK"
    assert source_value_display("null") == "NULL"
    assert source_value_display("null_model") == "NO HEF"
    assert source_value_display("null_runtime") == "NO HAILO"
    assert source_value_display("dev_hef") == "DEV HEF"


def test_source_value_display_preserves_real_backend_identity():
    assert source_value_display("bmd-rest") == "BMD-REST"
    assert source_value_display("dji-rsdk-ble") == "DJI-RSDK-BLE"


def test_is_fallback_source():
    assert is_fallback_source("simulated")
    assert is_fallback_source("unknown")
    assert is_fallback_source("null_model")
    assert not is_fallback_source("real")
    assert not is_fallback_source("hailo")
