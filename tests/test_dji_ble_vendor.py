from motocam.gimbal.dji_ble_vendor import DjiBleVendorAssembler, DjiBleVendorFrame


def test_vendor_assembler_splits_complete_frames():
    raw = bytes.fromhex("551204c7040219060004270084000021e42f")
    frames = DjiBleVendorAssembler().feed(raw)
    assert frames == [DjiBleVendorFrame(raw)]
    assert frames[0].length == 0x12
    assert frames[0].signature == "04c70402"


def test_vendor_assembler_reassembles_split_frames():
    assembler = DjiBleVendorAssembler()
    assert assembler.feed(bytes.fromhex("551204c704")) == []
    frames = assembler.feed(bytes.fromhex("0219060004270084000021e42f"))
    assert len(frames) == 1
    assert frames[0].raw.hex() == "551204c7040219060004270084000021e42f"


def test_vendor_assembler_resyncs_after_noise():
    assembler = DjiBleVendorAssembler()
    frames = assembler.feed(bytes.fromhex("000102551204c7040219060004270084000021e42f"))
    assert len(frames) == 1
    assert frames[0].signature == "04c70402"


def test_vendor_frame_capture_dict_is_stable():
    frame = DjiBleVendorFrame(bytes.fromhex("553304c2e4020000001c010701000100020001ff03000100021004ffffffff031004ffffffff041004ffffffff012001ffc253"))
    assert frame.to_capture_dict()["len"] == 0x33
    assert frame.to_capture_dict()["signature"] == "04c2e402"
    assert frame.to_capture_dict()["first_payload_hex"] == "0000001c01070100"
