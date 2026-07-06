#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    import numpy as np
    from hailo_platform import HEF, VDevice, FormatType
except Exception as exc:  # noqa: BLE001
    print(f'Hailo runtime unavailable: {exc}')
    sys.exit(2)

hef_path = '/motocam/ai/models/cyclist.hef'
if not Path(hef_path).exists():
    print(f'HEF not found: {hef_path}')
    sys.exit(3)

print('Import OK')
params = VDevice.create_params()
vdevice = VDevice(params)
hef = HEF(hef_path)
print('HEF loaded:', hef_path)
print('Input infos:', hef.get_input_vstream_infos())

try:
    infer_model = vdevice.create_infer_model(hef_path)
    infer_model.set_batch_size(1)
    infer_model.input().set_format_type(FormatType.UINT8)
    with infer_model.configure() as configured_model:
        print('Configured model OK')
except Exception as exc:  # noqa: BLE001
    print(f'Configure failed: {exc}')
    sys.exit(4)

print('Smoke test passed')
