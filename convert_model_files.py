"""Clean up the original model files from ultralytics/yolov5.

This script will download the original model files, extract the state_dict (model parameters),
ensure that values are CPU float32, and then save the state_dict to a new model file with the
SHA256 checksum prefix in its name.
"""

import hashlib
import os
import os.path
import sys
import tempfile

import torch

MODEL_NAMES = [
    'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
    'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6',
]


def main():
    legacy_package_dir = os.path.join(os.path.dirname(__file__), 'yolov5')
    sys.path.append(legacy_package_dir)

    for name in MODEL_NAMES:
        ckpt = torch.hub.load_state_dict_from_url(
            f'https://github.com/ultralytics/yolov5/releases/download/v5.0/{name}.pt',
            map_location=torch.device('cpu'),
        )
        state_dict = ckpt['model'].float().state_dict()
        tmp_filename = tempfile.mktemp(suffix='.pth', dir='.')
        torch.save(state_dict, tmp_filename)
        with open(tmp_filename, 'rb') as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()
        os.rename(tmp_filename, f'{name}-{sha256_hash[:8]}.pth')


if __name__ == '__main__':
    main()
