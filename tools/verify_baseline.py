#!/usr/bin/env python3
"""Read-only source/checkpoint verification. Never imports the robot SDK.

--gpu checks CUDA availability only; it does not load or run a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('host', 'robot', 'all'), default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'releases/right7_20261113/BASELINE_MANIFEST.json').read_text())
    errors, checked = [], []
    for entry in manifest['source_files']:
        if args.role != 'all' and args.role not in entry['roles']:
            continue
        path = root/entry['path']
        if not path.is_file() or digest(path) != entry['sha256']:
            errors.append('source_missing_or_changed:'+entry['path'])
        checked.append(entry['path'])
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        for entry in manifest['checkpoint_files']:
            path = checkpoint/entry['path']
            if not path.is_file() or digest(path) != entry['sha256']:
                errors.append('checkpoint_missing_or_changed:'+entry['path'])
    gpu = None
    if args.gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append('CUDA unavailable; do not start training or inference')
            else:
                gpu = {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__}
        except ImportError:
            errors.append('PyTorch is not installed in this Python environment')
    print(json.dumps({'baseline':manifest['baseline_id'], 'role':args.role,
                      'source_files_checked':checked, 'checkpoint_checked':bool(args.checkpoint),
                      'gpu':gpu, 'errors':errors, 'hardware_connected':False}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
