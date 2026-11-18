"""Run the head-camera, carried-box and staircase reconstruction benchmark."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from wtumi_recon.occlusion_demo import run_occlusion_demo


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Kinematic staircase occlusion benchmark")
    parser.add_argument('--config', type=Path, default=root/'configs/demos/stairs_occlusion.yaml')
    parser.add_argument('--output', type=Path, help='New/empty directory; otherwise a timestamped outputs directory')
    args = parser.parse_args()
    out = args.output or Path('outputs')/f"occlusion_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        report = run_occlusion_demo(args.config, out, progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'Occlusion demo error: {exc}\n')
    print(json.dumps({'output': str(out.resolve()), 'paths': report['paths'],
                      'paired_inputs_verified': report['paired_inputs_verified']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
