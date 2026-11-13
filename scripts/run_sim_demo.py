"""Run simulator capture, voxel-size sweep, comparisons and a 3D viewer."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from wtumi_recon.demo import run_sim_demo


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Static simulator RGB-D reconstruction demo')
    parser.add_argument('--config', type=Path, default=root/'configs/demos/mujoco_room.yaml')
    parser.add_argument('--output', type=Path, help='New/empty directory; defaults to outputs/sim_demo_<timestamp>')
    args = parser.parse_args()
    out = args.output or Path('outputs')/f"sim_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        report = run_sim_demo(args.config, out, progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'Demo error: {exc}\n')
    print(json.dumps({'output': str(out.resolve()), 'scene': report['scene'],
                      'paths': report['paths'], 'paired_inputs_verified': report['paired_inputs_verified']}, indent=2))


if __name__ == '__main__':
    main()
