"""Compare visual histories before and after carried-box occlusion."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from wtumi_recon.memory_demo import run_memory_demo


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Observation memory under carried-box occlusion')
    parser.add_argument('--config', type=Path, default=root/'configs/demos/stairs_memory.yaml')
    parser.add_argument('--output', type=Path, help='New/empty output directory')
    parser.add_argument('--capture', type=Path, help='Reuse a saved M6 capture directory without launching MuJoCo')
    args = parser.parse_args()
    out = args.output or Path('outputs')/f"memory_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        report = run_memory_demo(args.config,out,capture_dir=args.capture,progress=lambda message:print(message,flush=True))
    except (ValueError,OSError,KeyError,ImportError,RuntimeError) as exc:
        parser.exit(2,f'Memory demo error: {exc}\n')
    print(json.dumps({'output':str(out.resolve()),'paths':report['paths'],
                      'paired_inputs_verified':report['paired_inputs_verified']},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
