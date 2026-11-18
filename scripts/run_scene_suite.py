"""Run room, doorway and G1 stairs with synchronized camera/voxel videos."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from wtumi_recon.scene_suite import SCENES, run_scene_suite, scene_replacement_plan, suite_preset_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--preset', choices=('default', 'pace30'), default='default',
                        help='pace30 captures 121 fresh 640x480 observations per scene and exports 30 fps MP4')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and print the capture plan without rendering, encoding or creating outputs')
    parser.add_argument('--replace-scenes', nargs='+', choices=SCENES,
                        help='Recapture and reconstruct only these scenes in an existing completed --output suite')
    args = parser.parse_args()
    if args.replace_scenes:
        if args.output is None:
            parser.error('--replace-scenes requires an explicit --output')
        if len(args.replace_scenes) != len(set(args.replace_scenes)):
            parser.error('--replace-scenes must contain unique scene names')
    output = args.output or Path('outputs') / f'g1_vision_suite_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.dry_run:
        plan = (scene_replacement_plan(output, args.replace_scenes, args.preset)
                if args.replace_scenes else suite_preset_plan(args.preset))
        print(json.dumps({'output': str(output.resolve()), **plan}, indent=2))
        return
    options = {} if args.preset == 'default' else {'preset': args.preset}
    if args.replace_scenes:
        options['replace_scenes'] = args.replace_scenes
    report = run_scene_suite(output, progress=lambda text: print(text, flush=True), **options)
    print(json.dumps({'output': str(output.resolve()), 'paths': report['paths']}, indent=2))


if __name__ == '__main__':
    main()
