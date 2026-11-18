"""Create, run and inspect a portable single-scene reconstruction experiment."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


def _mac_view_runtime():
    """MuJoCo's passive GUI needs its bundled main-thread launcher on macOS."""
    if sys.platform != 'darwin' or os.environ.get('MJPYTHON_BIN'):
        return
    launcher = Path(sys.executable).parent / 'mjpython'
    if not launcher.is_file():
        raise RuntimeError('MuJoCo GUI launcher not found. Activate the project environment and install .[sim].')
    # Invoke the Python trampoline explicitly: a venv path containing spaces
    # may have an unusable launcher shebang on macOS.
    os.execve(sys.executable, [sys.executable, str(launcher), str(Path(__file__).resolve()), *sys.argv[1:]], os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List bundled templates')
    create = commands.add_parser('create', help='Copy a template into a new editable scene bundle')
    create.add_argument('--template', required=True, choices=['room', 'stairs', 'doorway', 'g1_stairs'])
    create.add_argument('--output', type=Path, required=True)
    for name in ('build', 'run'):
        command = commands.add_parser(name, help='Generate model.xml' if name == 'build' else 'Capture RGB-D and reconstruct each prefix')
        command.add_argument('--scene', required=True, help='Template name, bundle directory, or scene.yaml')
        command.add_argument('--output', type=Path)
    view = commands.add_parser('view', help='Open the native MuJoCo scene/reconstruction viewer')
    selection = view.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene', help='View an original scene and its camera path')
    selection.add_argument('--result', type=Path, help='View a completed scene run and its saved reconstruction history')
    args = parser.parse_args()
    try:
        if args.command == 'view':
            _mac_view_runtime()
            from wtumi_recon.scene_viewer import launch_scene_viewer
            launch_scene_viewer(scene_ref=args.scene, result_dir=args.result)
            return
        from wtumi_recon.scene_bundle import build_scene, create_scene
        if args.command == 'list':
            print('room       6.4 x 10 m room, center half-wall and Unitree G1\nstairs     Staircase with a carried-box occluder\ndoorway    6.4 x 10 m room, hinged doorway and Unitree G1\ng1_stairs  Unitree G1 visual model, head camera, box and stairs')
            return
        if args.command == 'create':
            result = create_scene(args.template, args.output)
            print(json.dumps(result, indent=2, default=str))
            return
        out = args.output or Path('outputs') / f"scene_{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if args.command == 'build':
            result = build_scene(args.scene, out)
        else:
            from wtumi_recon.scene_pipeline import run_scene
            result = run_scene(args.scene, out, progress=lambda message: print(message, flush=True))
        print(json.dumps({'output': str(out.resolve()), 'paths': result['paths']}, indent=2))
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'Scene error: {exc}\n')


if __name__ == '__main__':
    main()
