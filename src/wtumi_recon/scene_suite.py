"""Run and summarize the three G1 vision-to-voxel scene experiments."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
import shutil
import tempfile


SCENES = ('room', 'doorway', 'g1_stairs')
_TRANSACTION = '.scene-replacement'
_REPORT_FILES = ('index.html', 'summary.csv', 'README.md', 'preset.json', 'suite.json')


def _relative_artifact(directory, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Suite artifact path escapes its scene directory')
    path = (directory / value).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Suite artifact path escapes its scene directory')
    return path


def _replacement_names(value):
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value):
        raise ValueError('replace_scenes must contain unique scene names')
    if any(name not in SCENES for name in value):
        raise ValueError('replace_scenes supports room, doorway and g1_stairs only')
    return tuple(name for name in SCENES if name in value)


def _existing_suite(output_dir, selected):
    """Recognize complete saved maps; selected display media may be absent."""
    from .scene_bundle import load_scene_config
    from .scene_comparison import comparison_inputs

    path = Path(output_dir).expanduser().absolute()
    if '..' in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Replacement root must not use symlinks or parent-directory traversal')
    root = path.resolve()
    if not root.is_dir():
        raise ValueError('Replacement requires an existing complete scene suite directory')
    transaction = root / _TRANSACTION
    if transaction.exists() or transaction.is_symlink():
        raise RuntimeError(f'Interrupted replacement found at {transaction}. Inspect journal.json and its staged/backup paths before rerunning; no automatic deletion was performed.')
    if any(item.is_symlink() for item in root.rglob('*')):
        raise ValueError('Replacement suite must not contain symlinks')
    report = json.loads((root / 'suite.json').read_text(encoding='utf-8'))
    if (report.get('format') != 'wtumi-g1-scene-suite-v1'
            or [row.get('scene') for row in report.get('scenes', [])] != list(SCENES)
            or any(report.get('paths', {}).get(name) != f'{name}/scene_run.json' for name in SCENES)):
        raise ValueError('Replacement requires a recognized complete three-scene suite')
    configs = {}
    for name in SCENES:
        directory = root / name
        run = json.loads((directory / 'scene_run.json').read_text(encoding='utf-8'))
        if run.get('format') != 'wtumi-scene-run-v1':
            raise ValueError(f'Unsupported saved scene report: {name}')
        for key in ('scene_config', 'model', 'capture_manifest', 'capture_report', 'history', 'final_result'):
            artifact = _relative_artifact(directory, run['paths'][key])
            if not artifact.exists():
                raise FileNotFoundError(artifact)
        _, config, frames = comparison_inputs(directory)
        if len(frames) != run['integration_frames']:
            raise ValueError(f'Incomplete saved capture/map history: {name}')
        for key in ('gallery', 'inputs_animation', 'reconstruction_animation',
                    'camera_voxel_gif', 'camera_voxel_png', 'camera_voxel_metadata', 'camera_voxel_mp4'):
            if key not in run['paths']:
                continue
            artifact = _relative_artifact(directory, run['paths'][key])
            if name not in selected and not artifact.is_file():
                raise FileNotFoundError(artifact)
        configs[name] = config
        # Existing materialized inputs are retained only for untouched scenes.
        old_input = root / 'inputs' / name
        if old_input.exists():
            load_scene_config(old_input)
    return root, report, configs


def _saved_preset_plan(configs, preset, old_plan, selected, source_counts):
    """Summarize actual saved configurations without resampling them again."""
    from .scene_presets import describe_preset

    plan = describe_preset(configs, 'default')
    old_rows = {row['scene']: row for row in old_plan.get('scenes', [])}
    for row in plan['scenes']:
        name = row['scene']
        previous = old_rows.get(name, {})
        row['source_waypoints'] = source_counts.get(name, previous.get('source_waypoints', row['integration_frames']))
        row['scene_preset'] = preset if name in selected else previous.get('scene_preset', old_plan.get('preset', 'saved'))
        row['saved_scene_config'] = f'{name}/scene/scene.yaml'
        row['robot_model'] = configs[name].get('robot', {}).get('model')
        row['robot_pose'] = configs[name].get('robot', {}).get('pose')
        row['box_enabled'] = configs[name]['box']['enabled']
    labels = {row['scene_preset'] for row in plan['scenes']}
    plan['preset'] = next(iter(labels)) if len(labels) == 1 else 'mixed'
    plan['requires_mp4_encoder'] = any(row['scene_preset'] == 'pace30' for row in plan['scenes'])
    plan['sampling'] = 'Settings below describe each completed scene configuration. Selected scenes were freshly recaptured and reconstructed; untouched scenes retain their saved inputs and maps.'
    plan['last_replaced_scenes'] = list(selected)
    return plan


def scene_replacement_plan(output_dir, replace_scenes, preset='default'):
    """Validate replacement without creating files or starting any rendering."""
    from .scene_bundle import load_scene_config
    from .scene_presets import preset_config, validate_preset

    selected = _replacement_names(replace_scenes)
    validate_preset(preset)
    root, _, configs = _existing_suite(output_dir, selected)
    old_plan = json.loads((root / 'preset.json').read_text()) if (root / 'preset.json').exists() else {}
    source_counts = {}
    for name in selected:
        source = load_scene_config(name)
        source_counts[name] = len(source['trajectory'])
        configs[name] = preset_config(source, preset)
    plan = _saved_preset_plan(configs, preset, old_plan, selected, source_counts)
    plan['status'] = 'planned'
    plan['replace_scenes'] = plan.pop('last_replaced_scenes')
    plan['sampling'] = 'Selected scenes will be freshly recaptured and reconstructed from current templates; untouched scenes will retain their saved inputs and maps.'
    return plan


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.replacement-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _replace_scene_suite(output_dir, replace_scenes, preset, progress):
    from .scene_bundle import create_scene, load_scene_config, _dump_yaml
    from .scene_pipeline import run_scene
    from .scene_presets import preset_config, validate_preset

    selected = _replacement_names(replace_scenes)
    validate_preset(preset)
    root, _, old_configs = _existing_suite(output_dir, selected)
    sources = {name: load_scene_config(name) for name in selected}
    for config in sources.values():
        preset_config(config, preset)
    if preset == 'pace30':
        from .video import require_video_encoder
        require_video_encoder()
    old_plan = json.loads((root / 'preset.json').read_text()) if (root / 'preset.json').exists() else {}
    transaction = root / _TRANSACTION
    transaction.mkdir()
    staged, backup = transaction / 'new', transaction / 'backup'
    staged.mkdir()
    backup.mkdir()
    journal = {'format': 'wtumi-scene-replacement-v1', 'status': 'staging',
               'selected_scenes': list(selected), 'preset': preset, 'operations': []}
    journal_path = transaction / 'journal.json'
    _atomic_json(journal_path, journal)
    moved = []
    installed_reports = []
    input_parent_created = False

    def move(source, destination):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        operation = {'from': source.relative_to(root).as_posix(),
                     'to': destination.relative_to(root).as_posix(), 'state': 'pending'}
        journal['operations'].append(operation)
        _atomic_json(journal_path, journal)
        moved.append((source, destination))
        source.rename(destination)
        operation['state'] = 'done'
        _atomic_json(journal_path, journal)

    try:
        directories = {name: root / name for name in SCENES}
        configs = dict(old_configs)
        for index, name in enumerate(selected, start=1):
            if progress:
                progress(f'Rebuilding selected scene {index}/{len(selected)}: {name}; previous result remains intact')
            bundle = staged / 'inputs' / name
            create_scene(name, bundle)
            config = preset_config(load_scene_config(bundle), preset)
            (bundle / 'scene.yaml').write_text(_dump_yaml(config), encoding='utf-8')
            load_scene_config(bundle)
            run = run_scene(bundle, staged / name, progress=progress)
            if preset == 'pace30' and not run.get('paths', {}).get('camera_voxel_mp4'):
                raise RuntimeError(f'pace30 requires a completed MP4 comparison for scene {name}')
            directories[name] = staged / name
            configs[name] = load_scene_config(staged / name / 'scene')

        reports = staged / 'reports'
        reports.mkdir()
        report = write_scene_suite_report(reports, _scene_directories=directories)
        plan = _saved_preset_plan(configs, preset, old_plan, selected,
                                  {name: len(config['trajectory']) for name, config in sources.items()})
        _atomic_json(reports / 'preset.json', plan)
        report['preset'] = plan['preset']
        report['paths']['preset'] = 'preset.json'
        report['last_replaced_scenes'] = list(selected)
        _atomic_json(reports / 'suite.json', report)

        journal['status'] = 'committing'
        _atomic_json(journal_path, journal)
        for filename in _REPORT_FILES:
            if (root / filename).exists():
                move(root / filename, backup / filename)
        installed_reports.append(root / 'suite.json')
        _atomic_json(root / 'suite.json', {
            'format': 'wtumi-scene-suite-replacement-in-progress',
            'selected_scenes': list(selected), 'journal': f'{_TRANSACTION}/journal.json'})
        installed_reports.append(root / 'index.html')
        (root / 'index.html').write_text('<!doctype html><html lang="en"><title>Scene replacement in progress</title><h1>Scene replacement in progress</h1><p>The previous scene data and reports are retained in .scene-replacement/backup. Do not treat this directory as a completed suite until replacement finishes.</p></html>', encoding='utf-8')
        if not (root / 'inputs').exists():
            (root / 'inputs').mkdir()
            input_parent_created = True
        for name in selected:
            move(root / name, backup / name)
            move(staged / name, root / name)
            if (root / 'inputs' / name).exists():
                move(root / 'inputs' / name, backup / 'inputs' / name)
            move(staged / 'inputs' / name, root / 'inputs' / name)
        # Publish the complete suite marker last, after scene and report swaps.
        for filename in _REPORT_FILES:
            target = root / filename
            if target not in installed_reports:
                installed_reports.append(target)
            os.replace(reports / filename, target)
        journal['status'] = 'committed'
        _atomic_json(journal_path, journal)
    except BaseException:
        try:
            for path in installed_reports:
                if path.is_file():
                    path.unlink()
            for source, destination in reversed(moved):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
            if input_parent_created and not any((root / 'inputs').iterdir()):
                (root / 'inputs').rmdir()
            shutil.rmtree(transaction)
        except BaseException as rollback_error:
            journal['status'] = 'rollback_required'
            _atomic_json(journal_path, journal)
            raise RuntimeError(f'Replacement rollback needs inspection at {transaction}; backups were retained') from rollback_error
        raise
    # Only successful replacement retires the selected generated scene data.
    shutil.rmtree(transaction)
    return report


def write_scene_suite_report(output_dir, *, _scene_directories=None):
    """Create a portable landing page from three completed scene runs."""
    root = Path(output_dir).expanduser().resolve()
    rows, cards = [], []
    for name in SCENES:
        directory = (_scene_directories or {}).get(name, root / name)
        run = json.loads((directory / 'scene_run.json').read_text(encoding='utf-8'))
        if run.get('format') != 'wtumi-scene-run-v1':
            raise ValueError(f'Unsupported scene report: {name}')
        paths = run['paths']
        for key in ('gallery', 'camera_voxel_gif', 'camera_voxel_png', 'camera_voxel_metadata'):
            value = Path(paths[key])
            if value.is_absolute() or not (directory / value).resolve().is_relative_to(directory):
                raise ValueError(f'Suite artifact path escapes its scene directory: {name}/{key}')
            if not (directory / value).is_file():
                raise FileNotFoundError(directory / value)
        metrics = run['held_out']
        row = {'scene': name, 'integration_frames': run['integration_frames'],
               'held_out_views': run['held_out_frames'],
               'voxel_size_mm': run['effective_voxel_size_m'] * 1000,
               'depth_coverage_percent': metrics['coverage'] * 100,
               'mean_absolute_depth_error_mm': metrics['mae_m'] * 1000,
               'root_mean_square_depth_error_mm': metrics['rmse_m'] * 1000}
        # Describe the saved scene, never today's potentially changed template.
        config_path = paths.get('scene_config', 'scene/scene.yaml')
        saved_config = _relative_artifact(directory, config_path)
        configuration = None
        if saved_config.is_file():
            from .scene_bundle import load_scene_config
            configuration = load_scene_config(saved_config)
        row.update({'robot_model': configuration.get('robot', {}).get('model') if configuration else None,
                    'robot_pose': configuration.get('robot', {}).get('pose') if configuration else None,
                    'box_enabled': configuration['box']['enabled'] if configuration else None,
                    'playback_fps': configuration['playback']['fps'] if configuration else None,
                    'camera_width': configuration['camera']['width'] if configuration else None,
                    'camera_height': configuration['camera']['height'] if configuration else None})
        rows.append(row)
        picture = html.escape(f"{name}/{paths['camera_voxel_png']}", quote=True)
        video = paths.get('camera_voxel_mp4')
        if video:
            value = Path(video)
            if value.is_absolute() or not (directory / value).resolve().is_relative_to(directory):
                raise ValueError('Suite video path escapes scene directory')
            if not (directory / value).is_file():
                raise FileNotFoundError(directory / value)
            media = f'<video controls loop playsinline preload="none" poster="{picture}" src="{html.escape(name + "/" + video, quote=True)}"></video>'
        else:
            media = f'<img src="{html.escape(name + "/" + paths["camera_voxel_gif"], quote=True)}" alt="Camera and cumulative voxel map">'
        cards.append(f'<section><h2>{html.escape(name)}</h2>{media}'
                     f'<p><a href="{html.escape(name + "/" + paths["gallery"], quote=True)}">Scene results, saved reconstruction and evaluation</a></p>'
                     f'<p>Voxel edge: {row["voxel_size_mm"]:.3f} mm · Held-out depth coverage: {row["depth_coverage_percent"]:.2f}%</p>'
                     f'<p>Saved carried-box setting: {"enabled" if row["box_enabled"] else "disabled" if row["box_enabled"] is False else "unspecified"}</p></section>')
    fields = list(rows[0])
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {'format': 'wtumi-g1-scene-suite-v1', 'scenes': rows,
              'paths': {'gallery': 'index.html', 'summary': 'summary.csv',
                        **{name: f'{name}/scene_run.json' for name in SCENES}},
              'scope': 'Independent static-scene experiments with exact simulator poses. Left panels show recorded camera RGB; right panels show only matching cumulative voxel surface evidence from an overview. Different viewpoints, voxel sizes and occlusion conditions are not directly comparable method rankings.'}
    (root / 'suite.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 vision-to-voxel scene experiments</title>
<style>body{font:16px system-ui;max-width:1380px;margin:28px auto;padding:0 20px;background:#eef1f6;color:#182337}video,img{width:100%;border-radius:8px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}a{color:#1650ac}</style>
<h1>G1 vision-to-voxel scene experiments</h1>
<p>Left: recorded head-camera RGB. Right: the cumulative voxel reconstruction after the same observation, rendered in MuJoCo from a fixed overview. The viewpoints differ. Display sampling and camera markers are labeled; original environment geometry is absent from the voxel panel.</p>
<p>Room and doorway interiors are 6.4 x 10 m with a real-size G1. Each scene card reports its saved carried-box setting. Robots follow prescribed poses; these are mapping experiments, not locomotion or door-manipulation controllers.</p>
<p><a href="summary.csv">Evaluation summary</a> · <a href="suite.json">Suite report</a></p>
<p>Coverage refers to held-out image pixels, not whole-scene completeness. Depth errors use overlapping valid pixels. A small overlap error does not compensate for unobserved surfaces.</p>
''' + '\n'.join(cards) + '</html>'
    (root / 'index.html').write_text(page, encoding='utf-8')
    (root / 'README.md').write_text(
        '# G1 vision-to-voxel scene experiments\n\nOpen index.html for all three synchronized camera/voxel videos. '
        'Each scene directory contains its own model, captures, prefix maps, evaluation and native viewer inputs. '
        'Copy this entire suite directory when moving results.\n\n'
        'From the project root, open a native result with:\n\n'
        '```bash\npython scripts/scene.py view --result PATH_TO_SUITE/room\n'
        'python scripts/scene.py view --result PATH_TO_SUITE/doorway\n'
        'python scripts/scene.py view --result PATH_TO_SUITE/g1_stairs\n```\n', encoding='utf-8')
    return report


def suite_preset_plan(preset='default'):
    """Validate and describe a preset without files, rendering or encoding."""
    from .scene_bundle import load_scene_config
    from .scene_presets import describe_preset, validate_preset

    validate_preset(preset)
    return describe_preset({name: load_scene_config(name) for name in SCENES}, preset)


def run_scene_suite(output_dir, progress=None, *, preset='default', replace_scenes=None):
    from .scene_bundle import create_scene, load_scene_config, _dump_yaml
    from .scene_pipeline import run_scene
    from .scene_presets import describe_preset, preset_config, validate_preset

    if replace_scenes is not None:
        return _replace_scene_suite(output_dir, replace_scenes, preset, progress)
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f'Scene suite output must be new or empty: {root}')
    validate_preset(preset)
    configs = {name: load_scene_config(name) for name in SCENES}
    inputs = {name: name for name in SCENES}
    if preset != 'default':
        from .video import require_video_encoder

        plan = describe_preset(configs, preset)
        # A 30 fps run requires MP4; GIF is only a bounded preview format.
        # Check before creating output directories or starting any capture.
        require_video_encoder()
        for name in SCENES:
            bundle = root / 'inputs' / name
            create_scene(name, bundle)
            portable = load_scene_config(bundle)
            config = preset_config(portable, preset)
            (bundle / 'scene.yaml').write_text(_dump_yaml(config), encoding='utf-8')
            load_scene_config(bundle)
            inputs[name] = bundle
        (root / 'preset.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    for index, name in enumerate(SCENES, start=1):
        if progress:
            progress(f'Scene {index}/{len(SCENES)}: {name}')
        report = run_scene(inputs[name], root / name, progress=progress)
        if preset == 'pace30' and not report.get('paths', {}).get('camera_voxel_mp4'):
            raise RuntimeError(f'pace30 requires a completed MP4 comparison for scene {name}')
    report = write_scene_suite_report(root)
    if preset != 'default':
        report['preset'] = preset
        report['paths']['preset'] = 'preset.json'
        (root / 'suite.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
