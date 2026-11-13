"""One capture, a voxel-size sweep, held-out comparisons and an offline viewer."""

from __future__ import annotations

import copy
import json
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .config import normalize_occupancy_config
from .tsdf import TSDFConfig


def load_demo_config(path):
    """Validate all portable run settings before creating an output directory."""
    from .mujoco_capture import load_capture_config

    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    allowed = {'schema_version', 'capture_config', 'reconstruction', 'evaluation', 'resolutions', 'occupancy'}
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('simulation demo requires schema_version: 1')
    if set(raw) - allowed:
        raise ValueError(f'unsupported demo fields: {set(raw)-allowed}')
    capture = raw.get('capture_config')
    if not isinstance(capture, str) or not capture:
        raise ValueError('demo.capture_config must be a capture YAML path')
    capture_path = (path.parent / Path(capture).expanduser()).resolve()
    load_capture_config(capture_path)  # Does not import MuJoCo or create a GL context.
    rec = raw.get('reconstruction')
    rec_keys = {'backend', 'volume_length_m', 'resolution', 'origin_m', 'sdf_trunc_m'}
    if not isinstance(rec, dict) or set(rec)-rec_keys or rec.get('backend', 'uniform') != 'uniform':
        raise ValueError('demo reconstruction requires Uniform volume settings')
    if type(rec.get('resolution')) is not int or rec['resolution'] < 2:
        raise ValueError('demo reconstruction.resolution must be an integer >= 2')
    for key in ('volume_length_m', 'sdf_trunc_m'):
        value = rec.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'demo reconstruction.{key} must be finite and positive')
    settings = TSDFConfig(**rec)
    rec = {'backend': 'uniform', 'volume_length_m': float(settings.volume_length_m),
           'resolution': settings.resolution, 'origin_m': np.asarray(settings.origin_m, dtype=float).tolist(),
           'sdf_trunc_m': float(settings.sdf_trunc_m)}
    evaluation = raw.get('evaluation')
    if not isinstance(evaluation, dict) or set(evaluation) != {'depth_tolerance_m'}:
        raise ValueError('demo evaluation requires depth_tolerance_m')
    tolerance = evaluation['depth_tolerance_m']
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('demo evaluation.depth_tolerance_m must be finite and positive')
    resolutions = raw.get('resolutions')
    if not isinstance(resolutions, list) or not resolutions or any(type(n) is not int or n < 2 for n in resolutions):
        raise ValueError('demo resolutions must be a nonempty list of integers >= 2')
    if len(set(resolutions)) != len(resolutions):
        raise ValueError('demo resolutions must be unique')
    if rec['resolution'] not in resolutions:
        raise ValueError('base reconstruction resolution must be included in demo resolutions')
    occupancy = normalize_occupancy_config(copy.deepcopy(raw.get('occupancy')), 'uniform')
    return {'schema_version': 1, 'capture_config': str(capture_path), 'reconstruction': rec,
            'evaluation': {'depth_tolerance_m': float(tolerance)}, 'resolutions': resolutions.copy(),
            'occupancy': occupancy, '_path': str(path)}


def run_sim_demo(config_path, output_dir, progress=None):
    """Capture once and reconstruct the same observations at several voxel sizes.

    Capture is an external simulator boundary. All later steps operate on the
    portable saved sequence, so capture data can be copied and replayed on a
    different host without importing MuJoCo again.
    """
    from .mujoco_capture import capture_mujoco
    from .sequence import RGBDSequence
    from .experiments import run_experiment
    from .experiment_plots import save_experiment_plot
    from .viewer import export_viewer

    started = perf_counter()
    cfg = load_demo_config(config_path)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'demo output must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'demo_source.yaml').write_text(Path(cfg['_path']).read_text(encoding='utf-8'), encoding='utf-8')
    (out/'demo_resolved.yaml').write_text(yaml.safe_dump({k:v for k,v in cfg.items() if k!='_path'}, sort_keys=False), encoding='utf-8')
    if progress:
        progress('Capturing the static room once into a portable RGB-D sequence')
    manifest = Path(capture_mujoco(cfg['capture_config'], out/'capture')).resolve()
    # The capture contract writes inside the requested directory; do not emit
    # a misleading relative path if a different exporter violates that contract.
    if manifest != out/'capture/manifest.json':
        raise ValueError('capture must return <output>/capture/manifest.json')
    sequence = RGBDSequence(manifest)
    reconstruction = {
        'schema_version': 3,
        'source': {'type': 'rgbd_sequence', 'manifest': 'capture/manifest.json'},
        'reconstruction': cfg['reconstruction'], 'evaluation': cfg['evaluation'],
        'occupancy': cfg['occupancy'],
    }
    (out/'reconstruction.yaml').write_text(yaml.safe_dump(reconstruction, sort_keys=False), encoding='utf-8')
    experiment = {
        'schema_version': 1, 'base_config': 'reconstruction.yaml',
        'resolutions': cfg['resolutions'], 'seeds': [7],
        'profiles': [{'name': 'clean', 'depth_noise_std_m': 0.0, 'dropout_probability': 0.0}],
    }
    (out/'experiment.yaml').write_text(yaml.safe_dump(experiment, sort_keys=False), encoding='utf-8')
    if progress:
        progress('Reconstructing the same captured frames at each voxel size')
    result = run_experiment(out/'experiment.yaml', out/'experiment', progress=progress)
    selected_id = f"clean_n{cfg['reconstruction']['resolution']}_s7"
    selected = next(row for row in result['runs'] if row['case_id'] == selected_id)
    run_dir = out/'experiment'/selected['run_dir']
    if progress:
        progress('Exporting the base-resolution 3D viewer and voxel-size error plot')
    viewer_path, _ = export_viewer(run_dir)
    png_path, svg_path = save_experiment_plot(out/'experiment/summary.json')
    versions = json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))['versions']
    relative = lambda p: Path(p).resolve().relative_to(out).as_posix()
    report = {
        'format': 'wtumi-sim-demo-v1', 'scene': sequence.scene_name,
        'capture_count': 1, 'reference_kind': sequence.reference_kind,
        'integration_frames': len([v for v in sequence.views if v['split']=='integration']),
        'held_out_frames': len([v for v in sequence.views if v['split']=='held_out']),
        'resolutions': cfg['resolutions'], 'base_resolution': cfg['reconstruction']['resolution'],
        'paired_inputs_verified': result['paired_inputs_verified'],
        'capture_metadata': sequence.metadata, 'versions': versions,
        'paths': {
            'capture_manifest': relative(manifest), 'reconstruction_config': 'reconstruction.yaml',
            'experiment_config': 'experiment.yaml', 'summary_json': 'experiment/summary.json',
            'summary_csv': 'experiment/summary.csv', 'report': 'experiment/report.md',
            'overview': 'experiment/overview.png', 'base_run': relative(run_dir),
            'viewer': relative(viewer_path), 'error_plot_png': relative(png_path),
            'error_plot_svg': relative(svg_path),
            'comparisons': [relative(run_dir/'renders'/view['id']/'comparison.png')
                            for view in sequence.views if view['split']=='held_out'],
        },
        'selected_held_out_metrics': {key: selected[key] for key in (
            'voxel_size_m', 'coverage', 'mae_m', 'rmse_m', 'silhouette_iou',
            'accurate_reference_fraction', 'missing_pixels', 'extra_pixels')},
        'total_seconds': perf_counter()-started,
        'scope': 'Static camera waypoints, one RGB-D capture reused at all resolutions. No robot motion or tactile inputs; RGB-D maps do not imply runtime policy access.',
    }
    (out/'demo.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    first_comparison = report['paths']['comparisons'][0]
    lines = [
        '# Simulated RGB-D reconstruction demo', '',
        'Capture one sequence, then reconstruct it at different voxel sizes. Held-out observations are used only for evaluation.', '',
        f"- [Reference and reconstruction comparison]({first_comparison})",
        f"- [Error versus voxel size]({report['paths']['error_plot_png']})",
        f"- [Offline 3D viewer]({report['paths']['viewer']})",
        '- [All experiment results](experiment/report.md)',
        '- [Run summary](demo.json)', '',
        '`capture/` stores RGB, metric camera-Z depth, calibration, poses and observation splits.',
        '`reconstruction.yaml` and `experiment.yaml` use relative paths and can be replayed after moving the complete output directory.',
        'Resolved configs and source_manifest copies within individual runs record provenance and may contain absolute paths from the original machine.', '',
        'This captures a static scene at camera waypoints; it does not include robot motion, tactile input or a planner. The project experiment design determines whether reference maps are available at deployment.', '',
    ]
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return report
