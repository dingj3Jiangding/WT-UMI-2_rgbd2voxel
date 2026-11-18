"""Paired head-camera occlusion experiments with a static environment map.

The simulator supplies known poses and masks. Environment-only held-out views
and staircase geometry are evaluation references, never fusion observations.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .config import normalize_occupancy_config
from .sequence import RGBDSequence
from .tsdf import TSDFConfig


CASE_LABELS = {
    'unoccluded_multi': 'No box / multiple frames',
    'occluded_single': 'Box / first frame',
    'occluded_multi': 'Box / multiple frames',
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_occlusion_demo_config(path):
    """Validate the full benchmark before any capture or output creation."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    return _normalize_occlusion_demo_config(raw, path)


def _normalize_occlusion_demo_config(raw, path):
    """Shared validation for benchmarks that use the paired rig capture."""
    from .rig_capture import load_rig_capture_config

    path = Path(path).expanduser().resolve()
    allowed = {'schema_version', 'capture_config', 'reconstruction', 'occupancy', 'evaluation'}
    if not isinstance(raw, dict) or raw.get('schema_version') != 1 or set(raw)-allowed:
        raise ValueError('occlusion demo requires schema_version 1 and supported fields')
    source = raw.get('capture_config')
    if not isinstance(source, str) or not source:
        raise ValueError('capture_config must be a YAML path')
    capture_path = (path.parent/Path(source).expanduser()).resolve()
    capture = load_rig_capture_config(capture_path)
    rec = raw.get('reconstruction')
    keys = {'backend', 'volume_length_m', 'resolution', 'origin_m', 'sdf_trunc_m'}
    if not isinstance(rec, dict) or set(rec)-keys or rec.get('backend', 'uniform') != 'uniform':
        raise ValueError('occlusion demo requires uniform reconstruction')
    if type(rec.get('resolution')) is not int or not 2 <= rec['resolution'] <= 256:
        raise ValueError('reconstruction.resolution must be an integer in [2, 256]')
    for key in ('volume_length_m', 'sdf_trunc_m'):
        value = rec.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'reconstruction.{key} must be finite and positive')
    settings = TSDFConfig(**rec)
    rec = {'backend': 'uniform', 'volume_length_m': float(settings.volume_length_m),
           'origin_m': np.asarray(settings.origin_m, dtype=float).tolist(),
           'resolution': settings.resolution, 'sdf_trunc_m': float(settings.sdf_trunc_m)}
    occupancy = normalize_occupancy_config(copy.deepcopy(raw.get('occupancy')), 'uniform')
    if not occupancy['enabled'] or occupancy['resolution'] > 160:
        raise ValueError('occlusion demo needs occupancy enabled with resolution <= 160')
    evaluation = raw.get('evaluation')
    if not isinstance(evaluation, dict) or set(evaluation) != {'depth_tolerance_m', 'sample_spacing_m'}:
        raise ValueError('evaluation requires depth_tolerance_m and sample_spacing_m')
    for key, value in evaluation.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'evaluation.{key} must be finite and positive')
    lo, hi = np.array(capture['roi']['min_m']), np.array(capture['roi']['max_m'])
    origin = np.array(rec['origin_m'])
    if np.any(lo < origin) or np.any(hi > origin+rec['volume_length_m']):
        raise ValueError('the entire evaluation ROI must be inside the voxel volume')
    voxel_size = rec['volume_length_m']/occupancy['resolution']
    for axis in range(3):
        centers = origin[axis]+(np.arange(occupancy['resolution'])+.5)*voxel_size
        if not np.any((centers >= lo[axis]) & (centers < hi[axis])):
            raise ValueError('evaluation ROI must contain occupancy cell centers on every axis')
    from .stair_metrics import sample_stair_surfaces
    samples = sample_stair_surfaces(capture['stairs'], evaluation['sample_spacing_m'])
    if np.any(samples < lo) or np.any(samples > hi):
        raise ValueError('ROI must contain the sampled staircase treads and risers')
    if len(capture['trajectory']) < 2:
        raise ValueError('occlusion comparison requires at least two integration poses')
    return {'schema_version': 1, 'capture_config': str(capture_path), 'capture': capture,
            'reconstruction': rec, 'occupancy': occupancy,
            'evaluation': {k: float(v) for k, v in evaluation.items()}, '_path': str(path)}


def _array_digest(sequence, split):
    digest = hashlib.sha256()
    for view in sequence.views:
        if view['split'] != split:
            continue
        rgb, depth = sequence.read_reference(view) if split == 'held_out' else sequence.read(view)
        digest.update(view['id'].encode('utf-8'))
        digest.update(np.asarray(view['T_wc'], dtype='<f8').tobytes())
        digest.update(rgb.tobytes())
        digest.update(depth.astype('<f4').tobytes())
    return digest.hexdigest()


def verify_paired_capture(unoccluded_manifest, occluded_manifest):
    """Check paired geometry, common held-outs, and masks before reconstruction."""
    clear, blocked = RGBDSequence(unoccluded_manifest), RGBDSequence(occluded_manifest)
    if clear.reference_kind != 'simulator_ground_truth' or blocked.reference_kind != 'simulator_ground_truth':
        raise ValueError('paired simulated captures must declare simulator_ground_truth references')
    if clear.camera != blocked.camera or len(clear.views) != len(blocked.views):
        raise ValueError('paired capture camera or frame count differs')
    for a, b in zip(clear.views, blocked.views):
        if any(a[key] != b[key] for key in ('id', 'split', 'timestamp_s')) or not np.allclose(a['T_wc'], b['T_wc'], atol=1e-10, rtol=0):
            raise ValueError('paired capture poses, IDs, splits or timestamps differ')
    held_hash = _array_digest(clear, 'held_out')
    if held_hash != _array_digest(blocked, 'held_out'):
        raise ValueError('paired capture does not share identical held-out references')
    for sequence in (clear, blocked):
        for view in sequence.views:
            if view['split'] != 'integration':
                continue
            rgb, depth = sequence.read(view)
            frame = sequence.root/'frames'/view['id']
            with Image.open(frame/'excluded_mask.png') as image:
                mask = np.asarray(image) > 0
            raw_depth = np.load(frame/'raw_depth_m.npy', allow_pickle=False)
            with Image.open(frame/'raw_rgb.png') as image:
                raw_rgb = np.asarray(image)
            if mask.shape != depth.shape or raw_depth.shape != depth.shape or raw_rgb.shape != rgb.shape:
                raise ValueError('raw observations and exclusion mask must align with RGB-D')
            if np.any(depth[mask] != 0) or np.any(rgb[mask] != 0):
                raise ValueError('excluded self/box pixels reached fusion observations')
            if not np.array_equal(depth[~mask], raw_depth[~mask]) or not np.array_equal(rgb[~mask], raw_rgb[~mask]):
                raise ValueError('unmasked capture pixels changed during exclusion')
    return {'held_out_rgbd_sha256': held_hash,
            'unoccluded_integration_sha256': _array_digest(clear, 'integration'),
            'occluded_integration_sha256': _array_digest(blocked, 'integration'),
            'integration_frame_ids': [v['id'] for v in clear.views if v['split']=='integration'],
            'held_out_frame_ids': [v['id'] for v in clear.views if v['split']=='held_out']}


def write_prefix_manifest(source, count=1):
    """Select an observation prefix while keeping frame paths portable and shared."""
    source = Path(source).resolve()
    sequence = RGBDSequence(source)
    integration = [v for v in sequence.views if v['split']=='integration']
    if type(count) is not int or not 1 <= count <= len(integration):
        raise ValueError('prefix length must be within the integration sequence')
    selected = {v['id'] for v in integration[:count]}
    manifest = copy.deepcopy(sequence.manifest)
    manifest['frames'] = [v for v in manifest['frames'] if v['split']=='held_out' or v['id'] in selected]
    manifest['metadata']['observation_prefix'] = {'source_manifest': source.name, 'integration_frames': count}
    destination = source.with_name(f'manifest_prefix_{count:03d}.json')
    if destination.exists():
        raise FileExistsError(destination)
    _write_json(destination, manifest)
    RGBDSequence(destination)
    return destination


def _condition_comparison(out, case_rows, view_id):
    panels = []
    for row in case_rows:
        with Image.open(out/row['run_dir']/'renders'/view_id/'comparison.png') as source:
            source = source.convert('RGB')
            source.thumbnail((960, 620), Image.Resampling.LANCZOS)
            panel = Image.new('RGB', (source.width, source.height+30), '#151b24')
            ImageDraw.Draw(panel).text((10, 8), CASE_LABELS[row['case_id']], fill='white')
            panel.paste(source, (0, 30))
            panels.append(panel)
    canvas = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#151b24')
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y)); y += panel.height
    canvas.save(out/'condition_comparison.png')


def _save_progress(out, progression):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    colors = {'unoccluded_multi':'#2786b5', 'occluded_multi':'#d76f36'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for condition, (rows, _) in progression.items():
        x = [row['frames'] for row in rows]
        axes[0].plot(x, [100*row['roi_unknown_fraction'] for row in rows], marker='o',
                     label=CASE_LABELS[condition], color=colors[condition])
        axes[1].plot(x, [100*row['surface_samples_known_fraction'] for row in rows], marker='o',
                     label=CASE_LABELS[condition], color=colors[condition])
    for ax in axes:
        ax.set_xlabel('Integrated frames'); ax.set_ylim(0, 101); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Unknown voxel centers in fixed ROI (%)')
    axes[1].set_ylabel('Stair surface samples in observed cells (%)')
    fig.suptitle('Accumulated visual evidence near the stairs')
    fig.text(.5, .02, 'Observed cells may be free or occupied; this is not a surface-accuracy or collision-safety score.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .96))
    fig.savefig(out/'observation_progress.png', dpi=160); plt.close(fig)
    palette = np.array([[100,116,139],[45,212,191],[251,146,60]], dtype=np.uint8)
    conditions = list(progression)
    frame_count = len(progression[conditions[0]][0])
    frames = []
    for index in range(frame_count):
        canvas = Image.new('RGB', (800, 465), '#111827')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'Center-X slice: Y horizontal, Z vertical | Gray unknown / Teal free / Orange surface hit', fill='white')
        for j, condition in enumerate(conditions):
            rows, snapshots = progression[condition]
            # PIL starts at the top; data's Z axis starts at the bottom.
            image = Image.fromarray(palette[snapshots[index]][::-1]).resize((380,380), Image.Resampling.NEAREST)
            canvas.paste(image, (10+400*j, 55))
            draw.text((10+400*j, 35), CASE_LABELS[condition], fill='white')
            row = rows[index]
            draw.text((10+400*j, 442), f"Frame {row['frames']} | ROI unknown: {100*row['roi_unknown_fraction']:.1f}%", fill='white')
        frames.append(canvas)
    frames[0].save(out/'occupancy_progress.gif', save_all=True, append_images=frames[1:], duration=650, loop=0)


def _summary_row(case_id, run_dir, report, metrics):
    depth = metrics['held_out_depth']['pooled']
    surface = metrics['surface_completeness']
    occupancy = metrics['occupancy_roi']
    return {'case_id': case_id, 'integration_frames': len(report['integrated_frame_ids']),
            'voxel_size_m': report['effective_voxel_size_m'],
            'occupancy_voxel_size_m': report['occupancy']['voxel_size_m'],
            'stair_depth_coverage': depth['coverage'],
            'stair_depth_mean_absolute_error_m': depth['mean_absolute_error_m'],
            'stair_depth_root_mean_square_error_m': depth['root_mean_square_error_m'],
            'stair_depth_within_tolerance_fraction': depth['within_tolerance_fraction'],
            'surface_coverage_within_tolerance_fraction': surface['coverage_within_tolerance_fraction'],
            'surface_mean_nearest_mesh_distance_m': surface['mean_nearest_mesh_distance_m'],
            'roi_unknown_fraction': occupancy['roi_unknown_fraction'],
            'surface_samples_in_free_cells_fraction': occupancy['surface_samples_in_free_cells_fraction'],
            'total_seconds': report['total_seconds'], 'run_dir': run_dir}


def run_occlusion_demo(config_path, output_dir, progress=None):
    """Capture paired sequences once, compare three conditions, export diagnostics."""
    from .rig_capture import capture_rig
    from .pipeline import run_reconstruction
    from .stair_metrics import evaluate_stair_result, occupancy_progress
    from .viewer import export_viewer

    started = perf_counter()
    cfg = load_occlusion_demo_config(config_path)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'occlusion demo output must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'demo_source.yaml').write_text(Path(cfg['_path']).read_text(), encoding='utf-8')
    if progress: progress('Capturing paired head-camera sequences with and without the carried box')
    capture = capture_rig(cfg['capture_config'], out/'capture', progress=progress)
    clear_manifest = out/'capture'/capture['paths']['unoccluded_manifest']
    blocked_manifest = out/'capture'/capture['paths']['occluded_manifest']
    verification = verify_paired_capture(clear_manifest, blocked_manifest)
    single_manifest = write_prefix_manifest(blocked_manifest, 1)
    case_sources = {'unoccluded_multi': clear_manifest, 'occluded_single': single_manifest,
                    'occluded_multi': blocked_manifest}
    masks = {key: out/'capture'/value for key, value in capture['paths']['held_out_masks'].items()}
    expected_held = verification['held_out_frame_ids']
    if set(masks) != set(expected_held):
        raise ValueError('stair-only evaluation masks must cover all held-out views')
    config_dir = out/'configs'; config_dir.mkdir()
    case_rows, case_details = [], {}
    for case_id, manifest in case_sources.items():
        if progress: progress(f'Reconstructing {CASE_LABELS[case_id]}')
        run_config = {'schema_version': 3,
                      'source': {'type':'rgbd_sequence', 'manifest':os.path.relpath(manifest, config_dir)},
                      'reconstruction':cfg['reconstruction'], 'occupancy':cfg['occupancy'],
                      'evaluation': {'depth_tolerance_m':cfg['evaluation']['depth_tolerance_m']}}
        config = config_dir/f'{case_id}.yaml'
        config.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding='utf-8')
        run_relative = f'runs/{case_id}'
        report = run_reconstruction(config, out/run_relative)
        expected_inputs = verification['integration_frame_ids'][:1] if case_id=='occluded_single' else verification['integration_frame_ids']
        if report['integrated_frame_ids'] != expected_inputs or report['held_out_frame_ids'] != expected_held:
            raise RuntimeError('reconstruction did not respect the selected fusion/evaluation split')
        metrics = evaluate_stair_result(out/run_relative, capture['stairs'], capture['roi'], masks,
                                       tolerance_m=cfg['evaluation']['depth_tolerance_m'],
                                       sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
        _write_json(out/run_relative/'stair_metrics.json', metrics)
        case_details[case_id] = metrics
        case_rows.append(_summary_row(case_id, run_relative, report, metrics))
    if progress: progress('Evaluating observation history, surface completeness and unknown regions')
    occupancy_settings = {**{key:cfg['reconstruction'][key] for key in ('origin_m','volume_length_m')},
                          **{key:cfg['occupancy'][key] for key in ('resolution','pixel_stride')}}
    progression = {}
    for case_id, manifest in (('unoccluded_multi',clear_manifest), ('occluded_multi',blocked_manifest)):
        progression[case_id] = occupancy_progress(manifest, occupancy_settings, capture['roi'], capture['stairs'],
                                                  sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
        last = progression[case_id][0][-1]
        measured = case_details[case_id]['occupancy_roi']
        for key in ('roi_unknown_cell_count','roi_free_cell_count','roi_occupied_cell_count'):
            if last[key] != measured[key]:
                raise RuntimeError('progress accumulation and reconstructed occupancy disagree')
    _write_json(out/'observation_progress.json', {key: value[0] for key,value in progression.items()})
    _save_progress(out, progression)
    _condition_comparison(out, case_rows, expected_held[0])
    viewer, _ = export_viewer(out/'runs/occluded_multi')
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader(); writer.writerows(case_rows)
    result = {
        'format':'wtumi-stairs-occlusion-demo-v1', 'scene':cfg['capture']['scene_name'],
        'cases':case_rows, 'paired_inputs_verified':True, 'verification':verification,
        'evaluation':cfg['evaluation'], 'stairs':capture['stairs'], 'roi':capture['roi'],
        'versions':report['versions'], 'capture_metadata':capture.get('metadata', {}),
        'reconstruction':cfg['reconstruction'], 'occupancy':cfg['occupancy'],
        'paths':{'capture':'capture/capture.json', 'summary_csv':'summary.csv',
                 'condition_comparison':'condition_comparison.png', 'observation_progress':'observation_progress.png',
                 'occupancy_animation':'occupancy_progress.gif',
                 'capture_animation':f"capture/{capture['paths']['preview_gif']}",
                 'viewer':viewer.relative_to(out).as_posix(),
                 'replay_configs':{key:f'configs/{key}.yaml' for key in case_sources}},
        'total_seconds':perf_counter()-started,
        'scope':'Known simulator poses, kinematic head-camera/payload rig, static stairs, simulator masks. No walking controller, SLAM, dynamic clearing, tactile fusion or collision-safety guarantee.',
        'evaluation_policy':'Environment-only held-outs and analytic tread/riser samples are evaluation-only. Self/payload pixels are invalid observations in both TSDF and occupancy. Unknown cells are never filled with hidden simulator geometry.',
    }
    _write_json(out/'demo.json', result)
    lines = ['# Staircase occlusion benchmark', '',
             '- [Three-condition RGB-D comparison](condition_comparison.png)',
             '- [Raw and masked head-camera playback]('+result['paths']['capture_animation']+')',
             '- [Observation history](observation_progress.png)',
             '- [Occupancy slice animation](occupancy_progress.gif)',
             '- [Interactive reconstructed map]('+result['paths']['viewer']+')',
             '- [Numeric results](summary.csv)', '- [Run record](demo.json)', '',
             'Each condition uses the same held-out environment references. The single-frame case uses the first frame of the same masked box sequence.', '',
             '| Condition | Frames | Stair depth coverage | Within tolerance | Surface coverage | ROI unknown |',
             '|---|---:|---:|---:|---:|---:|']
    percentage = lambda value: 'N/A' if value is None else f'{100*value:.2f}%'
    for row in case_rows:
        lines.append(f"| {CASE_LABELS[row['case_id']]} | {row['integration_frames']} | {percentage(row['stair_depth_coverage'])} | {percentage(row['stair_depth_within_tolerance_fraction'])} | {percentage(row['surface_coverage_within_tolerance_fraction'])} | {percentage(row['roi_unknown_fraction'])} |")
    lines += ['', f"Distance tolerance: {cfg['evaluation']['depth_tolerance_m']*1000:g} mm. TSDF cells: {case_rows[0]['voxel_size_m']*1000:g} mm; occupancy cells: {case_rows[0]['occupancy_voxel_size_m']*1000:g} mm.", '',
              'Depth errors use overlap within the held-out stair masks. Surface coverage measures distance from analytic tread/riser samples to the reconstructed mesh; it does not measure mesh precision. ROI unknown is measured at occupancy cell centers.', '',
              'Surface samples in free cells are finite-cell diagnostics, not collision-safety probabilities. Previously observed static geometry persists. Unobserved regions remain unknown; the box and robot geometry are retained separately in the capture record.', '',
              'Replay configs use relative dataset paths and remain usable after copying the entire output directory. Resolved configs and source metadata retain provenance paths.', '', result['scope'], '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
