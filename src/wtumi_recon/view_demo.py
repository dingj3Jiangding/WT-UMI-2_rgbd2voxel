"""Controlled camera orientation experiments with a continuously carried box."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .memory_demo import _validate_saved_capture, _write_run_config
from .occlusion_demo import _normalize_occlusion_demo_config, _array_digest, _summary_row, write_prefix_manifest
from .pipeline import ReconstructionState, run_reconstruction, write_json
from .sequence import RGBDSequence

CASES = ('fixed_head', 'head_scan')
LABELS = {'fixed_head': 'Fixed head orientation', 'head_scan': 'Independent head scan'}


def load_view_demo_config(path):
    """Reject unmatched body paths and settings before capture or filesystem writes."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or type(raw.get('schema_version')) is not int:
        raise ValueError('view demo requires a schema_version 1 configuration mapping')
    raw = copy.deepcopy(raw)
    fixed, scan = raw.pop('fixed_capture_config', None), raw.pop('scan_capture_config', None)
    playback = raw.pop('playback', {'fps': 3, 'scale': 1, 'mp4': False})
    if not isinstance(playback, dict) or set(playback)-{'fps', 'scale', 'mp4'}:
        raise ValueError('playback supports fps, scale and mp4 only')
    fps, scale, mp4 = playback.get('fps', 3), playback.get('scale', 1), playback.get('mp4', False)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError('playback.fps must be finite and between 1 and 60')
    if type(scale) is not int or scale not in (1, 2) or type(mp4) is not bool:
        raise ValueError('playback.scale must be 1 or 2 and mp4 must be boolean')
    if 'capture_config' in raw:
        raise ValueError('use fixed_capture_config and scan_capture_config')
    configs = {case: _normalize_occlusion_demo_config({**raw, 'capture_config': source}, path)
               for case, source in zip(CASES, (fixed, scan))}
    a, b = [configs[case]['capture'] for case in CASES]
    for key in ('scene_name', 'camera', 'stairs', 'box', 'torso', 'roi', 'held_out'):
        if a[key] != b[key]:
            raise ValueError(f'view conditions must share {key}')
    if len(a['trajectory']) != len(b['trajectory']):
        raise ValueError('view conditions must share the same frame budget')
    different = False
    for first, second in zip(a['trajectory'], b['trajectory']):
        for key in ('id', 'timestamp_s', 'position_m', 'yaw_degrees'):
            if first[key] != second[key]:
                raise ValueError(f'view conditions must share body trajectories and timestamps: {key}')
        if first['head_yaw_degrees'] != 0 or first['head_pitch_degrees'] != 0:
            raise ValueError('fixed_head must have zero head yaw and pitch offsets')
        different |= second['head_yaw_degrees'] != 0 or second['head_pitch_degrees'] != 0
    if not different:
        raise ValueError('head_scan must include an independent head orientation change')
    result = configs['fixed_head']
    result['capture_configs'] = {case: configs[case]['capture_config'] for case in CASES}
    result['captures'] = {case: configs[case]['capture'] for case in CASES}
    result['playback'] = {'fps': float(fps), 'scale': scale, 'mp4': mp4}
    return result


def _verify_conditions(out, captures, cfg):
    """Verify generated observations, not just the requested configuration."""
    loaded = {}
    for case in CASES:
        root = out/'captures'/case
        _validate_saved_capture(root, {'capture': cfg['captures'][case]})
        loaded[case] = RGBDSequence(root/'occluded/manifest.json')
    a, b = [loaded[case] for case in CASES]
    if a.camera != b.camera:
        raise RuntimeError('captured calibration differs between view conditions')
    different = False
    for first, second in zip(a.views, b.views):
        if any(first[key] != second[key] for key in ('id', 'timestamp_s', 'split')):
            raise RuntimeError('captured view schedules differ')
        if not np.allclose(np.array(first['T_wc'])[:3,3], np.array(second['T_wc'])[:3,3], atol=1e-8, rtol=0):
            raise RuntimeError('camera centers changed in an orientation-only comparison')
        if first['split'] == 'integration':
            different |= not np.allclose(first['T_wc'], second['T_wc'], atol=1e-8, rtol=0)
        elif not np.allclose(first['T_wc'], second['T_wc'], atol=1e-8, rtol=0):
            raise RuntimeError('held-out camera poses differ')
    if not different:
        raise RuntimeError('capture did not apply the requested head rotation')
    hashes = [_array_digest(loaded[case], 'held_out') for case in CASES]
    if hashes[0] != hashes[1]:
        raise RuntimeError('view conditions must share identical held-out RGB-D references')
    from .rig_capture import rig_transform
    for case in CASES:
        records = captures[case].get('frames')
        planned = cfg['captures'][case]['trajectory']
        if not isinstance(records, list) or len(records) != len(planned):
            raise RuntimeError('capture body/payload records must cover every integration frame')
        for record, frame in zip(records, planned):
            if any(record.get(key) != frame[key] for key in ('id', 'timestamp_s')):
                raise RuntimeError('capture body/payload record schedule differs from integration frames')
            rig = rig_transform(frame['position_m'], frame['yaw_degrees'])
            box_local = np.eye(4)
            box_local[:3, 3] = cfg['captures'][case]['box']['position_in_rig_m']
            for key, expected in (('T_wrig', rig), ('T_wbox', rig @ box_local)):
                actual = np.asarray(record.get(key), dtype=float)
                if actual.shape != (4, 4) or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-8, rtol=0):
                    raise RuntimeError('body or payload pose changed from the shared configured trajectory')
    masks = {}
    for case in CASES:
        masks[case] = {name: out/'captures'/case/rel for name, rel in captures[case]['paths']['held_out_masks'].items()}
    from PIL import Image
    for name in masks['fixed_head']:
        if not np.array_equal(np.asarray(Image.open(masks['fixed_head'][name])), np.asarray(Image.open(masks['head_scan'][name]))):
            raise RuntimeError('held-out staircase masks differ')
    return loaded, masks, {'equal_body_and_payload_poses': True, 'equal_camera_centers': True,
        'different_head_orientations': True, 'equal_frame_budget': True,
        'held_out_rgbd_sha256': hashes[0], 'box_present_in_all_integration_frames': True,
        'integration_frame_ids': [v['id'] for v in a.views if v['split']=='integration'],
        'held_out_frame_ids': [v['id'] for v in a.views if v['split']=='held_out']}


def _write_readme(out, result):
    lines = ['# M8: independent head orientation while carrying a box', '',
        '[Open the playback gallery](index.html)', '',
        'Both conditions use the same body and payload path, camera centers, timestamps, frame budget and mapping settings. '
        'The box is present throughout. Only the camera orientation schedule changes. '
        'This is a prescribed scan, not a learned or online view planner.', '',
        '| Condition | Stair depth coverage | Surface samples within tolerance | Never observed surface samples | Unknown region fraction |',
        '|---|---:|---:|---:|---:|']
    for row in result['cases']:
        lines.append(f"| {LABELS[row['case_id']]} | {100*row['stair_depth_coverage']:.2f}% | "
                     f"{100*row['surface_coverage_within_tolerance_fraction']:.2f}% | "
                     f"{row['never_observed_sample_count']} | {100*row['roi_unknown_fraction']:.2f}% |")
    lines += ['', 'Rotation can bring geometry into the field of view but cannot unblock a physical box occlusion from a fixed camera center. '
              'This experiment does not assume that scanning improves coverage. The complete body path is shared between conditions.', '',
              'Observed surface samples use nearest-pixel projection and camera-Z agreement with actual masked integration depth. '
              'Never observed samples are an evaluation-only surface diagnostic; they are not a count of unknown voxels. '
              'Analytic stair geometry, clean paired captures and held-out references never update the maps.', '',
              f"Surface-distance and visibility tolerance: {1000*result['evaluation']['depth_tolerance_m']:g} mm. "
              'Completeness does not establish mesh precision or collision clearance. Depth errors exclude missing predictions.', '',
              '## Outputs', '', '- [Numerical results](summary.csv)', '- [Per-frame history](observation_history.json)',
              '- [Common-sample visibility comparison](visibility_evaluation.json)',
              '- [Run configuration and provenance](demo.json)', '',
              'Each input is fused once per condition. Intermediate outputs omit full TSDF archives and integration-camera renders. '
              'Final outputs include full geometry, occupancy and replay configuration. Playback FPS is not measured processing throughput.', '',
              'Known simulator poses and segmentation, a static environment and kinematic camera/body motion are used. '
              'No neck dynamics, locomotion controller, tactile fusion or localization error is simulated.', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def run_view_demo(config_path, output_dir, progress=None):
    """Capture, integrate and evaluate equal-budget fixed/rotating head histories."""
    from .rig_capture import capture_rig
    from .stair_metrics import evaluate_stair_result
    from .view_metrics import evaluate_view_visibility, compare_view_visibility
    from .view_visuals import export_view_visuals
    from .viewer import export_viewer

    started = perf_counter()
    cfg = load_view_demo_config(config_path)
    if cfg['playback']['mp4']:
        from .video import require_video_encoder
        require_video_encoder()
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'view demo output must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'demo_source.yaml').write_text(Path(config_path).read_text(),encoding='utf-8')
    captures = {}
    for case in CASES:
        if progress: progress(f'Capturing {LABELS[case]} with the box present')
        captures[case] = capture_rig(cfg['capture_configs'][case],out/'captures'/case,progress=progress,preview=False)
    loaded, masks, verification = _verify_conditions(out,captures,cfg)
    histories, final_rows, visibility = {}, [], {}
    for case in CASES:
        sequence = loaded[case]
        views = [v for v in sequence.views if v['split']=='integration']
        state, meshes, histories[case] = ReconstructionState(), {}, []
        for index, view in enumerate(views,1):
            if progress: progress(f'Reconstructing {LABELS[case]}: cumulative frame {index}/{len(views)}')
            final = index == len(views)
            source = sequence.manifest_path if final else write_prefix_manifest(sequence.manifest_path,index)
            relative = f'runs/{case}' if final else f'history/{case}/frame_{index:03d}'
            config = out/'configs'/(f'{case}.yaml' if final else f'history/{case}/frame_{index:03d}.yaml')
            _write_run_config(config,source,cfg)
            report = run_reconstruction(config,out/relative,prefix_state=state,compact=not final)
            if report['integrated_frame_ids'] != [v['id'] for v in views[:index]] or report['held_out_frame_ids'] != verification['held_out_frame_ids']:
                raise RuntimeError('reconstruction violated the observation/evaluation split')
            measured = evaluate_stair_result(out/relative,cfg['capture']['stairs'],cfg['capture']['roi'],masks[case],
                tolerance_m=cfg['evaluation']['depth_tolerance_m'],sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
            write_json(out/relative/'stair_metrics.json',measured)
            row = _summary_row(case,relative,report,measured)
            row.update(id=view['id'],frames=index,timestamp_s=view['timestamp_s'],timing_scope=report['timing_scope'])
            histories[case].append(row)
            meshes[view['id']] = out/relative/'mesh.ply'
            if final: export_viewer(out/relative)
        if progress: progress(f'Evaluating observed and never-observed staircase samples: {LABELS[case]}')
        visibility[case] = evaluate_view_visibility(sequence.manifest_path,cfg['capture']['stairs'],meshes,
            tolerance_m=cfg['evaluation']['depth_tolerance_m'],sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
        for row, observed in zip(histories[case],visibility[case]['frames']):
            if row['id'] != observed['id']: raise RuntimeError('visibility schedule differs from history')
            row['visibility'] = observed
        final_row = dict(histories[case][-1])
        final_row.pop('visibility')
        for key in ('cumulative_observed_sample_count','never_observed_sample_count'):
            final_row[key] = visibility[case]['final'][key]
        final_rows.append(final_row)
    visibility_comparison = compare_view_visibility(visibility['fixed_head'],visibility['head_scan'])
    write_json(out/'visibility_evaluation.json',{'cases':visibility,'comparison':visibility_comparison})
    write_json(out/'observation_history.json',histories)
    if progress: progress('Exporting head-camera, reconstructed map and occupancy playback')
    paths = export_view_visuals(out,histories,{case:sequence.manifest_path for case,sequence in loaded.items()},playback=cfg['playback'])
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(final_rows[0])); writer.writeheader(); writer.writerows(final_rows)
    result={'format':'wtumi-view-demo-v1','scene':cfg['capture']['scene_name'],'cases':final_rows,
        'histories':histories,'verification':verification,'visibility_comparison':visibility_comparison,
        'playback':cfg['playback'],'reconstruction':cfg['reconstruction'],'occupancy':cfg['occupancy'],
        'evaluation':cfg['evaluation'],'versions':report['versions'],
        'capture_metadata':{case:cap['metadata'] for case,cap in captures.items()},
        'paths':{**paths,'summary_csv':'summary.csv','history':'observation_history.json',
                 'visibility_evaluation':'visibility_evaluation.json',
                 'viewers':{case:f'runs/{case}/viewer.html' for case in CASES},
                 'replay_configs':{case:f'configs/{case}.yaml' for case in CASES}},
        'scope':'Independent prescribed camera orientation only; same body/payload path and camera centers. '
                'Box present throughout; exact simulator poses and masks; no physics stepping or tactile fusion.',
        'evaluation_policy':'Analytic surface samples, no-box captures and held-out references are evaluation-only and never enter fusion.',
        'total_seconds':perf_counter()-started}
    write_json(out/'demo.json',result)
    _write_readme(out,result)
    return result
