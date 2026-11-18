"""Same-pose observation histories for mapping before and after box occlusion.

The two histories intentionally contain different early visual information.
Paired capture replay schedules that difference; it does not simulate pickup.
Prefix reconstructions replay cumulative observations, not future frames.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import yaml

from .occlusion_demo import (_normalize_occlusion_demo_config, _summary_row,
                             verify_paired_capture, write_prefix_manifest)
from .sequence import RGBDSequence


CASE_LABELS = {
    'always_occluded': 'Always carrying the box',
    'observe_then_occluded': 'Observe first, then carry the box',
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_memory_demo_config(path):
    """Validate configuration before capture, filesystem writes or rendering."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or type(raw.get('schema_version')) is not int:
        raise ValueError('memory demo requires a schema_version 1 mapping')
    raw = copy.deepcopy(raw)
    count = raw.pop('preobserve_frames', None)
    incremental = raw.pop('incremental_prefixes', False)
    if type(incremental) is not bool:
        raise ValueError('incremental_prefixes must be a boolean')
    playback = raw.pop('playback', {})
    if not isinstance(playback, dict) or set(playback)-{'fps', 'scale', 'mp4'}:
        raise ValueError('playback supports fps, scale and mp4 only')
    fps, scale, mp4 = playback.get('fps', 1000/900), playback.get('scale', 1), playback.get('mp4', False)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError('playback.fps must be finite and between 1 and 60')
    if type(scale) is not int or scale not in (1, 2):
        raise ValueError('playback.scale must be 1 or 2')
    if type(mp4) is not bool:
        raise ValueError('playback.mp4 must be a boolean')
    cfg = _normalize_occlusion_demo_config(raw, path)
    frames = len(cfg['capture']['trajectory'])
    if type(count) is not int or not 1 <= count < frames:
        raise ValueError('preobserve_frames must be an integer from 1 to the integration frame count minus 1')
    if frames > 20 and not incremental:
        raise ValueError('more than 20 integration frames requires incremental_prefixes: true')
    cfg['incremental_prefixes'] = incremental
    cfg['playback'] = {'fps': float(fps), 'scale': scale, 'mp4': mp4}
    cfg['preobserve_frames'] = count
    return cfg


def _capture_asset(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('capture asset paths must be relative')
    path = (root/value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'capture asset must be a file inside the capture directory: {value}')
    return path


def _validate_saved_capture(root, cfg):
    """Do not silently pair saved observations with a different scene/config."""
    from .mujoco_capture import intrinsics_from_fovy
    from .render import look_at_pose
    from .rig_capture import rig_transform, head_camera_pose

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('saved capture may not contain symlinks')
    capture = json.loads((root/'capture.json').read_text(encoding='utf-8'))
    expected = cfg['capture']
    if capture.get('format') != 'wtumi-rig-capture-v1':
        raise ValueError('saved capture must be a paired rig capture')
    for name in ('scene_name', 'stairs', 'roi'):
        if capture.get(name) != expected[name]:
            raise ValueError(f'saved capture {name} differs from the selected configuration')
    if capture.get('rig') != {name:expected[name] for name in ('camera','torso','box')}:
        raise ValueError('saved capture rig differs from the selected configuration')
    paths = capture.get('paths', {})
    clear = _capture_asset(root, paths.get('unoccluded_manifest'))
    blocked = _capture_asset(root, paths.get('occluded_manifest'))
    # The sequence preparation API uses this stable portable capture layout.
    if clear != root/'unoccluded/manifest.json' or blocked != root/'occluded/manifest.json':
        raise ValueError('saved capture must use the paired unoccluded/occluded layout')
    sequence = RGBDSequence(clear)
    camera = expected['camera']
    k = intrinsics_from_fovy(camera['width'],camera['height'],camera['fovy_degrees'])
    if (sequence.camera['width'] != camera['width'] or sequence.camera['height'] != camera['height']
            or sequence.camera['depth_trunc_m'] != camera['depth_trunc_m']
            or not np.allclose(sequence.camera['K'], k, rtol=0, atol=1e-8)):
        raise ValueError('saved capture calibration differs from the selected configuration')
    integration = [v for v in sequence.views if v['split']=='integration']
    held_out = [v for v in sequence.views if v['split']=='held_out']
    for saved, planned, split in ((integration,expected['trajectory'],'integration'),
                                  (held_out,expected['held_out'],'held_out')):
        if len(saved) != len(planned):
            raise ValueError(f'saved capture {split} frame count differs from configuration')
        for view, frame in zip(saved,planned):
            pose = (rig_transform(frame['position_m'],frame['yaw_degrees']) @ head_camera_pose(camera,frame)
                    if split=='integration' else look_at_pose(frame['eye'],frame['target'],frame['up']))
            if (view['id'] != frame['id'] or view['timestamp_s'] != frame['timestamp_s']
                    or not np.allclose(view['T_wc'],pose,rtol=0,atol=1e-8)):
                raise ValueError(f'saved capture {split} pose schedule differs from configuration')
    ids = [v['id'] for v in held_out]
    masks = paths.get('held_out_masks', {})
    if set(masks) != set(ids):
        raise ValueError('saved capture must supply masks for every held-out view')
    for value in masks.values():
        _capture_asset(root,value)
    verify_paired_capture(clear,blocked)
    return capture


def _write_run_config(path, manifest, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {'schema_version':3,
                  'source':{'type':'rgbd_sequence','manifest':os.path.relpath(manifest,path.parent)},
                  'reconstruction':cfg['reconstruction'], 'occupancy':cfg['occupancy'],
                  'evaluation':{'depth_tolerance_m':cfg['evaluation']['depth_tolerance_m']}}
    path.write_text(yaml.safe_dump(run_config,sort_keys=False),encoding='utf-8')


def _write_readme(out, result):
    lines = ['# M7: observation memory under box occlusion', '',
             '[Open the playback gallery](index.html) · [Temporal 3D maps](history_viewer.html)', '',
             'Both histories use the same poses, timestamps, frame count and held-out references. '
             'The observe-first history intentionally has more early visual information; this is an observation-history comparison, not a comparison of two mapping algorithms.', '',
             f"The box is absent for the first {result['preobserve_frames']} frames of the observe-first history and present thereafter. "
             'Paired simulator observations are replayed on this schedule. Pickup and locomotion are not simulated.', '',
             '| Condition | Frames | Stair depth coverage | Mean absolute depth error | Surface samples within 30 mm* | Unknown region fraction |',
             '|---|---:|---:|---:|---:|---:|']
    for row in result['cases']:
        mean = row['stair_depth_mean_absolute_error_m']
        error = 'N/A' if mean is None else f'{1000*mean:.2f} mm'
        lines.append(f"| {CASE_LABELS[row['case_id']]} | {row['integration_frames']} | "
                     f"{100*row['stair_depth_coverage']:.2f}% | {error} | "
                     f"{100*row['surface_coverage_within_tolerance_fraction']:.2f}% | "
                     f"{100*row['roi_unknown_fraction']:.2f}% |")
    tolerance = result['evaluation']['depth_tolerance_m']*1000
    lines += ['', f'*The configured surface-distance tolerance is {tolerance:g} mm. '
              'Surface coverage is ground-truth sample to mesh completeness, not mesh precision. '
              'Depth errors exclude missing predictions. Unknown region fractions include hidden volumes and solid interiors.', '',
              '## Previously observed, currently hidden samples', '',
              '| Condition | Common hidden samples at the final frame | Samples represented within tolerance |',
              '|---|---:|---:|']
    for row in result['cases']:
        fraction = row['preobserved_hidden_reconstruction_fraction']
        displayed = 'N/A' if fraction is None else f'{100*fraction:.2f}%'
        lines.append(f"| {CASE_LABELS[row['case_id']]} | {row['preobserved_hidden_sample_count']} | {displayed} |")
    lines += ['', 'Both conditions use the same evaluation cohort: stair samples visible in the no-box prefix '
              'and currently behind a measured box pixel. Visibility uses nearest-pixel projection and camera-depth agreement. '
              'This is sampled surface representation, not exact triangle identity or a guarantee about every unobserved voxel.', '',
              '## Inspect and replay', '']
    for case in CASE_LABELS:
        lines.append(f"- [{CASE_LABELS[case]}](runs/{case}/viewer.html): final map.")
    lines += [f"- [Capture history]({result['paths']['capture_animation']}): the actual scheduled raw/masked observations.",
              f"- [Reconstruction history]({result['paths']['reconstruction_animation']}): cumulative maps rendered from one shared evaluation viewpoint.",
              f"- [Occupancy history]({result['paths']['occupancy_animation']}): a fixed spatial slice over time.",
              '- [Observation curves](observation_progress.png): completeness and unknown-state progression.',
              '- [Memory retention diagnostics](memory_retention.json): common preobserved samples currently hidden by the box.',
              '- [Per-frame metrics](observation_history.json) and [final numeric results](summary.csv).', '',
              'The temporal viewer changes the number of fused observations. Final-viewer occupancy sliders select spatial slices. '
              'Prefix maps contain only observations available at that frame. With incremental_prefixes enabled, each observation is fused once per condition. '
              'Intermediate snapshots export held-out evaluations and omit the dense TSDF archive; final results include full exports. '
              'This benchmark does not establish real-time performance.', '',
              f"Requested playback: {result['playback']['fps']:g} frames per second. Browser mesh playback is best effort. "
              'Encoded playback speed is separate from mapping throughput and camera timestamp spacing.', '',
              'Copy the entire output directory to preserve relative replay paths. For example:', '', '```bash',
              'python -m wtumi_recon run --config outputs/your_run/configs/observe_then_occluded.yaml', '```', '',
              'Known simulator poses and masks are used. Box/self pixels are invalid before both map updates. '
              'Held-out images and analytic stair samples are evaluation-only. There is no hidden-geometry filling, tactile fusion or dynamic clearing.', '']
    # Avoid fixing the displayed threshold when a user changes the configuration.
    lines = [line.replace('within 30 mm*',f'within {tolerance:g} mm*') for line in lines]
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def run_memory_demo(config_path, output_dir, capture_dir=None, progress=None):
    """Capture or reuse paired frames, replay both histories, export every prefix."""
    from .rig_capture import capture_rig
    from .memory_sequence import prepare_memory_sequences, evaluate_memory_retention
    from .pipeline import ReconstructionState, run_reconstruction
    from .stair_metrics import evaluate_stair_result
    from .viewer import export_viewer
    from .memory_visuals import export_memory_visuals

    started = perf_counter()
    cfg = load_memory_demo_config(config_path)
    if cfg['playback']['mp4']:
        from .video import require_video_encoder
        require_video_encoder()
    out = Path(output_dir).expanduser().resolve()
    source = Path(capture_dir).expanduser().resolve() if capture_dir is not None else None
    if source is not None and (out.is_relative_to(source) or source.is_relative_to(out)):
        raise ValueError('memory output and source capture directories must not overlap')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'memory demo output must be new or empty: {out}')
    if source is not None:
        _validate_saved_capture(source,cfg)
    out.mkdir(parents=True,exist_ok=True)
    (out/'demo_source.yaml').write_text(Path(cfg['_path']).read_text(),encoding='utf-8')
    if source is None:
        if progress: progress('Capturing paired head-camera observations for the two histories')
        capture_rig(cfg['capture_config'],out/'capture',progress=progress)
    else:
        if progress: progress('Copying the saved paired capture; source files remain unchanged')
        shutil.copytree(source,out/'capture')
    capture = _validate_saved_capture(out/'capture',cfg)
    prepared = prepare_memory_sequences(out/'capture',out/'sequences',cfg['preobserve_frames'])
    _write_json(out/'sequence_preparation.json',prepared)
    sequences = {case:Path(path) for case,path in prepared['manifests'].items()}
    if set(sequences) != set(CASE_LABELS):
        raise RuntimeError('prepared memory histories do not match the required cases')
    masks = {name:_capture_asset(out/'capture',path) for name,path in capture['paths']['held_out_masks'].items()}
    histories, final_rows, mesh_paths = {}, [], {}
    versions = None
    for case in CASE_LABELS:
        sequence = RGBDSequence(sequences[case])
        views = [v for v in sequence.views if v['split']=='integration']
        held_ids = [v['id'] for v in sequence.views if v['split']=='held_out']
        histories[case], mesh_paths[case] = [], {}
        prefix_state = ReconstructionState() if cfg['incremental_prefixes'] else None
        for index, view in enumerate(views,1):
            if progress: progress(f'Reconstructing {CASE_LABELS[case]}: cumulative frame {index}/{len(views)}')
            final = index == len(views)
            manifest = sequences[case] if final else write_prefix_manifest(sequences[case],index)
            relative = f'runs/{case}' if final else f'history/{case}/frame_{index:03d}'
            config = out/'configs'/(f'{case}.yaml' if final else f'history/{case}/frame_{index:03d}.yaml')
            _write_run_config(config,manifest,cfg)
            if prefix_state is None:
                report = run_reconstruction(config,out/relative)
            else:
                report = run_reconstruction(config,out/relative,prefix_state=prefix_state,compact=not final)
            if report['integrated_frame_ids'] != [v['id'] for v in views[:index]] or report['held_out_frame_ids'] != held_ids:
                raise RuntimeError('prefix reconstruction used an incorrect observation/evaluation split')
            metrics = evaluate_stair_result(out/relative,capture['stairs'],capture['roi'],masks,
                       tolerance_m=cfg['evaluation']['depth_tolerance_m'],sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
            _write_json(out/relative/'stair_metrics.json',metrics)
            row = _summary_row(case,relative,report,metrics)
            row['timing_scope'] = report['timing_scope']
            early_clear = case=='observe_then_occluded' and index <= cfg['preobserve_frames']
            row.update(id=view['id'],frames=index,timestamp_s=view['timestamp_s'],
                       phase='pre_observation' if early_clear else 'occluded',
                       source_condition='unoccluded' if early_clear else 'occluded')
            histories[case].append(row)
            mesh_paths[case][view['id']] = out/relative/'mesh.ply'
            if final:
                export_viewer(out/relative)
                final_rows.append(dict(row))
                versions = report['versions']
    if progress: progress('Measuring recovery of previously observed stair samples now hidden by the box')
    retention = {}
    for case in CASE_LABELS:
        retention[case] = evaluate_memory_retention(out/'capture',sequences[case],mesh_paths[case],
            preobserve_frames=cfg['preobserve_frames'],tolerance_m=cfg['evaluation']['depth_tolerance_m'],
            sample_spacing_m=cfg['evaluation']['sample_spacing_m'])
        if len(retention[case]) != len(histories[case]):
            raise RuntimeError('memory diagnostic length differs from observation history')
        for row,metric in zip(histories[case],retention[case]):
            row['memory_retention'] = metric
    for index in range(len(retention['always_occluded'])):
        a,b = (retention[case][index] for case in CASE_LABELS)
        for key in ('id','cohort_sample_count','cohort_sample_indices_sha256','preobserved_sample_indices_sha256'):
            if a[key] != b[key]:
                raise RuntimeError('memory conditions must share the same evaluation cohort')
    for row in final_rows:
        last = retention[row['case_id']][-1]
        row['preobserved_hidden_sample_count'] = last['cohort_sample_count']
        row['preobserved_hidden_reconstructed_sample_count'] = last['reconstructed_cohort_sample_count']
        row['preobserved_hidden_reconstruction_fraction'] = last['reconstructed_cohort_fraction']
    _write_json(out/'memory_retention.json',retention)
    _write_json(out/'observation_history.json',histories)
    if progress: progress('Exporting capture, reconstruction, occupancy and interactive 3D playback')
    paths = export_memory_visuals(out,histories,sequences,preobserve_frames=cfg['preobserve_frames'],playback=cfg['playback'])
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(final_rows[0]))
        writer.writeheader(); writer.writerows(final_rows)
    result = {
        'format':'wtumi-memory-demo-v1','scene':capture['scene_name'],'cases':final_rows,
        'preobserve_frames':cfg['preobserve_frames'],'histories':histories,
        'incremental_prefixes':cfg['incremental_prefixes'],'playback':cfg['playback'],
        'paired_inputs_verified':True,'verification':prepared['verification'],
        'frame_schedule':prepared['frame_schedule'],'capture_metadata':capture.get('metadata',{}),
        'capture_origin':{'mode':'saved_capture_replay' if source is not None else 'fresh_simulator_capture',
                          'source_capture_dir':str(source) if source is not None else None},
        'versions':versions,'reconstruction':cfg['reconstruction'],'occupancy':cfg['occupancy'],
        'evaluation':cfg['evaluation'],'stairs':capture['stairs'],'roi':capture['roi'],
        'paths':{**paths,'summary_csv':'summary.csv','history':'observation_history.json',
                 'memory_retention':'memory_retention.json','capture':'capture/capture.json',
                 'viewers':{case:f'runs/{case}/viewer.html' for case in CASE_LABELS},
                 'replay_configs':{case:f'configs/{case}.yaml' for case in CASE_LABELS}},
        'scope':'Scheduled replay of paired simulator observations; no pickup or locomotion physics. '
                'Known poses/masks, static environment and cumulative prefix reconstruction. No tactile fusion or dynamic clearing.',
        'evaluation_policy':'Same poses, frame budget and held-outs; different early visual information by design. '
                            'Held-out references and analytical stair samples never update either map.',
        'total_seconds':perf_counter()-started,
    }
    _write_json(out/'demo.json',result)
    _write_readme(out,result)
    return result
