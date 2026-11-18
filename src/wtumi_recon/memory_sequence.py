"""Portable observation schedules and evaluation of memory under box occlusion.

The mixed history selects actual paired observations. It never substitutes
clean reference images for masked fusion inputs. Analytic stair samples and
the counterfactual no-box view are used only by the evaluation function.
"""

from __future__ import annotations

import copy
import hashlib
import json
from numbers import Real
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .occlusion_demo import _array_digest, verify_paired_capture
from .sequence import RGBDSequence, SequenceWriter
from .stair_metrics import sample_stair_surfaces


CASE_IDS = ('always_occluded', 'observe_then_occluded')
OBSERVATION_FILES = ('raw_rgb.png', 'raw_depth_m.npy', 'excluded_mask.png',
                     'box_mask.png', 'robot_mask.png')


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _capture_sequences(capture_dir):
    root = Path(capture_dir).expanduser().resolve()
    report = json.loads((root/'capture.json').read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('paths'), dict):
        raise ValueError('capture.json must contain the paired sequence paths')
    sequences = {}
    for condition in ('unoccluded', 'occluded'):
        value = report['paths'].get(condition+'_manifest')
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError('capture sequence paths must be relative to the capture directory')
        path = (root/value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('capture sequence path escapes the capture directory')
        sequences[condition] = RGBDSequence(path)
    if sequences['unoccluded'].scene_name != sequences['occluded'].scene_name:
        raise ValueError('paired sequences must describe the same scene')
    return root, report, sequences


def _prefix_length(value, count):
    if type(value) is not int or not 1 <= value < count:
        raise ValueError('preobserve_frames must be an integer from 1 to frame_count - 1')
    return value


def _mask(path, shape):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.shape != shape:
        raise ValueError(f'mask must be a single-channel image aligned with depth: {path}')
    return values != 0


def _observation_masks(sequence, view):
    folder = sequence.root/'frames'/view['id']
    shape = (sequence.camera['height'], sequence.camera['width'])
    for name in OBSERVATION_FILES:
        if not (folder/name).is_file():
            raise FileNotFoundError(folder/name)
    masks = {name: _mask(folder/(name+'_mask.png'), shape)
             for name in ('excluded', 'box', 'robot')}
    if not np.array_equal(masks['excluded'], masks['box'] | masks['robot']):
        raise ValueError('excluded mask must equal the union of box and robot masks')
    return masks


def prepare_memory_sequences(capture_dir, output_dir, preobserve_frames=2):
    """Create equal-length blocked and clear-prefix/blocked-suffix histories.

    The source contains paired no-box and box captures at matching poses and
    timestamps. This function schedules those observations as a kinematic
    replay; it does not simulate picking up a box. Every integration record
    uses the selected source's *observation*, including zeroed self pixels.
    Clean integration reference files, if present in the source, are not
    propagated. Shared held-out references remain evaluation-only records.

    Both output sequences own their image files and remain usable after the
    entire sequence directory is moved. Absolute provenance strings are
    descriptive metadata, not dependencies required for replay.
    """
    root, capture, sources = _capture_sequences(capture_dir)
    clear, blocked = sources['unoccluded'], sources['occluded']
    integration = [view for view in clear.views if view['split'] == 'integration']
    k = _prefix_length(preobserve_frames, len(integration))
    out = Path(output_dir).expanduser().resolve()
    if out == root or root.is_relative_to(out) or out.is_relative_to(root):
        raise ValueError('sequence output and source capture directories must not overlap')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'memory sequence output must be new or empty: {out}')
    verification = verify_paired_capture(clear.manifest_path, blocked.manifest_path)
    for sequence in sources.values():
        for view in integration:
            _observation_masks(sequence, view)
    schedule = [
        {'id': view['id'], 'timestamp_s': view['timestamp_s'], 'frame_number': index+1,
         'phase': 'preobserve' if index < k else 'occluded',
         'source_conditions': {'always_occluded': 'occluded',
                               'observe_then_occluded': 'unoccluded' if index < k else 'occluded'}}
        for index, view in enumerate(integration)
    ]
    by_id = {row['id']: row for row in schedule}
    out.mkdir(parents=True, exist_ok=True)
    manifests, observation_hashes = {}, {}
    for case in CASE_IDS:
        metadata = copy.deepcopy(clear.metadata)
        metadata.update(producer='wtumi_recon.memory_sequence', memory_case=case,
                        source_capture=str(root), preobserve_frames=k,
                        memory_schedule=[{**row, 'source_condition': row['source_conditions'][case]}
                                         for row in schedule],
                        capture_motion='Scheduled paired observations; no physical box pickup or physics stepping',
                        reference_policy='Selected masked observations only for integration; shared held-outs are evaluation only')
        writer = SequenceWriter(out/case, clear.camera, clear.scene_name,
                                clear.reference_kind, metadata)
        for view in clear.views:
            if view['split'] == 'integration':
                condition = by_id[view['id']]['source_conditions'][case]
                source = sources[condition]
                rgb, depth = source.read(view)
                writer.add_frame(view['id'], 'integration', rgb, depth, view['T_wc'], view['timestamp_s'])
                for name in OBSERVATION_FILES:
                    shutil.copy2(source.root/'frames'/view['id']/name,
                                 out/case/'frames'/view['id']/name)
            else:
                rgb, depth = clear.read(view)
                reference_rgb, reference_depth = clear.read_reference(view)
                writer.add_frame(view['id'], 'held_out', rgb, depth, view['T_wc'], view['timestamp_s'],
                                 reference_rgb=reference_rgb, reference_depth_m=reference_depth)
        path = writer.finish()
        sequence = RGBDSequence(path)
        manifests[case] = str(path)
        observation_hashes[case] = _array_digest(sequence, 'integration')
        if _array_digest(sequence, 'held_out') != verification['held_out_rgbd_sha256']:
            raise RuntimeError('prepared sequence changed a held-out reference')
        _write_json(out/case/'frame_schedule.json', metadata['memory_schedule'])
    result = {'format': 'wtumi-memory-sequences-v1', 'manifests': manifests,
              'preobserve_frames': k, 'frame_schedule': schedule,
              'verification': {**verification, 'prepared_integration_sha256': observation_hashes},
              'capture_metadata': copy.deepcopy(capture.get('metadata', {})),
              'scope': 'Equal frame count, poses and timestamps; only the first observation conditions differ. '
                       'Paired replay does not simulate a physical pickup action.'}
    # Persist portable paths; callers receive resolved paths for immediate use.
    saved = {**result, 'manifests': {case: f'{case}/manifest.json' for case in CASE_IDS}}
    _write_json(out/'sequences.json', saved)
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _project_samples(samples, camera, pose):
    """Return nearest integer pixel indices and camera-Z for bounded samples."""
    transform = np.asarray(pose, dtype=np.float64)
    local = (samples-transform[:3, 3]) @ transform[:3, :3]
    z = local[:, 2]
    projected = local @ np.asarray(camera['K'], dtype=np.float64).T
    valid = np.isfinite(local).all(axis=1) & (z > 0) & (z < camera['depth_trunc_m'])
    uv = np.zeros((len(samples), 2), dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        uv[valid] = projected[valid, :2]/z[valid, None]
    valid &= np.isfinite(uv).all(axis=1)
    valid &= ((uv[:, 0] >= -.5) & (uv[:, 0] < camera['width']-.5) &
              (uv[:, 1] >= -.5) & (uv[:, 1] < camera['height']-.5))
    pixel = np.zeros((len(samples), 2), dtype=np.int64)
    pixel[valid] = np.floor(uv[valid]+.5).astype(np.int64)
    return pixel[:, 1], pixel[:, 0], z, valid


def _visible_samples(samples, sequence, view, depth, tolerance):
    rows, cols, z, valid = _project_samples(samples, sequence.camera, view['T_wc'])
    observed = depth[rows, cols]
    valid &= (observed > 0) & (np.abs(observed-z) <= tolerance)
    return valid, rows, cols, z


def _cohort_distances(mesh_path, samples):
    import open3d as o3d

    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not len(samples):
        return np.empty(0, dtype=np.float32)
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError('memory reconstruction mesh has nonfinite vertices')
    if not len(mesh.vertices) or not len(mesh.triangles):
        return np.full(len(samples), np.inf, dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = []
    for offset in range(0, len(samples), 100_000):
        values = samples[offset:offset+100_000].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('stair sample coordinates exceed mesh query precision')
        distances.append(scene.compute_distance(o3d.core.Tensor(values)).numpy())
    result = np.concatenate(distances)
    if not np.isfinite(result).all():
        raise ValueError('memory mesh distance query returned nonfinite distances')
    return result


def evaluate_memory_retention(capture_dir, sequence_manifest, mesh_paths, preobserve_frames=2,
                              tolerance_m=.03, sample_spacing_m=.04):
    """Measure reconstruction of a common preobserved, currently hidden cohort.

    The cohort is shared by both conditions: analytic stair samples matching
    an actual no-box depth in the first k frames, and matching current no-box
    depth while current raw box depth lies in front of them. Nearest-pixel
    projection and camera-Z agreement within tolerance_m are visibility
    proxies; no claim is made about exact continuous visibility or area.

    Before the switch the cohort is empty. Afterwards, a sample is counted
    as represented when its nearest reconstructed triangle is within the
    same metric tolerance. This measures current representation of the
    cohort, not persistence of triangle IDs or collision safety. Held-out
    arrays are not read. Evaluation geometry never enters fusion.
    """
    tolerance = _positive(tolerance_m, 'tolerance_m')
    spacing = _positive(sample_spacing_m, 'sample_spacing_m')
    _, capture, sources = _capture_sequences(capture_dir)
    clear, blocked = sources['unoccluded'], sources['occluded']
    sequence = RGBDSequence(sequence_manifest)
    views = [view for view in clear.views if view['split'] == 'integration']
    selected = [view for view in sequence.views if view['split'] == 'integration']
    blocked_views = [view for view in blocked.views if view['split'] == 'integration']
    k = _prefix_length(preobserve_frames, len(views))
    if sequence.camera != clear.camera or blocked.camera != clear.camera:
        raise ValueError('memory evaluation cameras must match')
    if len(selected) != len(views) or len(blocked_views) != len(views):
        raise ValueError('memory evaluation requires the complete paired integration schedule')
    for source_views in (selected, blocked_views):
        for a, b in zip(views, source_views):
            if any(a[key] != b[key] for key in ('id', 'timestamp_s')) or not np.allclose(a['T_wc'], b['T_wc'], atol=1e-10, rtol=0):
                raise ValueError('memory evaluation poses, timestamps and IDs must match')
    if not isinstance(mesh_paths, dict) or set(mesh_paths) != {view['id'] for view in views}:
        raise ValueError('mesh_paths must contain one prefix mesh for every integration frame')
    for path in mesh_paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    samples = sample_stair_surfaces(capture['stairs'], spacing)
    preobserved = np.zeros(len(samples), dtype=bool)
    for view in views[:k]:
        _, depth = clear.read(view)
        visible, _, _, _ = _visible_samples(samples, clear, view, depth, tolerance)
        preobserved |= visible
    preobserved_hash = hashlib.sha256(np.flatnonzero(preobserved).astype('<i8').tobytes()).hexdigest()
    actually_observed = np.zeros(len(samples), dtype=bool)
    reports = []
    for index, view in enumerate(views):
        _, selected_depth = sequence.read(view)
        visible, _, _, _ = _visible_samples(samples, sequence, view, selected_depth, tolerance)
        actually_observed |= visible
        cohort = np.zeros(len(samples), dtype=bool)
        if index >= k:
            _, clear_depth = clear.read(view)
            visible, rows, cols, z = _visible_samples(samples, clear, view, clear_depth, tolerance)
            masks = _observation_masks(blocked, view)
            raw = np.load(blocked.root/'frames'/view['id']/'raw_depth_m.npy', allow_pickle=False)
            _, blocked_depth = blocked.read(view)
            shape = (blocked.camera['height'], blocked.camera['width'])
            if raw.shape != shape or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('raw box depth must be finite, nonnegative and aligned')
            box_depth = raw[rows, cols]
            cohort = (preobserved & visible & masks['box'][rows, cols] &
                      (box_depth > 0) & (box_depth+tolerance < z) & (blocked_depth[rows, cols] == 0))
        distances = _cohort_distances(mesh_paths[view['id']], samples[cohort])
        count, recovered = int(cohort.sum()), int(np.count_nonzero(distances <= tolerance))
        finite = distances[np.isfinite(distances)]
        reports.append({
            'id': view['id'], 'timestamp_s': view['timestamp_s'], 'frames': index+1,
            'phase': 'preobserve' if index < k else 'occluded',
            'surface_sample_count': len(samples), 'preobserved_sample_count': int(preobserved.sum()),
            'preobserved_sample_indices_sha256': preobserved_hash,
            'cohort_sample_count': count,
            'cohort_sample_indices_sha256': hashlib.sha256(np.flatnonzero(cohort).astype('<i8').tobytes()).hexdigest(),
            'reconstructed_cohort_sample_count': recovered,
            'reconstructed_cohort_fraction': recovered/count if count else None,
            'mean_nearest_mesh_distance_m': float(finite.mean()) if len(finite) else None,
            'actually_observed_sample_count': int(actually_observed.sum()),
            'never_observed_sample_count': int((~actually_observed).sum()),
            'visibility_depth_tolerance_m': tolerance, 'surface_distance_tolerance_m': tolerance,
            'sample_spacing_max_m': spacing, 'ground_truth_used_for_fusion': False,
            'cohort_definition': 'Stair samples visible in the shared no-box prefix and currently hidden by measured box pixels',
            'scope': 'Nearest-pixel camera-Z visibility proxy and nearest-triangle representation; '
                     'not an exact surface-area, triangle-persistence or collision-safety score. '
                     'Unobserved samples do not imply entirely unknown finite occupancy cells.'})
    return reports
