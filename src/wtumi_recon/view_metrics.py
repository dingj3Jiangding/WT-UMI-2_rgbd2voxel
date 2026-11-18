"""Evaluation-only visibility histories for independent camera-view experiments.

Analytic samples are compared with masked integration observations, never with
clean references. Unobserved surface samples are distinct from unknown cells:
a coarse cell can contain ray evidence even when a particular sample was never
seen. These diagnostics neither integrate nor fill any map.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .memory_sequence import _cohort_distances, _positive, _visible_samples
from .sequence import RGBDSequence
from .stair_metrics import sample_stair_surfaces


VISIBILITY_SCOPE = (
    'Nearest-pixel projection and camera-Z agreement with actual masked integration depth. '
    'Samples have equal weight and cover exposed stair treads and front risers only. '
    'Never observed means not depth-consistent in any supplied observation; it includes '
    'occlusion, missing depth and locations outside the image or depth range. It does not '
    'identify why a sample was unseen or imply that its entire finite occupancy cell is unknown.'
)


def _indices(mask):
    return np.flatnonzero(mask).astype(np.int64).tolist()


def _indices_hash(mask):
    return hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()


def evaluate_view_visibility(sequence_manifest, stairs, mesh_paths=None,
                             tolerance_m=.03, sample_spacing_m=.04):
    """Evaluate each cumulative integration prefix without reading held-outs.

    ``mesh_paths`` may supply one actual prefix mesh per integration frame ID.
    In that case, nearest-triangle representation is reported separately from
    measured visibility. A represented but unseen sample can arise from surface
    interpolation and must not be described as a new camera observation.

    Final sample indices and the ordered coordinate hash support set comparisons
    between different camera orientations on the same staircase. Reference files
    need to exist as required by RGBDSequence, but their contents are not read.
    """
    tolerance = _positive(tolerance_m, 'tolerance_m')
    spacing = _positive(sample_spacing_m, 'sample_spacing_m')
    sequence = RGBDSequence(sequence_manifest)
    views = [view for view in sequence.views if view['split'] == 'integration']
    if mesh_paths is not None:
        if not isinstance(mesh_paths, dict) or set(mesh_paths) != {v['id'] for v in views}:
            raise ValueError('mesh_paths must contain one prefix mesh for every integration frame')
        for path in mesh_paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    samples = sample_stair_surfaces(stairs, spacing)
    count = len(samples)
    samples_hash = hashlib.sha256(np.asarray(samples, dtype='<f8').tobytes()).hexdigest()
    cumulative = np.zeros(count, dtype=bool)
    rows = []
    for index, view in enumerate(views):
        _, depth = sequence.read(view)
        # Match the mapper's effective input range, including the strict far cutoff.
        depth = np.array(depth, copy=True)
        depth[(depth <= 0) | (depth >= sequence.camera['depth_trunc_m']) | ~np.isfinite(depth)] = 0
        visible, _, _, _ = _visible_samples(samples, sequence, view, depth, tolerance)
        newly_observed = visible & ~cumulative
        cumulative |= visible
        row = {
            'id': view['id'], 'timestamp_s': view['timestamp_s'], 'frames': index+1,
            'sample_count': count,
            'observed_sample_count': int(visible.sum()),
            'observed_sample_fraction': float(visible.mean()),
            'newly_observed_sample_count': int(newly_observed.sum()),
            'newly_observed_sample_fraction': float(newly_observed.mean()),
            'cumulative_observed_sample_count': int(cumulative.sum()),
            'cumulative_observed_sample_fraction': float(cumulative.mean()),
            'never_observed_sample_count': int((~cumulative).sum()),
            'never_observed_sample_fraction': float((~cumulative).mean()),
            'observed_sample_indices_sha256': _indices_hash(visible),
            'cumulative_observed_sample_indices_sha256': _indices_hash(cumulative),
        }
        if mesh_paths is not None:
            distances = _cohort_distances(mesh_paths[view['id']], samples)
            represented = distances <= tolerance
            represented_observed = represented & cumulative
            represented_unseen = represented & ~cumulative
            row.update(
                represented_sample_count=int(represented.sum()),
                represented_sample_fraction=float(represented.mean()),
                represented_observed_sample_count=int(represented_observed.sum()),
                represented_observed_sample_fraction=(float(represented_observed.sum()/cumulative.sum())
                                                     if cumulative.any() else None),
                represented_never_observed_sample_count=int(represented_unseen.sum()),
                represented_never_observed_sample_fraction=(float(represented_unseen.sum()/(~cumulative).sum())
                                                           if (~cumulative).any() else None),
            )
        rows.append(row)
    return {
        'format': 'wtumi-view-visibility-v1', 'sample_count': count,
        'samples_sha256': samples_hash, 'sample_coordinate_unit': 'metre',
        'visibility_depth_tolerance_m': tolerance, 'surface_distance_tolerance_m': tolerance,
        'sample_spacing_max_m': spacing, 'frames': rows,
        'final': {**rows[-1], 'cumulative_observed_sample_indices': _indices(cumulative),
                  'never_observed_sample_indices': _indices(~cumulative)},
        'ground_truth_used_for_fusion': False, 'scope': VISIBILITY_SCOPE,
        'representation_scope': ('Nearest reconstructed triangle within the configured tolerance; '
                                 'not direct observation, mesh precision or collision safety.'),
    }


def compare_view_visibility(baseline, candidate):
    """Compare final observed sets on the exact same ordered analytic samples.

    Gained/lost are relative to the supplied baseline and describe sampled
    visibility, not guaranteed reachable sensor poses or a mapping algorithm
    improvement. The caller controls observation-budget comparability.
    """
    if (baseline['sample_count'] != candidate['sample_count'] or
            baseline['samples_sha256'] != candidate['samples_sha256'] or
            baseline['visibility_depth_tolerance_m'] != candidate['visibility_depth_tolerance_m']):
        raise ValueError('visibility comparisons require identical ordered samples and depth tolerance')
    count = baseline['sample_count']
    sets = []
    for report in (baseline, candidate):
        values = report['final']['cumulative_observed_sample_indices']
        if (not isinstance(values, list) or any(type(v) is not int or not 0 <= v < count for v in values)
                or len(set(values)) != len(values)
                or len(values) != report['final']['cumulative_observed_sample_count']):
            raise ValueError('invalid cumulative observed sample indices')
        sets.append(set(values))
    a, b = sets
    groups = {'gained': b-a, 'lost': a-b, 'observed_in_both': a & b,
              'never_observed_in_either': set(range(count))-(a | b)}
    result = {'sample_count': count, 'samples_sha256': baseline['samples_sha256'],
              'baseline_observed_sample_count': len(a), 'candidate_observed_sample_count': len(b),
              'net_observed_sample_count_change': len(b)-len(a),
              'visibility_depth_tolerance_m': baseline['visibility_depth_tolerance_m'],
              'scope': 'Final sampled visibility sets on the same ordered stair samples. '+VISIBILITY_SCOPE}
    for name, values in groups.items():
        result[name+'_sample_count'] = len(values)
        result[name+'_sample_fraction'] = len(values)/count
        result[name+'_sample_indices'] = sorted(values)
    return result
