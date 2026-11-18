"""Masked-observation visibility, prefix accumulation and shared sample sets."""

import copy
from pathlib import Path

import numpy as np
import pytest

from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.view_metrics import evaluate_view_visibility, compare_view_visibility


STAIRS = {'origin_m': [0, 1, 0], 'width_m': 1, 'tread_m': .3, 'rise_m': .2, 'steps': 2}
SAMPLES = np.array([[0., 0., 2.], [.5, 0., 2.], [-.5, 0., 2.],
                    [0., 0., -1.], [20., 0., 2.], [0., .5, 4.99]])


def _sequence(tmp_path, depths=None):
    camera = {'width': 5, 'height': 5, 'K': [[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]],
              'depth_trunc_m': 5.}
    writer = SequenceWriter(tmp_path/'sequence', camera, 'stairs', 'simulator_ground_truth')
    if depths is None:
        depths = [np.zeros((5, 5), np.float32) for _ in range(3)]
        depths[0][2, 1] = 2.  # Left sample only.
        depths[1][2, 2] = 2.  # Center sample newly seen; left currently absent.
        depths[2][2, 1] = 2.  # Repeated left view contributes no new sample.
        depths[2][2, 3] = 2.  # Right sample newly seen.
        depths[2][2, 2] = 5.  # At the far cutoff: no usable observation.
    for i, depth in enumerate(depths):
        writer.add_frame(f'frame_{i}', 'integration', np.zeros((5, 5, 3), np.uint8),
                         depth, np.eye(4), float(i),
                         reference_rgb=np.full((5, 5, 3), 255, np.uint8),
                         reference_depth_m=np.full((5, 5), 2., np.float32))
    held_pose = np.eye(4)
    held_pose[0, 3] = .1
    writer.add_frame('novel', 'held_out', np.zeros((5, 5, 3), np.uint8),
                     np.ones((5, 5), np.float32), held_pose, float(len(depths)))
    return writer.finish()


def _mock_samples(monkeypatch):
    from wtumi_recon import view_metrics
    monkeypatch.setattr(view_metrics, 'sample_stair_surfaces', lambda *a: SAMPLES.copy())


def test_masked_observations_only_and_monotonic_prefix_sets(tmp_path, monkeypatch):
    _mock_samples(monkeypatch)
    manifest = _sequence(tmp_path)
    actual_read = RGBDSequence.read
    reads = []
    def integration_only(self, view):
        assert view['split'] == 'integration'
        reads.append(view['id'])
        return actual_read(self, view)
    monkeypatch.setattr(RGBDSequence, 'read', integration_only)
    monkeypatch.setattr(RGBDSequence, 'read_reference', lambda *a: pytest.fail('reference data read'))
    # Existing references and held-outs can be unreadable: only paths are validated.
    for path in manifest.parent.rglob('reference_depth_m.npy'):
        path.write_bytes(b'not a depth array')
    (manifest.parent/'frames/novel/depth_m.npy').write_bytes(b'not a depth array')
    result = evaluate_view_visibility(manifest, STAIRS)
    assert reads == ['frame_0', 'frame_1', 'frame_2']
    rows = result['frames']
    assert [r['observed_sample_count'] for r in rows] == [1, 1, 2]
    assert [r['newly_observed_sample_count'] for r in rows] == [1, 1, 1]
    assert [r['cumulative_observed_sample_count'] for r in rows] == [1, 2, 3]
    assert [r['never_observed_sample_count'] for r in rows] == [5, 4, 3]
    assert result['final']['cumulative_observed_sample_indices'] == [0, 1, 2]
    assert result['final']['never_observed_sample_indices'] == [3, 4, 5]
    assert result['ground_truth_used_for_fusion'] is False
    assert result['final']['cumulative_observed_sample_fraction'] == .5


def test_far_cutoff_and_zero_depth_cannot_mark_a_sample_seen(tmp_path, monkeypatch):
    _mock_samples(monkeypatch)
    depth = np.full((5, 5), 5., np.float32)
    manifest = _sequence(tmp_path, [depth, np.zeros((5, 5), np.float32)])
    result = evaluate_view_visibility(manifest, STAIRS)
    assert result['final']['cumulative_observed_sample_count'] == 0
    assert result['final']['never_observed_sample_count'] == len(SAMPLES)
    assert all(row['newly_observed_sample_count'] == 0 for row in result['frames'])


def _plane(path, z):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def test_mesh_representation_is_separate_from_actual_observation(tmp_path, monkeypatch):
    _mock_samples(monkeypatch)
    manifest = _sequence(tmp_path)
    near, far = _plane(tmp_path/'near.ply', 2.), _plane(tmp_path/'far.ply', 3.)
    result = evaluate_view_visibility(manifest, STAIRS,
                                     {'frame_0': near, 'frame_1': near, 'frame_2': far})
    first, middle, final = result['frames']
    assert first['represented_sample_count'] == 3
    assert first['represented_observed_sample_count'] == 1
    assert first['represented_never_observed_sample_count'] == 2
    assert middle['represented_observed_sample_count'] == 2
    assert final['represented_sample_count'] == 0
    assert final['cumulative_observed_sample_count'] == 3


def test_comparison_reports_gains_losses_and_common_unseen(tmp_path, monkeypatch):
    _mock_samples(monkeypatch)
    manifest = _sequence(tmp_path)
    a = evaluate_view_visibility(manifest, STAIRS)
    b = copy.deepcopy(a)
    b['final']['cumulative_observed_sample_indices'] = [0, 2, 3, 4]
    b['final']['cumulative_observed_sample_count'] = 4
    result = compare_view_visibility(a, b)
    assert result['gained_sample_indices'] == [3, 4]
    assert result['lost_sample_indices'] == [1]
    assert result['observed_in_both_sample_indices'] == [0, 2]
    assert result['never_observed_in_either_sample_indices'] == [5]
    assert result['net_observed_sample_count_change'] == 1
    assert sum(result[k+'_sample_count'] for k in
               ('gained', 'lost', 'observed_in_both', 'never_observed_in_either')) == len(SAMPLES)


@pytest.mark.parametrize('field,value', [('samples_sha256', 'different'),
                                        ('sample_count', 99),
                                        ('visibility_depth_tolerance_m', .1)])
def test_comparison_rejects_different_sample_definitions(tmp_path, monkeypatch, field, value):
    _mock_samples(monkeypatch)
    a = evaluate_view_visibility(_sequence(tmp_path), STAIRS)
    b = copy.deepcopy(a)
    b[field] = value
    with pytest.raises(ValueError, match='identical ordered samples'):
        compare_view_visibility(a, b)


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf')])
def test_rejects_invalid_tolerance_before_sequence_read(value):
    with pytest.raises(ValueError, match='tolerance_m'):
        evaluate_view_visibility('missing_manifest.json', STAIRS, tolerance_m=value)


def test_missing_or_wrong_prefix_meshes_are_rejected(tmp_path):
    manifest = _sequence(tmp_path)
    with pytest.raises(ValueError, match='one prefix mesh'):
        evaluate_view_visibility(manifest, STAIRS, {'frame_0': 'missing.ply'})
    with pytest.raises(FileNotFoundError):
        evaluate_view_visibility(manifest, STAIRS, {f'frame_{i}': 'missing.ply' for i in range(3)})
