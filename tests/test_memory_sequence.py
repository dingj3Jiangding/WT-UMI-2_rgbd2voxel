"""Mixed observation histories, portable replay, and analytic hidden cohorts."""

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.memory_sequence import prepare_memory_sequences, evaluate_memory_retention
from wtumi_recon.sequence import RGBDSequence, SequenceWriter


def _capture(tmp_path):
    root = tmp_path/'capture'
    camera = {'width': 5, 'height': 5, 'K': [[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]],
              'depth_trunc_m': 5.}
    poses = []
    for index in range(4):
        pose = np.eye(4)
        if index == 3:
            pose[0, 3] = .1
        poses.append(pose)
    for condition in ('unoccluded', 'occluded'):
        writer = SequenceWriter(root/condition, camera, 'test_stairs', 'simulator_ground_truth',
                                {'world_up': [0, 0, 1]})
        for index in range(4):
            name, split = (f'frame_{index}', 'integration') if index < 3 else ('novel', 'held_out')
            raw_rgb = np.full((5, 5, 3), 100+index, dtype=np.uint8)
            raw_depth = np.full((5, 5), 2., dtype=np.float32)
            box = np.zeros((5, 5), dtype=bool)
            if condition == 'occluded' and split == 'integration':
                box[2, 2] = True
                raw_rgb[box], raw_depth[box] = [200, 100, 0], .5
            rgb, depth = raw_rgb.copy(), raw_depth.copy()
            rgb[box], depth[box] = 0, 0
            # Explicit hidden integration references must never become inputs.
            ref_rgb = np.full((5, 5, 3), 220, dtype=np.uint8)
            ref_depth = np.full((5, 5), 2.5, dtype=np.float32)
            writer.add_frame(name, split, rgb, depth, poses[index], float(index),
                             reference_rgb=ref_rgb, reference_depth_m=ref_depth)
            if split == 'integration':
                folder = root/condition/'frames'/name
                np.save(folder/'raw_depth_m.npy', raw_depth, allow_pickle=False)
                Image.fromarray(raw_rgb).save(folder/'raw_rgb.png')
                Image.fromarray(box.astype(np.uint8)*255).save(folder/'excluded_mask.png')
                Image.fromarray(box.astype(np.uint8)*255).save(folder/'box_mask.png')
                Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(folder/'robot_mask.png')
        writer.finish()
    report = {'stairs': {'origin_m': [0, 1, 0], 'width_m': 1, 'tread_m': .3,
                         'rise_m': .2, 'steps': 2},
              'paths': {'unoccluded_manifest': 'unoccluded/manifest.json',
                        'occluded_manifest': 'occluded/manifest.json'},
              'metadata': {'test_fixture': True}}
    (root/'capture.json').write_text(json.dumps(report))
    return root


def test_prepared_histories_select_observations_and_remain_portable(tmp_path):
    capture = _capture(tmp_path)
    result = prepare_memory_sequences(capture, tmp_path/'sequences', preobserve_frames=2)
    assert result['preobserve_frames'] == 2
    assert [r['source_conditions']['observe_then_occluded'] for r in result['frame_schedule']] == [
        'unoccluded', 'unoccluded', 'occluded']
    source = {key: RGBDSequence(capture/key/'manifest.json') for key in ('unoccluded', 'occluded')}
    for case, manifest in result['manifests'].items():
        sequence = RGBDSequence(manifest)
        for index, view in enumerate(v for v in sequence.views if v['split'] == 'integration'):
            condition = 'unoccluded' if case == 'observe_then_occluded' and index < 2 else 'occluded'
            expected = source[condition].read(view)
            for actual, wanted in zip(sequence.read(view), expected):
                np.testing.assert_array_equal(actual, wanted)
            # Explicit source clean references (.5 m behind observations) were dropped.
            for actual, wanted in zip(sequence.read_reference(view), expected):
                np.testing.assert_array_equal(actual, wanted)
            folder = sequence.root/'frames'/view['id']
            for name in ('box_mask.png', 'robot_mask.png', 'excluded_mask.png', 'raw_rgb.png', 'raw_depth_m.npy'):
                assert (folder/name).read_bytes() == (source[condition].root/'frames'/view['id']/name).read_bytes()
        for actual, expected in zip(sequence.read_reference('novel'), source['unoccluded'].read_reference('novel')):
            np.testing.assert_array_equal(actual, expected)
        assert sequence.metadata['memory_case'] == case
        assert len(sequence.metadata['memory_schedule']) == 3
    relocated = tmp_path/'relocated'
    shutil.copytree(tmp_path/'sequences', relocated)
    shutil.rmtree(tmp_path/'sequences')
    shutil.rmtree(capture)
    saved = json.loads((relocated/'sequences.json').read_text())
    for case, manifest in saved['manifests'].items():
        sequence = RGBDSequence(relocated/manifest)
        assert sequence.read('frame_0')[1][2, 2] == (2. if case == 'observe_then_occluded' else 0.)


@pytest.mark.parametrize('value', [0, 3, 4, True, 1.5])
def test_invalid_prefix_does_not_create_output(tmp_path, value):
    capture = _capture(tmp_path)
    with pytest.raises(ValueError, match='preobserve_frames'):
        prepare_memory_sequences(capture, tmp_path/'output', value)
    assert not (tmp_path/'output').exists()


def test_missing_box_mask_and_existing_output_are_rejected(tmp_path):
    capture = _capture(tmp_path)
    out = tmp_path/'output'
    out.mkdir()
    (out/'keep.txt').write_text('unchanged')
    with pytest.raises(FileExistsError):
        prepare_memory_sequences(capture, out, 1)
    assert (out/'keep.txt').read_text() == 'unchanged'
    (capture/'occluded/frames/frame_2/box_mask.png').unlink()
    with pytest.raises(FileNotFoundError, match='box_mask'):
        prepare_memory_sequences(capture, tmp_path/'fresh', 1)
    assert not (tmp_path/'fresh').exists()


def test_mask_union_and_paired_reference_mismatch_rejected(tmp_path):
    capture = _capture(tmp_path)
    path = capture/'unoccluded/frames/frame_0/robot_mask.png'
    Image.fromarray(np.full((5, 5), 255, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match='union'):
        prepare_memory_sequences(capture, tmp_path/'fresh', 1)
    assert not (tmp_path/'fresh').exists()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(path)
    path = capture/'occluded/frames/novel/reference_depth_m.npy'
    np.save(path, np.full((5, 5), 3., dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match='held-out references'):
        prepare_memory_sequences(capture, tmp_path/'fresh', 1)
    assert not (tmp_path/'fresh').exists()


def _plane(path, z):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def test_common_hidden_cohort_has_known_distances_and_no_reference_reads(tmp_path, monkeypatch):
    from wtumi_recon import memory_sequence
    capture = _capture(tmp_path)
    prepared = prepare_memory_sequences(capture, tmp_path/'sequences', 1)
    # Center sample maps to the box pixel; side samples remain visible.
    samples = np.array([[0., 0., 2.], [.5, 0., 2.], [-.5, 0., 2.], [0., 0., -1.], [20., 0., 2.]])
    monkeypatch.setattr(memory_sequence, 'sample_stair_surfaces', lambda *args: samples)
    near = _plane(tmp_path/'near.ply', 2.)
    far = _plane(tmp_path/'far.ply', 3.)
    meshes = {'frame_0': near, 'frame_1': near, 'frame_2': far}
    # File contents of held-outs and explicit clean references are unnecessary.
    for path in capture.rglob('reference_depth_m.npy'):
        path.write_bytes(b'not loaded by evaluation')
    for path in capture.rglob('frames/novel/depth_m.npy'):
        path.write_bytes(b'not loaded by evaluation')
    for path in (tmp_path/'sequences').rglob('frames/novel/reference_depth_m.npy'):
        path.write_bytes(b'not loaded by evaluation')
    results = {case: evaluate_memory_retention(capture, manifest, meshes, 1)
               for case, manifest in prepared['manifests'].items()}
    a, b = results['always_occluded'], results['observe_then_occluded']
    assert [row['cohort_sample_count'] for row in a] == [0, 1, 1]
    assert [row['reconstructed_cohort_fraction'] for row in a] == [None, 1., 0.]
    assert a[2]['mean_nearest_mesh_distance_m'] == pytest.approx(1.)
    assert [row['cohort_sample_indices_sha256'] for row in a] == [row['cohort_sample_indices_sha256'] for row in b]
    assert a[0]['preobserved_sample_count'] == 3
    assert a[-1]['actually_observed_sample_count'] == 2
    assert b[-1]['actually_observed_sample_count'] == 3
    assert all(row['ground_truth_used_for_fusion'] is False for row in a+b)


def test_memory_cohort_rejects_box_behind_surface(tmp_path, monkeypatch):
    from wtumi_recon import memory_sequence
    capture = _capture(tmp_path)
    prepared = prepare_memory_sequences(capture, tmp_path/'sequences', 1)
    monkeypatch.setattr(memory_sequence, 'sample_stair_surfaces', lambda *args: np.array([[0., 0., 2.]]))
    mesh = _plane(tmp_path/'plane.ply', 2.)
    for index in (1, 2):
        path = capture/f'occluded/frames/frame_{index}/raw_depth_m.npy'
        raw = np.load(path)
        raw[2, 2] = 2.5
        np.save(path, raw, allow_pickle=False)
    rows = evaluate_memory_retention(capture, prepared['manifests']['always_occluded'],
                                     {f'frame_{index}': mesh for index in range(3)}, 1)
    assert all(row['cohort_sample_count'] == 0 for row in rows)
    assert all(row['reconstructed_cohort_fraction'] is None for row in rows)


def test_memory_evaluation_rejects_missing_prefix_and_pose_mismatch(tmp_path):
    capture = _capture(tmp_path)
    prepared = prepare_memory_sequences(capture, tmp_path/'sequences', 1)
    manifest = Path(prepared['manifests']['always_occluded'])
    mesh = _plane(tmp_path/'plane.ply', 2.)
    with pytest.raises(ValueError, match='one prefix mesh'):
        evaluate_memory_retention(capture, manifest, {'frame_0': mesh}, 1)
    data = json.loads(manifest.read_text())
    data['frames'][0]['T_wc'][0][3] = .2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='poses, timestamps and IDs'):
        evaluate_memory_retention(capture, manifest, {f'frame_{i}': mesh for i in range(3)}, 1)
