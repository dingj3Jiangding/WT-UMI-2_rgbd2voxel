"""Incremental snapshots preserve batch geometry without repeatedly fusing inputs."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.occlusion_demo import write_prefix_manifest
from wtumi_recon.pipeline import ReconstructionState, run_reconstruction
from wtumi_recon.sequence import SequenceWriter
from wtumi_recon.synthetic import SyntheticScene
from wtumi_recon.tsdf import Open3DTSDFMap


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path):
    raw = yaml.safe_load((ROOT / 'configs/scenes/multiview_cube.yaml').read_text())
    raw['camera'].update(width=64, height=48, fx=60., fy=60., cx=31.5, cy=23.5)
    raw['reconstruction']['resolution'] = 48
    scene_path = tmp_path / 'scene.yaml'
    scene_path.write_text(yaml.safe_dump(raw))
    cfg = load_run_config(scene_path)
    scene = SyntheticScene.from_config(cfg['scene'])
    writer = SequenceWriter(tmp_path / 'capture', cfg['camera'], 'prefix_cube', 'simulator_ground_truth')
    for index, view in enumerate(cfg['views']):
        rgb, depth = scene.render(np.asarray(cfg['camera']['K']), np.asarray(view['T_wc']), 64, 48)
        clean = depth.copy()
        if view['split'] == 'integration':
            depth[::4, ::4] = 0
            rgb[depth == 0] = 0
        writer.add_frame(view['id'], view['split'], rgb, depth, view['T_wc'], float(index),
                         reference_rgb=rgb, reference_depth_m=clean)
    manifest = writer.finish()
    base = {'schema_version': 3,
            'source': {'type': 'rgbd_sequence', 'manifest': str(manifest)},
            'reconstruction': raw['reconstruction'],
            'occupancy': {'enabled': True, 'resolution': 24, 'pixel_stride': 2},
            'evaluation': raw['evaluation']}
    paths = []
    for count in (1, 2, 4):
        data = copy.deepcopy(base)
        data['source']['manifest'] = str(write_prefix_manifest(manifest, count)) if count < 4 else str(manifest)
        path = tmp_path / f'prefix_{count}.yaml'
        path.write_text(yaml.safe_dump(data))
        paths.append(path)
    return paths, manifest


def test_incremental_snapshots_match_batch_and_fuse_each_frame_once(tmp_path, monkeypatch):
    paths, manifest = _fixture(tmp_path)
    baseline = run_reconstruction(paths[-1], tmp_path / 'batch')
    state = ReconstructionState()
    calls = []
    integrate = Open3DTSDFMap.integrate

    def track(self, rgbd, intrinsic, pose):
        calls.append(np.asarray(rgbd.depth).copy())
        return integrate(self, rgbd, intrinsic, pose)

    monkeypatch.setattr(Open3DTSDFMap, 'integrate', track)
    snapshots = []
    for index, config in enumerate(paths):
        out = tmp_path / f'snapshot_{index}'
        report = run_reconstruction(config, out, prefix_state=state, compact=index < 2)
        assert report['incremental_prefix'] is True
        expected_count = (1, 2, 4)[index]
        assert len(calls) == expected_count
        assert len(report['integrated_frame_ids']) == expected_count
        if index < 2:
            assert report['export_scope'] == 'held_out_snapshot'
            assert not (out / 'tsdf_volume.npz').exists()
            assert not (out / 'pointcloud.ply').exists()
            exported = json.loads((out / 'manifest.json').read_text())
            assert {v['split'] for v in exported['frames']} == {'held_out'}
            assert {v['split'] for v in report['views']} == {'held_out'}
            assert (out / 'mesh.ply').is_file()
            assert (out / 'occupancy.npz').is_file()
            snapshots.append({p: p.read_bytes() for p in out.rglob('*') if p.is_file()})
    assert report['summary'] == baseline['summary']
    assert report['mesh_vertices'] == baseline['mesh_vertices']
    assert report['mesh_triangles'] == baseline['mesh_triangles']
    assert len(report['newly_integrated_frame_ids']) == 2
    assert (tmp_path / 'snapshot_2/pointcloud.ply').is_file()
    for filename in ('tsdf_volume.npz', 'occupancy.npz'):
        with np.load(tmp_path / 'batch' / filename) as expected, np.load(tmp_path / 'snapshot_2' / filename) as actual:
            assert set(actual.files) == set(expected.files)
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    for snapshot in snapshots:
        for path, content in snapshot.items():
            assert path.read_bytes() == content
    frames = json.loads(manifest.read_text())['frames']
    fused = [frame for frame in frames if frame['split'] == 'integration']
    for actual, frame in zip(calls, fused):
        np.testing.assert_array_equal(actual, np.load(manifest.parent / frame['depth_m']))
        # Explicit clean references have extra depth and must not enter fusion.
        assert np.count_nonzero(np.load(manifest.parent / frame['reference_depth_m'])) > np.count_nonzero(actual)


@pytest.mark.parametrize('change', ['same_prefix', 'settings', 'pose', 'source_asset'])
def test_invalid_state_reuse_is_rejected_before_output(tmp_path, change):
    paths, manifest = _fixture(tmp_path)
    state = ReconstructionState()
    run_reconstruction(paths[0], tmp_path / 'first', prefix_state=state, compact=True)
    next_config = paths[1]
    if change == 'same_prefix':
        next_config = paths[0]
    elif change == 'settings':
        cfg = yaml.safe_load(next_config.read_text())
        cfg['reconstruction']['resolution'] += 1
        next_config.write_text(yaml.safe_dump(cfg))
    elif change == 'pose':
        cfg = yaml.safe_load(next_config.read_text())
        path = Path(cfg['source']['manifest'])
        saved = json.loads(path.read_text())
        saved['frames'][0]['T_wc'][0][3] += .01
        path.write_text(json.dumps(saved))
    else:
        saved = json.loads(manifest.read_text())
        path = manifest.parent / saved['frames'][0]['depth_m']
        depth = np.load(path)
        depth[depth > 0] += .001
        np.save(path, depth)
    with pytest.raises(ValueError, match='prefix'):
        run_reconstruction(next_config, tmp_path / 'invalid', prefix_state=state, compact=True)
    assert not (tmp_path / 'invalid').exists()


def test_failed_snapshot_state_cannot_be_reused(tmp_path, monkeypatch):
    from wtumi_recon import pipeline

    paths, _ = _fixture(tmp_path)
    state = ReconstructionState()

    def fail(*args, **kwargs):
        raise RuntimeError('simulated export error')

    monkeypatch.setattr(pipeline, 'render_mesh_rgbd', fail)
    with pytest.raises(RuntimeError, match='simulated export'):
        run_reconstruction(paths[0], tmp_path / 'failed', prefix_state=state, compact=True)
    with pytest.raises(ValueError, match='failed/in-progress'):
        run_reconstruction(paths[1], tmp_path / 'retry', prefix_state=state, compact=True)
    assert not (tmp_path / 'retry').exists()


def test_compact_volume_snapshot_matches_full_export(tmp_path):
    paths, _ = _fixture(tmp_path)
    full = run_reconstruction(paths[0], tmp_path / 'full')
    compact = run_reconstruction(paths[0], tmp_path / 'compact', compact=True, snapshot_volume=True)
    assert compact['volume'] == full['volume'] == 'tsdf_volume.npz'
    assert not (tmp_path / 'compact/pointcloud.ply').exists()
    assert {row['split'] for row in compact['views']} == {'held_out'}
    with np.load(tmp_path / 'full/tsdf_volume.npz') as expected, np.load(tmp_path / 'compact/tsdf_volume.npz') as actual:
        assert set(actual.files) == set(expected.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
