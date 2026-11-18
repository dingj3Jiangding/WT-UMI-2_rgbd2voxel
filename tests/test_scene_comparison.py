"""A camera observation must be paired with only its accumulated map evidence."""

import hashlib
import json
import os

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.scene_bundle import load_scene_config
from wtumi_recon.scene_comparison import (
    comparison_inputs, export_camera_voxel_comparison, make_voxel_renderer,
    overview_camera, render_voxel_panel,
)
from wtumi_recon.scene_viewer import load_display_voxels


RENDER = pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                            reason='Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL')


@pytest.fixture
def saved_comparison(tmp_path):
    cfg = {key: value for key, value in load_scene_config('stairs').items() if not key.startswith('_')}
    cfg['trajectory'] = cfg['trajectory'][:2]
    cfg['held_out'] = cfg['held_out'][:1]
    cfg['playback']['fps'] = 3
    (tmp_path / 'scene').mkdir()
    (tmp_path / 'scene/scene.yaml').write_text(yaml.safe_dump(cfg))
    (tmp_path / 'capture').mkdir()
    rows, history = [], []
    for index, color in enumerate(([220, 30, 40], [30, 80, 220])):
        name = cfg['trajectory'][index]['id']
        directory = tmp_path / 'capture/frames' / name
        directory.mkdir(parents=True)
        Image.new('RGB', (80, 60), tuple(color)).save(directory / 'raw_rgb.png')
        transform = np.eye(4)
        transform[:3, 3] = [index, 0, 1.]
        rows.append({'id': name, 'timestamp_s': cfg['trajectory'][index]['timestamp_s'],
                     'frame_directory': f'frames/{name}', 'T_wc': transform.tolist()})
        prefix = tmp_path / 'history' / f'frame_{index + 1:03d}'
        prefix.mkdir(parents=True)
        (prefix / 'metrics.json').write_text(json.dumps({'integrated_frame_ids': [r['id'] for r in rows]}))
        weight = np.zeros(64, dtype=np.float32)
        weight[:2 + index * 4] = 1
        sdf = np.ones(64, dtype=np.float32)
        sdf[:2 + index * 4] = 0
        sdf[-1] = 0  # Unobserved zero crossings must not become visible evidence.
        np.savez_compressed(prefix / 'tsdf_volume.npz', resolution=4, voxel_size_m=.2,
                            sdf_trunc_m=.4, origin_m=[0., 1., 0.], tsdf=sdf, weight=weight,
                            color=np.tile([90., 150., 210.], (64, 1)), color_scale=255.,
                            axis_order='xyz', flat_order='C', tsdf_normalized=True)
        history.append({'id': name, 'frames': index + 1,
                        'timestamp_s': cfg['trajectory'][index]['timestamp_s'],
                        'run_dir': prefix.relative_to(tmp_path).as_posix()})
    (tmp_path / 'capture/capture.json').write_text(json.dumps({
        'frames': rows, 'integration_frame_ids': [r['id'] for r in rows],
        'held_out_frame_ids': [cfg['held_out'][0]['id']]}))
    (tmp_path / 'history.json').write_text(json.dumps(history))
    return tmp_path


def test_comparison_pairs_each_raw_image_with_its_own_prefix(saved_comparison):
    root, cfg, rows = comparison_inputs(saved_comparison)
    assert [row['frames'] for row in rows] == [1, 2]
    assert [row['volume'].parent.name for row in rows] == ['frame_001', 'frame_002']
    first = load_display_voxels(rows[0]['volume'], max_voxels=3)
    second = load_display_voxels(rows[1]['volume'], max_voxels=3)
    assert (len(first.centers), first.selected_count, first.observed_count) == (2, 2, 2)
    assert (len(second.centers), second.selected_count, second.observed_count) == (3, 6, 6)
    np.testing.assert_array_equal(rows[0]['T_wc'][:3, 3], [0, 0, 1])
    np.testing.assert_array_equal(rows[1]['T_wc'][:3, 3], [1, 0, 1])


@pytest.mark.parametrize('case', ['future_map', 'wrong_frame', 'wrong_time', 'missing_prefix', 'path_escape'])
def test_mismatched_comparison_inputs_fail_before_rendering(saved_comparison, case):
    root = saved_comparison
    history = json.loads((root / 'history.json').read_text())
    if case == 'future_map':
        path = root / 'history/frame_001/metrics.json'
        value = json.loads(path.read_text())
        value['integrated_frame_ids'].append(history[1]['id'])
        path.write_text(json.dumps(value))
    else:
        if case == 'wrong_frame':
            history[0]['id'] = history[1]['id']
        elif case == 'wrong_time':
            history[0]['timestamp_s'] += 1
        elif case == 'missing_prefix':
            history.pop()
        else:
            history[0]['run_dir'] = '../outside'
        (root / 'history.json').write_text(json.dumps(history))
    with pytest.raises((ValueError, FileNotFoundError)):
        comparison_inputs(root)
    assert not (root / 'camera_voxel_frames').exists()


def test_overview_camera_does_not_depend_on_future_tsdf_geometry():
    config = load_scene_config('room')
    camera = overview_camera(config)
    assert np.isfinite(camera['lookat']).all() and camera['distance_m'] > 0
    # The framing helper only reads explicit layout bounds and the path; it
    # accepts no saved map and therefore cannot draw later observations.
    assert 'source geometry is not rendered' in camera['framing_source']


@pytest.mark.parametrize('error', [ImportError('video extra is absent'), RuntimeError('encoder is unavailable')])
def test_optional_video_failure_preserves_gif_and_png_exports(saved_comparison, monkeypatch, error):
    import wtumi_recon.scene_comparison as comparison
    import wtumi_recon.video as video

    def unavailable():
        raise error

    class Renderer:
        def close(self):
            pass

    monkeypatch.setattr(video, 'require_video_encoder', unavailable)
    monkeypatch.setattr(comparison, 'make_voxel_renderer',
                        lambda *args: (Renderer(), None, None, None, {}))
    monkeypatch.setattr(comparison, 'render_voxel_panel',
                        lambda *args: Image.new('RGB', (160, 120), 'black'))
    report = export_camera_voxel_comparison(saved_comparison, panel_width=160, panel_height=120)
    assert report['video'] == {'status': 'unavailable', 'reason': str(error)}
    assert 'mp4' not in report['paths']
    assert not (saved_comparison / 'camera_voxel.mp4').exists()
    assert all((saved_comparison / value).exists() for value in report['paths'].values())


def test_pillow_10_default_font_fallback(monkeypatch):
    import wtumi_recon.scene_comparison as comparison

    original = comparison.ImageFont.load_default
    calls = []

    def legacy_font(**kwargs):
        calls.append(kwargs)
        if kwargs:
            raise TypeError('unexpected size')
        return original()

    monkeypatch.setattr(comparison.ImageFont, 'load_default', legacy_font)
    assert comparison._font(19) is not None
    assert calls == [{'size': 19}, {}]


@RENDER
def test_renderer_uses_empty_world_and_only_actual_voxel_cells(saved_comparison, monkeypatch):
    import mujoco

    root, config, rows = comparison_inputs(saved_comparison)
    monkeypatch.setattr(mujoco, 'mj_step', lambda *args: pytest.fail('comparison must not advance physics'))
    renderer, model, data, camera, _ = make_voxel_renderer(config, 160, 120, 3)
    try:
        assert model.ngeom == 0 and model.nmesh == 0
        cells = load_display_voxels(rows[1]['volume'], max_voxels=3)
        panel = render_voxel_panel(renderer, model, data, camera, cells, rows[1]['T_wc'])
        assert panel.size == (160, 120)
        assert renderer.scene.ngeom == 5  # Three actual cells and two labeled camera markers.
        for index, center in enumerate(cells.centers):
            np.testing.assert_allclose(renderer.scene.geoms[index].pos, center)
            assert renderer.scene.geoms[index].type == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_allclose(renderer.scene.geoms[3].pos, rows[1]['T_wc'][:3, 3])
    finally:
        renderer.close()


@RENDER
def test_export_is_portable_and_preserves_saved_maps_and_observations(saved_comparison):
    root = saved_comparison
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    result = export_camera_voxel_comparison(root, panel_width=160, panel_height=120,
                                            max_voxels=3, export_mp4=False)
    assert result['frame_count'] == 2 and result['playback_fps'] == 3
    assert result['video']['status'] == 'disabled'
    assert result['gif_frame_indices'] == [0, 1]
    assert [row['selected_surface_cells'] for row in result['frames']] == [2, 6]
    assert [row['displayed_cells'] for row in result['frames']] == [2, 3]
    assert result['frames'][0]['tsdf_volume'] != result['frames'][1]['tsdf_volume']
    for row, color in zip(result['frames'], ([220, 30, 40], [30, 80, 220])):
        with Image.open(root / row['image']) as panel:
            np.testing.assert_array_equal(np.asarray(panel)[76 + 60, 80], color)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original.items())
    moved = root.parent / (root.name + '_moved')
    root.rename(moved)
    saved = json.loads((moved / 'camera_voxel.json').read_text())
    assert all((moved / value).exists() for value in saved['paths'].values())
    for row in saved['frames']:
        assert (moved / row['image']).is_file()
        assert (moved / row['raw_rgb']).is_file()
        assert (moved / row['tsdf_volume']).is_file()
    with pytest.raises(FileExistsError, match='already exist'):
        export_camera_voxel_comparison(moved, export_mp4=False)
