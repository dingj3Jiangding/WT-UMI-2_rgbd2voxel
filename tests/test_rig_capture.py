"""Input checks and opt-in rendering checks for the attached RGB-D rig."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.mujoco_capture import opencv_pose_from_mujoco
from wtumi_recon.render import look_at_pose
from wtumi_recon.rig_capture import capture_rig, load_rig_capture_config
from wtumi_recon.sequence import RGBDSequence


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/capture/stairs_rig.yaml'
RENDER = pytest.mark.skipif(
    os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
    reason='Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL',
)


def _config(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    path = tmp_path / 'rig.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


def _rig_pose(view):
    angle = np.deg2rad(view['yaw_degrees'])
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = view['position_m']
    return pose


def test_default_config_resolves_camera_motion_and_stair_extent():
    config = load_rig_capture_config(DEFAULT_CONFIG)
    assert len(config['trajectory']) == 5
    assert len(config['held_out']) == 2
    assert config['stairs']['steps'] == 4
    assert len(config['box']['size_m']) == 3
    assert all(value > 0 for value in config['box']['size_m'])
    poses = [_rig_pose(view) for view in config['trajectory']]
    assert any(not np.allclose(pose, poses[0]) for pose in poses[1:])
    stairs = config['stairs']
    upper = np.array(stairs['origin_m']) + [stairs['width_m']/2,
             stairs['steps']*stairs['tread_m'], stairs['steps']*stairs['rise_m']]
    assert np.all(upper <= config['roi']['max_m'])


@pytest.mark.parametrize('section,key,value', [
    ('camera', 'width', True), ('camera', 'height', 0),
    ('camera', 'fovy_degrees', 180), ('camera', 'depth_trunc_m', float('nan')),
    ('camera', 'position_in_rig_m', [0, 1]),
    ('box', 'size_m', [0.7, -0.6, 0.6]),
    ('torso', 'size_m', [0.4, 0.3, 0]),
    ('stairs', 'steps', True), ('stairs', 'rise_m', -0.1),
    ('roi', 'min_m', [1, 3, 2]),
])
def test_invalid_geometry_is_rejected_before_output_creation(tmp_path, section, key, value):
    path, config = _config(tmp_path)
    config[section][key] = value
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / 'not_created'
    with pytest.raises(ValueError):
        capture_rig(path, out)
    assert not out.exists()


@pytest.mark.parametrize('invalid', ['duplicate_id', 'unsafe_id', 'timestamp', 'yaw',
                                    'same_camera_target', 'held_out_overlap', 'unknown_field'])
def test_invalid_frame_definitions_are_rejected(tmp_path, invalid):
    path, config = _config(tmp_path)
    if invalid == 'duplicate_id':
        config['held_out'][0]['id'] = config['trajectory'][0]['id']
    elif invalid == 'unsafe_id':
        config['trajectory'][0]['id'] = '../escape'
    elif invalid == 'timestamp':
        config['trajectory'][1]['timestamp_s'] = -1
    elif invalid == 'yaw':
        config['trajectory'][0]['yaw_degrees'] = float('inf')
    elif invalid == 'same_camera_target':
        config['camera']['target_in_rig_m'] = config['camera']['position_in_rig_m'][:]
    elif invalid == 'held_out_overlap':
        base = np.array(config['trajectory'][0]['position_m'])
        config['held_out'][0]['eye'] = (base + config['camera']['position_in_rig_m']).tolist()
        config['held_out'][0]['target'] = (base + config['camera']['target_in_rig_m']).tolist()
    else:
        config['unsupported_field'] = 'ignored settings would be ambiguous'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_rig_capture_config(path)


def test_capture_refuses_nonempty_output_before_rendering(tmp_path):
    path, _ = _config(tmp_path)
    output = tmp_path / 'keep'
    output.mkdir()
    (output / 'existing.txt').write_text('preserve this')
    with pytest.raises(FileExistsError):
        capture_rig(path, output)
    assert (output / 'existing.txt').read_text() == 'preserve this'


@pytest.fixture
def rendered_capture(tmp_path):
    pytest.importorskip('mujoco')
    path, config = _config(tmp_path)
    output = tmp_path / 'capture'
    report = capture_rig(path, output)
    return output, report, config


def _read_mask(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


@RENDER
def test_actual_camera_and_payload_are_attached_to_moving_mocap_body(rendered_capture):
    import mujoco

    output, report, config = rendered_capture
    sequence = RGBDSequence(output / report['paths']['occluded_manifest'])
    model = mujoco.MjModel.from_xml_path(str(output / 'model_source.xml'))
    data = mujoco.MjData(model)
    camera_ids = np.flatnonzero(model.cam_bodyid != 0)
    assert len(camera_ids) == 1
    camera_id = int(camera_ids[0])
    body_id = int(model.cam_bodyid[camera_id])
    mocap_id = int(model.body_mocapid[body_id])
    assert mocap_id >= 0
    attached_geoms = np.flatnonzero(model.geom_bodyid == body_id)
    assert len(attached_geoms) >= 2
    local_camera = look_at_pose(config['camera']['position_in_rig_m'],
                                config['camera']['target_in_rig_m'], [0, 0, 1])
    actual_camera_positions, actual_box_positions = [], []
    for source, saved in zip(config['trajectory'], sequence.views):
        expected_rig = _rig_pose(source)
        angle = np.deg2rad(source['yaw_degrees']) / 2
        data.mocap_pos[mocap_id] = source['position_m']
        data.mocap_quat[mocap_id] = [np.cos(angle), 0, 0, np.sin(angle)]
        mujoco.mj_forward(model, data)
        actual_camera = opencv_pose_from_mujoco(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
        np.testing.assert_allclose(actual_camera, expected_rig @ local_camera, atol=1e-8)
        np.testing.assert_allclose(saved['T_wc'], actual_camera, atol=1e-8)
        actual_camera_positions.append(actual_camera[:3, 3])
        for kind in ('box', 'torso'):
            local_position = np.asarray(config[kind]['position_in_rig_m'])
            matched = [i for i in attached_geoms if np.allclose(model.geom_pos[i], local_position)]
            assert len(matched) == 1
            actual = data.geom_xpos[matched[0]]
            expected = expected_rig[:3, :3] @ local_position + expected_rig[:3, 3]
            np.testing.assert_allclose(actual, expected, atol=1e-8)
            if kind == 'box':
                actual_box_positions.append(actual.copy())
    assert np.linalg.norm(actual_camera_positions[-1] - actual_camera_positions[0]) > .1
    assert np.linalg.norm(actual_box_positions[-1] - actual_box_positions[0]) > .1


def _expected_box_depth(camera, box, intrinsic):
    local_pose = look_at_pose(camera['position_in_rig_m'], camera['target_in_rig_m'], [0, 0, 1])
    u, v = np.meshgrid(np.arange(camera['width']), np.arange(camera['height']))
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsic).T
    directions = rays @ local_pose[:3, :3].T
    center, half = np.asarray(box['position_in_rig_m']), np.asarray(box['size_m'])/2
    with np.errstate(divide='ignore', invalid='ignore'):
        a = (center - half - local_pose[:3, 3]) / directions
        b = (center + half - local_pose[:3, 3]) / directions
    near = np.min(np.stack((a, b)), axis=0).max(axis=-1)
    far = np.max(np.stack((a, b)), axis=0).min(axis=-1)
    return near, far >= np.maximum(near, 0)


@RENDER
def test_paired_capture_preserves_raw_alignment_and_never_fills_excluded_pixels(rendered_capture):
    output, report, config = rendered_capture
    clear = RGBDSequence(output / report['paths']['unoccluded_manifest'])
    blocked = RGBDSequence(output / report['paths']['occluded_manifest'])
    assert clear.camera == blocked.camera
    assert clear.views == blocked.views
    assert clear.metadata['world_up'] == blocked.metadata['world_up'] == [0, 0, 1]
    assert report['integration_frame_ids'] == [v['id'] for v in config['trajectory']]
    assert report['held_out_frame_ids'] == [v['id'] for v in config['held_out']]
    assert json.loads((output / 'capture.json').read_text()) == report
    expected_box_depth, expected_box_hit = _expected_box_depth(
        config['camera'], config['box'], np.asarray(clear.camera['K']))
    changed_observations = 0
    for view in clear.views:
        name = view['id']
        if view['split'] == 'held_out':
            for a, b in zip(clear.read(view), blocked.read(view)):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(clear.read_reference(view), blocked.read_reference(view)):
                np.testing.assert_array_equal(a, b)
            stairs = _read_mask(output / report['paths']['held_out_masks'][name])
            assert stairs.shape == (config['camera']['height'], config['camera']['width'])
            assert stairs.any()
            assert np.all(clear.read(view)[1][stairs] > 0)
            continue
        for condition, sequence in [('unoccluded', clear), ('occluded', blocked)]:
            rgb, depth = sequence.read(view)
            frame = sequence.root / 'frames' / name
            with Image.open(frame / 'raw_rgb.png') as image:
                raw_rgb = np.asarray(image)
            raw_depth = np.load(frame / 'raw_depth_m.npy', allow_pickle=False)
            excluded = _read_mask(frame / 'excluded_mask.png')
            box = _read_mask(frame / 'box_mask.png')
            robot = _read_mask(frame / 'robot_mask.png')
            np.testing.assert_array_equal(excluded, box | robot)
            assert np.isfinite(raw_depth).all() and np.all(raw_depth >= 0)
            np.testing.assert_array_equal(raw_rgb[raw_depth == 0], 0)
            np.testing.assert_array_equal(rgb[excluded], 0)
            np.testing.assert_array_equal(depth[excluded], 0)
            np.testing.assert_array_equal(rgb[~excluded], raw_rgb[~excluded])
            np.testing.assert_array_equal(depth[~excluded], raw_depth[~excluded])
            if condition == 'unoccluded':
                assert not box.any()
            else:
                assert box.any() and np.all(raw_depth[box] > 0)
                assert np.all(expected_box_hit[box])
                # Segmentation, RGB and metric depth must share the same rays.
                # Exact silhouette pixels may differ by rasterization rounding.
                np.testing.assert_allclose(raw_depth[box], expected_box_depth[box], atol=3e-4, rtol=0)
                clear_depth = clear.read(view)[1]
                changed_observations += np.count_nonzero(box & (clear_depth > 0) & (depth == 0))
    assert changed_observations > 0
    with Image.open(output / report['paths']['preview_gif']) as animation:
        assert animation.format == 'GIF' and animation.n_frames >= 2
