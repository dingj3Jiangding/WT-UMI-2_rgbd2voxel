"""Independent camera rotation preserves the rigid payload and observation masks."""

import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.mujoco_capture import opencv_pose_from_mujoco
from wtumi_recon.render import look_at_pose
from wtumi_recon.rig_capture import (
    _set_local_head_pose, build_rig_mjcf, capture_rig, head_camera_pose,
    load_rig_capture_config, rig_transform,
)
from wtumi_recon.sequence import RGBDSequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'configs/capture/stairs_rig.yaml'
RENDER = pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                           reason='Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL')


def _write(tmp_path, cfg):
    path = tmp_path/'head_capture.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_zero_angles_preserve_calibration_and_signed_axes():
    cfg = load_rig_capture_config(DEFAULT)
    camera = cfg['camera']
    nominal = look_at_pose(camera['position_in_rig_m'], camera['target_in_rig_m'], [0, 0, 1])
    np.testing.assert_array_equal(head_camera_pose(camera, {}), nominal)
    left = head_camera_pose(camera, {'head_yaw_degrees': 30})
    raised = head_camera_pose(camera, {'head_pitch_degrees': 15})
    assert left[0, 2] < nominal[0, 2]  # +Y forward turns toward -X.
    assert raised[2, 2] > nominal[2, 2]
    for pose in (left, raised):
        np.testing.assert_array_equal(pose[:3, 3], nominal[:3, 3])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    assert all(frame['head_yaw_degrees'] == frame['head_pitch_degrees'] == 0
               for frame in cfg['trajectory'])


@pytest.mark.parametrize('field,value', [
    ('head_yaw_degrees', True), ('head_pitch_degrees', False),
    ('head_yaw_degrees', 90.1), ('head_yaw_degrees', -90.1),
    ('head_pitch_degrees', 60.1), ('head_pitch_degrees', -60.1),
    ('head_pitch_degrees', float('nan')), ('head_yaw_degrees', float('inf')),
])
def test_invalid_head_angles_rejected_before_capture(tmp_path, field, value):
    cfg = yaml.safe_load(DEFAULT.read_text())
    cfg['trajectory'][0][field] = value
    with pytest.raises(ValueError):
        capture_rig(_write(tmp_path, cfg), tmp_path/'output', preview=False)
    assert not (tmp_path/'output').exists()


def test_held_out_pose_must_differ_from_articulated_integration_camera(tmp_path):
    cfg = yaml.safe_load(DEFAULT.read_text())
    frame = cfg['trajectory'][0]
    frame['head_yaw_degrees'], frame['head_pitch_degrees'] = 20., 10.
    pose = rig_transform(frame['position_m'], frame['yaw_degrees']) @ head_camera_pose(cfg['camera'], frame)
    # A look-at up vector may encode roll, so this exactly recreates the pose.
    cfg['held_out'][0].update(eye=pose[:3, 3].tolist(),
                              target=(pose[:3, 3]+pose[:3, 2]).tolist(),
                              up=(-pose[:3, 1]).tolist())
    with pytest.raises(ValueError, match='held-out poses must differ'):
        load_rig_capture_config(_write(tmp_path, cfg))


def test_mujoco_forward_rotates_camera_without_moving_rig_payload_or_torso():
    mujoco = pytest.importorskip('mujoco')
    cfg = load_rig_capture_config(DEFAULT)
    model = mujoco.MjModel.from_xml_string(build_rig_mjcf(cfg))
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
    rig_id = int(model.cam_bodyid[camera_id])
    mocap_id = int(model.body_mocapid[rig_id])
    frame = cfg['trajectory'][2]
    data.mocap_pos[mocap_id] = frame['position_m']
    yaw = np.deg2rad(frame['yaw_degrees'])/2
    data.mocap_quat[mocap_id] = [np.cos(yaw), 0, 0, np.sin(yaw)]
    mujoco.mj_forward(model, data)
    original_geom_pos, original_geom_mat = data.geom_xpos.copy(), data.geom_xmat.copy()
    original_cam_pos = data.cam_xpos[camera_id].copy()
    previous = None
    for yaw, pitch in ((0, 0), (30, 10), (-30, -5)):
        local = head_camera_pose(cfg['camera'], {'head_yaw_degrees': yaw, 'head_pitch_degrees': pitch})
        _set_local_head_pose(mujoco, model, camera_id, local)
        mujoco.mj_forward(model, data)
        actual = opencv_pose_from_mujoco(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
        expected = rig_transform(frame['position_m'], frame['yaw_degrees']) @ local
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(data.geom_xpos, original_geom_pos)
        np.testing.assert_array_equal(data.geom_xmat, original_geom_mat)
        np.testing.assert_array_equal(data.cam_xpos[camera_id], original_cam_pos)
        if previous is not None:
            assert not np.allclose(actual[:3, :3], previous[:3, :3])
        previous = actual


def test_preview_option_rejects_nonboolean_without_creating_output(tmp_path):
    with pytest.raises(ValueError, match='preview'):
        capture_rig(DEFAULT, tmp_path/'output', preview=1)
    assert not (tmp_path/'output').exists()


@RENDER
def test_rotating_capture_records_exact_pose_and_masked_observations(tmp_path):
    pytest.importorskip('mujoco')
    cfg = yaml.safe_load(DEFAULT.read_text())
    cfg['camera'].update(width=96, height=72)
    first = cfg['trajectory'][0]
    second = copy.deepcopy(first)
    second.update(id='turn_left', timestamp_s=.5, head_yaw_degrees=20., head_pitch_degrees=5.)
    cfg['trajectory'] = [first, second]
    path = _write(tmp_path, cfg)
    out = tmp_path/'capture'
    report = capture_rig(path, out, preview=False)
    assert 'preview_gif' not in report['paths']
    assert not (out/'capture_preview.gif').exists()
    assert report == json.loads((out/'capture.json').read_text())
    clear = RGBDSequence(out/report['paths']['unoccluded_manifest'])
    blocked = RGBDSequence(out/report['paths']['occluded_manifest'])
    rows = report['frames']
    np.testing.assert_array_equal(rows[0]['T_wrig'], rows[1]['T_wrig'])
    np.testing.assert_array_equal(rows[0]['T_wbox'], rows[1]['T_wbox'])
    assert not np.allclose(rows[0]['T_wc'], rows[1]['T_wc'])
    box_masks = []
    for frame, row in zip(cfg['trajectory'], rows):
        local = head_camera_pose(cfg['camera'], frame)
        np.testing.assert_allclose(row['T_rig_camera'], local, atol=1e-12)
        np.testing.assert_allclose(row['T_wc'], np.asarray(row['T_wrig']) @ local, atol=1e-8)
        directory = blocked.root/'frames'/frame['id']
        with Image.open(directory/'box_mask.png') as image:
            box = np.asarray(image) > 0
        with Image.open(directory/'excluded_mask.png') as image:
            excluded = np.asarray(image) > 0
        assert box.any()
        view = next(v for v in blocked.views if v['id'] == frame['id'])
        rgb, depth = blocked.read(view)
        raw = np.load(directory/'raw_depth_m.npy')
        assert np.any(depth > 0)
        np.testing.assert_array_equal(depth[excluded], 0)
        np.testing.assert_array_equal(rgb[excluded], 0)
        np.testing.assert_array_equal(depth[~excluded], raw[~excluded])
        box_masks.append(box)
    assert not np.array_equal(*box_masks)
    for view in clear.views:
        if view['split'] == 'held_out':
            for left, right in zip(clear.read(view), blocked.read(view)):
                np.testing.assert_array_equal(left, right)
