"""Camera geometry and opt-in MuJoCo rendering checks (Mac/Linux)."""

import json
import os

import numpy as np
import pytest
import yaml

from wtumi_recon.mujoco_capture import (
    capture_mujoco,
    intrinsics_from_fovy,
    load_capture_config,
    opencv_pose_from_mujoco,
    set_camera_pose,
)
from wtumi_recon.render import look_at_pose
from wtumi_recon.sequence import RGBDSequence


RENDER = pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                            reason='Set WTUMI_TEST_MUJOCO_RENDER=1 on a machine with working MuJoCo GL')


def test_fovy_intrinsics_and_camera_axis_conversion():
    k = intrinsics_from_fovy(80, 60, 90)
    np.testing.assert_allclose(k, [[30, 0, 39.5], [0, 30, 29.5], [0, 0, 1]])
    pose = opencv_pose_from_mujoco([1, 2, 3], np.eye(3))
    np.testing.assert_allclose(pose[:3, :3], np.diag([1, -1, -1]))
    np.testing.assert_array_equal(pose[:3, 3], [1, 2, 3])
    assert np.linalg.det(pose[:3, :3]) == 1


@pytest.mark.parametrize('width,height,fovy', [(0,60,60),(80,True,60),(80,60,0),(80,60,180),(80,60,np.nan)])
def test_invalid_intrinsics_are_rejected(width, height, fovy):
    with pytest.raises(ValueError):
        intrinsics_from_fovy(width,height,fovy)


def _config(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'small.xml').write_text('''<mujoco>
      <visual><global offwidth="128" offheight="128"/><map znear="0.01" zfar="10"/></visual>
      <worldbody><camera name="capture" pos="0 0 2" fovy="80"/>
      <geom type="box" size="0.4 0.4 0.1" rgba="0.8 0.2 0.1 1"/></worldbody></mujoco>''')
    cfg = {'schema_version':1, 'scene_name':'small', 'mjcf':'assets/small.xml',
           'camera':{'name':'capture','width':80,'height':60,'depth_trunc_m':5},
           'views':[
               {'id':'front','split':'integration','timestamp_s':0.,'eye':[0,0,2],'target':[0,0,0],'up':[0,1,0]},
               {'id':'novel','split':'held_out','timestamp_s':1.,'eye':[.1,0,2],'target':[0,0,0],'up':[0,1,0]},
           ]}
    path = tmp_path / 'capture.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path,cfg


def test_capture_paths_and_held_out_validation(tmp_path, monkeypatch):
    path,cfg = _config(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    resolved = load_capture_config(path)
    assert resolved['mjcf'] == str(tmp_path/'assets/small.xml')
    assert len(resolved['views']) == 2
    expected = look_at_pose([0,0,2],[0,0,0],[0,1,0])
    np.testing.assert_array_equal(resolved['views'][0]['T_wc'],expected)
    cfg['views'][1]['eye'] = [0,0,2]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='held_out poses'):
        load_capture_config(path)


def test_capture_refuses_nonempty_output_before_rendering(tmp_path):
    path,_ = _config(tmp_path)
    out = tmp_path/'results'
    out.mkdir()
    (out/'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        capture_mujoco(path,out)
    assert (out/'keep.txt').read_text() == 'keep'


def test_actual_mujoco_camera_pose_roundtrip_without_gl():
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="capture"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    expected = look_at_pose([1.2,-.4,1.5],[.2,1,.7],[0,0,1])
    actual = set_camera_pose(mujoco,model,data,0,expected)
    np.testing.assert_allclose(actual,expected,atol=1e-10)


@RENDER
def test_mujoco_depth_is_camera_z_and_matches_analytic_tilted_plane():
    mujoco = pytest.importorskip('mujoco')
    # The finite box top lies exactly on world z=0 and covers the entire view.
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="128" offheight="128"/>
      <map znear="0.005" zfar="10"/></visual><worldbody><camera name="capture" fovy="58"/>
      <geom type="box" pos="0 0 -0.05" size="5 5 0.05"/></worldbody></mujoco>''')
    # Match capture_mujoco: metric depth is sampled at pixel centers.
    # MSAA depth resolve can select subpixel samples differently across drivers.
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    width,height=96,64
    k=intrinsics_from_fovy(width,height,58)
    u,v=np.meshgrid(np.arange(width),np.arange(height))
    camera_rays=np.stack((u,v,np.ones_like(u)),axis=-1) @ np.linalg.inv(k).T
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        for target in ([0,0,0],[.35,.2,0]):
            pose=look_at_pose([0,0,2],target,[0,1,0])
            actual=set_camera_pose(mujoco,model,data,0,pose)
            renderer.update_scene(data,camera='capture')
            renderer.enable_depth_rendering()
            depth=renderer.render().copy()
            directions=camera_rays @ actual[:3,:3].T
            expected=-actual[2,3]/directions[:,:,2]
            np.testing.assert_allclose(depth,expected,atol=2e-4,rtol=0)
            if target == [0,0,0]:
                # Euclidean range would grow off-axis; optical-axis depth does not.
                np.testing.assert_allclose(depth,2,atol=2e-4,rtol=0)
                assert np.linalg.norm(camera_rays[0,0]*depth[0,0]) > 2.2


@RENDER
def test_real_capture_exports_actual_rgb_depth_pose_and_zero_background(tmp_path):
    pytest.importorskip('mujoco')
    path,cfg=_config(tmp_path)
    manifest=capture_mujoco(path,tmp_path/'result')
    sequence=RGBDSequence(manifest)
    assert sequence.reference_kind == 'simulator_ground_truth'
    assert sequence.metadata['world_up'] == [0,0,1]
    assert len(sequence.views) == 2
    for actual,view in zip(sequence.views,cfg['views']):
        rgb,depth=sequence.read(actual)
        assert rgb.dtype == np.uint8 and depth.dtype == np.float32
        assert rgb.shape == (60,80,3) and depth.shape == (60,80)
        assert np.any(depth == 0) and np.any(depth > 0)
        np.testing.assert_array_equal(rgb[depth == 0],0)
        assert np.isfinite(depth).all() and depth.max() < 5
        expected=look_at_pose(view['eye'],view['target'],view['up'])
        np.testing.assert_allclose(actual['T_wc'],expected,atol=1e-8)
    _,front=sequence.read('front')
    assert front[30,40] == pytest.approx(1.9,abs=2e-4)
    k=np.array(sequence.camera['K'])
    u,v=np.meshgrid(np.arange(80),np.arange(60))
    expected_inside=(np.abs((u-k[0,2])/k[0,0]*1.9) < .4) & (np.abs((v-k[1,2])/k[1,1]*1.9) < .4)
    np.testing.assert_array_equal(front>0,expected_inside)
    np.testing.assert_allclose(front[expected_inside],1.9,atol=2e-4,rtol=0)
    assert sequence.metadata['offscreen_multisample_count'] == 0
    saved=json.loads((tmp_path/'result/capture.json').read_text())
    assert saved['frames'][0]['valid_depth_pixels'] == np.count_nonzero(front)
