"""Frozen G1 geometry matches source kinematics and remains portable."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wtumi_recon.g1_model import (
    G1_JOINT_NAMES, append_g1_visual, normalize_g1_config, _source_pose,
)


SOURCE = Path(__file__).resolve().parents[1] / 'assets/robots/unitree_g1'


def _assembled(tmp_path, config):
    mujoco = pytest.importorskip('mujoco')
    root = ET.Element('mujoco', model='frozen_g1_test')
    ET.SubElement(root, 'compiler', angle='degree')
    ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    rig = ET.SubElement(world, 'body', name='sensor_rig', mocap='true', pos='1.1 -0.8 0.2', euler='0 0 23')
    prefix = 'robot_assets/unitree_g1'
    report = append_g1_visual(root, rig, SOURCE, prefix, config)
    shutil.copytree(SOURCE, tmp_path / prefix)
    path = tmp_path / 'model.xml'
    ET.ElementTree(root).write(path, encoding='utf-8')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return root, model, data, report


def _vertices(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


@pytest.mark.parametrize('pose', ['stand', 'carry'])
def test_frozen_mesh_vertices_match_source_forward_kinematics(tmp_path, pose):
    mujoco = pytest.importorskip('mujoco')
    config = {'pose': pose, 'base_height_m': .812,
              'joint_positions_degrees': {'waist_yaw_joint': 12, 'left_wrist_roll_joint': 20}}
    original_hashes = {p: p.read_bytes() for p in SOURCE.rglob('*') if p.is_file()}
    source_model, source_data, _ = _source_pose(SOURCE, config)
    root, model, data, report = _assembled(tmp_path, config)
    assert len(G1_JOINT_NAMES) == 29
    assert report['visual_geom_count'] == 35
    assert model.njnt == model.nq == model.nv == model.nu == 0
    assert not root.findall('.//joint') and not root.findall('.//freejoint')
    assert not root.findall('.//actuator') and not root.findall('.//inertial')
    np.testing.assert_array_equal(model.geom_group, 1)
    np.testing.assert_array_equal(model.geom_contype, 0)
    np.testing.assert_array_equal(model.geom_conaffinity, 0)
    rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'sensor_rig')
    yaw90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation = data.xmat[rig].reshape(3, 3) @ yaw90
    source_visuals = np.flatnonzero(source_model.geom_group == 2)
    for source_geom, new_name in zip(source_visuals, report['visual_geom_names']):
        new_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, new_name)
        expected = _vertices(source_model, source_data, source_geom) @ rotation.T + data.xpos[rig]
        actual = _vertices(model, data, new_geom)
        # This catches the common double-application of MuJoCo mesh centering,
        # wrong source-to-rig yaw and incorrect fixed-body joint transforms.
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert all(path.read_bytes() == value for path, value in original_hashes.items())


def test_assets_and_license_survive_moving_generated_model(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    before = tmp_path / 'before'
    before.mkdir()
    _assembled(before, {'pose': 'stand'})
    after = tmp_path / 'after'
    before.rename(after)
    mujoco.MjModel.from_xml_path(str(after / 'model.xml'))
    assert (after / 'robot_assets/unitree_g1/LICENSE').read_text() == (SOURCE / 'LICENSE').read_text()
    source = json.loads((after / 'robot_assets/unitree_g1/SOURCE.json').read_text())
    assert source['license'] == 'BSD-3-Clause'
    assert len(source['commit']) == 40


@pytest.mark.parametrize('value', [None, {'enabled': True}, {'model': 'unknown'}, {'pose': 'walk'},
                                    {'base_height_m': False}, {'base_height_m': -1},
                                    {'joint_positions_degrees': {'unknown_joint': 0}},
                                    {'joint_positions_degrees': {'left_elbow_joint': float('nan')}}])
def test_invalid_robot_settings_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_g1_config(value)


def test_joint_limits_and_conflicting_names_fail_before_mutation(tmp_path):
    pytest.importorskip('mujoco')
    root = ET.Element('mujoco')
    rig = ET.SubElement(ET.SubElement(root, 'worldbody'), 'body', name='sensor_rig')
    before = ET.tostring(root)
    with pytest.raises(ValueError, match='joint limits'):
        append_g1_visual(root, rig, SOURCE, 'robot_assets/unitree_g1',
                         {'joint_positions_degrees': {'left_elbow_joint': 1000}})
    assert ET.tostring(root) == before
    ET.SubElement(rig, 'body', name='g1_pelvis')
    before = ET.tostring(root)
    with pytest.raises(ValueError, match='conflict'):
        append_g1_visual(root, rig, SOURCE, 'robot_assets/unitree_g1', {})
    assert ET.tostring(root) == before


def test_camera_only_import_does_not_load_graphics_library():
    script = 'import sys; import wtumi_recon.g1_model; assert "open3d" not in sys.modules; assert "mujoco" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
