"""Frozen Unitree G1 visual geometry for the RGB-D reconstruction rig.

The vendored MJCF and meshes remain unchanged. This adapter evaluates a single
joint configuration, then writes fixed body transforms and authored visual
geometries into the scene. It supplies no walking or grasp controller.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np


G1_JOINT_NAMES = tuple(
    [f'{side}_{part}_joint' for side in ('left', 'right') for part in
     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    + [f'{side}_{part}_joint' for side in ('left', 'right') for part in
       ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
        'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
)

# An illustrative pose with the hands alongside a small carried box. It is
# specified in the original +X-forward G1 model frame, not the +Y-forward rig.
CARRY_JOINT_POSITIONS_DEGREES = {
    'left_shoulder_pitch_joint': -40.0, 'right_shoulder_pitch_joint': -40.0,
    'left_shoulder_roll_joint': 40.0, 'right_shoulder_roll_joint': -40.0,
    'left_shoulder_yaw_joint': -15.0, 'right_shoulder_yaw_joint': 15.0,
    'left_elbow_joint': 50.0, 'right_elbow_joint': 50.0,
}


def normalize_g1_config(value):
    """Validate the robot settings without loading MuJoCo or opening graphics."""
    allowed = {'model', 'pose', 'base_height_m', 'joint_positions_degrees'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('robot supports model, pose, base_height_m and joint_positions_degrees only')
    cfg = {'model': 'unitree_g1', 'pose': 'carry', 'base_height_m': .793,
           'joint_positions_degrees': {}, **copy.deepcopy(value)}
    if cfg['model'] != 'unitree_g1' or cfg['pose'] not in ('stand', 'carry'):
        raise ValueError('robot requires model unitree_g1 and pose stand or carry')
    height = cfg['base_height_m']
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height) or height <= 0:
        raise ValueError('robot.base_height_m must be a finite positive number')
    cfg['base_height_m'] = float(height)
    joints = cfg['joint_positions_degrees']
    if not isinstance(joints, dict) or set(joints) - set(G1_JOINT_NAMES):
        raise ValueError('robot joint_positions_degrees must map known G1 joint names to angles')
    for name, angle in joints.items():
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle):
            raise ValueError(f'robot angle for {name} must be a finite number in degrees')
        joints[name] = float(angle)
    return cfg


def _source_pose(asset_root, robot_cfg):
    import mujoco

    cfg = normalize_g1_config(robot_cfg)
    model = mujoco.MjModel.from_xml_path(str(Path(asset_root) / 'g1.xml'))
    data = mujoco.MjData(model)
    if cfg['pose'] == 'stand':
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        if key < 0:
            raise ValueError('vendored G1 model is missing its stand keyframe')
        mujoco.mj_resetDataKeyframe(model, data, key)
    angles = {**(CARRY_JOINT_POSITIONS_DEGREES if cfg['pose'] == 'carry' else {}),
              **cfg['joint_positions_degrees']}
    for name, angle in angles.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f'vendored G1 model is missing hinge joint: {name}')
        radians = np.deg2rad(angle)
        if model.jnt_limited[joint] and not model.jnt_range[joint, 0] <= radians <= model.jnt_range[joint, 1]:
            raise ValueError(f'robot angle for {name} exceeds the source model joint limits')
        data.qpos[model.jnt_qposadr[joint]] = radians
    floating = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    if floating < 0 or model.jnt_type[floating] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('vendored G1 model is missing its floating base')
    offset = int(model.jnt_qposadr[floating])
    data.qpos[offset:offset + 7] = [0, 0, cfg['base_height_m'], 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return model, data, cfg


def _values(values):
    return ' '.join(f'{float(v):.17g}' for v in values)


def append_g1_visual(root_xml, sensor_rig_element, asset_root, asset_prefix, robot_cfg):
    """Append fixed G1 geometry, with all self-occluding meshes in group 1.

    asset_root contains the original g1.xml, assets/, LICENSE and SOURCE.json.
    asset_prefix is the portable path to that directory in the generated scene.
    The caller copies the assets and retains the license/provenance alongside
    the generated model. No joints, actuators or collision proxies are copied.
    """
    import mujoco

    source_root = Path(asset_root).resolve()
    prefix = PurePosixPath(asset_prefix)
    if prefix.is_absolute() or '..' in prefix.parts or not prefix.parts or '\\' in str(asset_prefix):
        raise ValueError('G1 asset_prefix must be a relative path inside the scene')
    source = ET.parse(source_root / 'g1.xml').getroot()
    model, data, cfg = _source_pose(source_root, robot_cfg)
    copied_assets = []
    for element in source.findall('./asset/*'):
        if element.tag not in ('mesh', 'material'):
            raise ValueError(f'unsupported vendored G1 asset: {element.tag}')
        copied = copy.deepcopy(element)
        old_name = element.get('name', Path(element.get('file', '')).stem)
        copied.set('name', 'g1_' + old_name)
        if element.tag == 'mesh':
            mesh_path = (source_root / 'assets' / element.get('file')).resolve()
            if not mesh_path.is_relative_to(source_root) or not mesh_path.is_file():
                raise ValueError('vendored G1 mesh must exist inside its asset directory')
            copied.set('file', str(prefix / 'assets' / element.get('file')))
        copied_assets.append(copied)
    wrapper = ET.Element('body', name='g1_root', quat=_values([np.sqrt(.5), 0, 0, np.sqrt(.5)]))
    geom_names = []

    def freeze_body(source_body, destination_parent, parent_id):
        name = source_body.get('name')
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f'G1 body missing from compiled model: {name}')
        parent_rotation = data.xmat[parent_id].reshape(3, 3)
        rotation = parent_rotation.T @ data.xmat[body].reshape(3, 3)
        position = parent_rotation.T @ (data.xpos[body] - data.xpos[parent_id])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).reshape(9))
        destination = ET.SubElement(destination_parent, 'body', name='g1_' + name,
                                    pos=_values(position), quat=_values(quat))
        for authored in source_body.findall('geom'):
            if authored.get('class') != 'visual':
                continue
            attrs = {key: value for key, value in authored.attrib.items() if key not in ('class', 'name')}
            name = f'g1_visual_{len(geom_names):03d}'
            # Keep the authored local mesh pose, not compiled geom_pos: MuJoCo
            # applies its mesh-centering transform again when compiling MJCF.
            attrs.update(name=name, type='mesh', group='1', contype='0', conaffinity='0', density='0',
                         mesh='g1_' + authored.get('mesh'), material='g1_' + authored.get('material', 'metal'))
            ET.SubElement(destination, 'geom', **attrs)
            geom_names.append(name)
        for child in source_body.findall('body'):
            freeze_body(child, destination, body)

    for body in source.findall('./worldbody/body'):
        freeze_body(body, wrapper, 0)
    created_names = [element.get('name') for element in copied_assets] + [element.get('name') for element in wrapper.iter()]
    existing_names = {element.get('name') for element in root_xml.iter() if element.get('name')}
    if existing_names.intersection(created_names):
        raise ValueError('scene names conflict with reserved g1_ visual model names')
    assets = root_xml.find('asset')
    if assets is None:
        assets = ET.SubElement(root_xml, 'asset')
    assets.extend(copied_assets)
    sensor_rig_element.append(wrapper)
    provenance = json.loads((source_root / 'SOURCE.json').read_text())
    actual_angles = {name: float(np.rad2deg(data.qpos[model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])) for name in G1_JOINT_NAMES}
    return {
        'model': cfg['model'], 'pose': cfg['pose'], 'base_height_m': cfg['base_height_m'],
        'joint_positions_degrees': actual_angles, 'visual_geom_names': geom_names,
        'visual_geom_count': len(geom_names), 'source_commit': provenance['commit'],
        'source_url': provenance['source_url'], 'license': provenance['license'],
        'source_to_rig_yaw_degrees': 90.0,
        'scope': 'Frozen source-model forward kinematics for visual self-occlusion; no dynamic controller, physical grasp, or walking simulation.',
    }
