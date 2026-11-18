"""Paired RGB-D capture from a kinematic head-camera and carried-box rig.

Simulator poses and segmentation are privileged inputs to this baseline.
Only observed, masked depth is exported for fusion; hidden geometry is never
inserted behind the payload. No physics stepping or locomotion is performed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .mujoco_capture import intrinsics_from_fovy, opencv_pose_from_mujoco, set_camera_pose
from .render import look_at_pose
from .sequence import SequenceWriter


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be a finite '+('positive number' if positive else 'number'))
    return float(value)


def _vector(value, name, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must contain three coordinates')
    return [_number(v, name, positive) for v in value]


def _keys(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f'{name} requires exactly: {", ".join(sorted(required))}')


def rig_transform(position_m, yaw_degrees):
    angle = np.deg2rad(yaw_degrees)
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    pose[:3,3] = position_m
    return pose



def head_camera_pose(camera, frame):
    """Local OpenCV camera pose with independent yaw and pitch at the mount.

    Yaw is around rig +Z and rotates forward +Y toward -X. Pitch is around
    the calibrated camera's local +X and raises its optical axis when positive.
    The mount translation, payload and torso are unaffected by either angle.
    """
    yaw = _number(frame.get('head_yaw_degrees', 0), 'trajectory.head_yaw_degrees')
    pitch = _number(frame.get('head_pitch_degrees', 0), 'trajectory.head_pitch_degrees')
    if not -90 <= yaw <= 90 or not -60 <= pitch <= 60:
        raise ValueError('head yaw must be in [-90, 90] and pitch in [-60, 60] degrees')
    pose = look_at_pose(camera['position_in_rig_m'], camera['target_in_rig_m'], [0, 0, 1])
    yaw_rotation = rig_transform([0, 0, 0], yaw)[:3, :3]
    angle = np.deg2rad(pitch)
    c, s = np.cos(angle), np.sin(angle)
    pitch_rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pose[:3, :3] = yaw_rotation @ pose[:3, :3] @ pitch_rotation
    return pose


def _set_local_head_pose(mujoco, model, camera_id, local_pose):
    """Update the attached camera orientation without changing its parent body."""
    gl_rotation = local_pose[:3, :3] @ np.diag([1., -1., -1.])
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(gl_rotation).reshape(9))
    model.cam_quat[camera_id] = quaternion


def load_rig_capture_config(path):
    """Validate configuration without importing MuJoCo or initializing GL."""
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    _keys(cfg, {'schema_version','scene_name','camera','stairs','box','torso','trajectory','held_out','roi'}, 'rig capture')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('rig capture requires schema_version: 1')
    if not isinstance(cfg['scene_name'], str) or not cfg['scene_name'].strip():
        raise ValueError('scene_name must be a nonempty string')
    camera = cfg['camera']
    _keys(camera, {'width','height','fovy_degrees','depth_trunc_m','position_in_rig_m','target_in_rig_m'}, 'camera')
    if any(type(camera[k]) is not int or not 8 <= camera[k] <= 1024 for k in ('width','height')):
        raise ValueError('camera width/height must be integers in [8, 1024]')
    camera['fovy_degrees'] = _number(camera['fovy_degrees'], 'camera.fovy_degrees', True)
    camera['depth_trunc_m'] = _number(camera['depth_trunc_m'], 'camera.depth_trunc_m', True)
    intrinsics_from_fovy(camera['width'], camera['height'], camera['fovy_degrees'])
    for key in ('position_in_rig_m','target_in_rig_m'):
        camera[key] = _vector(camera[key], f'camera.{key}')
    head_camera_pose(camera, {})
    stairs = cfg['stairs']
    _keys(stairs, {'origin_m','width_m','tread_m','rise_m','steps'}, 'stairs')
    stairs['origin_m'] = _vector(stairs['origin_m'], 'stairs.origin_m')
    for key in ('width_m','tread_m','rise_m'):
        stairs[key] = _number(stairs[key], f'stairs.{key}', True)
    if type(stairs['steps']) is not int or not 1 <= stairs['steps'] <= 32:
        raise ValueError('stairs.steps must be an integer in [1, 32]')
    for name in ('box','torso'):
        _keys(cfg[name], {'size_m','position_in_rig_m'}, name)
        cfg[name]['size_m'] = _vector(cfg[name]['size_m'], f'{name}.size_m', True)
        cfg[name]['position_in_rig_m'] = _vector(cfg[name]['position_in_rig_m'], f'{name}.position_in_rig_m')
    _keys(cfg['roi'], {'min_m','max_m'}, 'roi')
    for key in ('min_m','max_m'):
        cfg['roi'][key] = _vector(cfg['roi'][key], f'roi.{key}')
    if np.any(np.array(cfg['roi']['min_m']) >= cfg['roi']['max_m']):
        raise ValueError('ROI min must be strictly below max on every axis')
    ids, previous, integration_poses = set(), -np.inf, []
    for split in ('trajectory','held_out'):
        frames = cfg[split]
        if not isinstance(frames, list) or not 1 <= len(frames) <= 100:
            raise ValueError(f'{split} must contain between 1 and 100 views')
        for frame in frames:
            keys = {'id','timestamp_s','position_m','yaw_degrees'} if split=='trajectory' else {'id','timestamp_s','eye','target','up'}
            if split == 'trajectory':
                optional = {'head_yaw_degrees', 'head_pitch_degrees'}
                if not isinstance(frame, dict) or not keys <= set(frame) or set(frame)-keys-optional:
                    raise ValueError('trajectory requires id, timestamp_s, position_m and yaw_degrees; '
                                     'only head_yaw_degrees and head_pitch_degrees are optional')
            else:
                _keys(frame, keys, split)
            name = frame['id']
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', name) or name in ids:
                raise ValueError('frame IDs must be unique safe strings')
            ids.add(name)
            time = _number(frame['timestamp_s'], 'timestamp_s')
            if time < 0 or time < previous:
                raise ValueError('capture timestamps must be nonnegative and nondecreasing, including held-outs')
            frame['timestamp_s'], previous = time, time
            if split == 'trajectory':
                frame['position_m'] = _vector(frame['position_m'], 'trajectory.position_m')
                frame['yaw_degrees'] = _number(frame['yaw_degrees'], 'trajectory.yaw_degrees')
                for key in ('head_yaw_degrees', 'head_pitch_degrees'):
                    frame[key] = _number(frame.get(key, 0), f'trajectory.{key}')
                integration_poses.append(rig_transform(frame['position_m'],frame['yaw_degrees']) @ head_camera_pose(camera, frame))
            else:
                for key in ('eye','target','up'):
                    frame[key] = _vector(frame[key], f'held_out.{key}')
                pose = look_at_pose(frame['eye'],frame['target'],frame['up'])
                if any(np.allclose(pose, other, atol=1e-8, rtol=0) for other in integration_poses):
                    raise ValueError('held-out poses must differ from all integration poses')
    cfg['_path'] = str(path)
    return cfg


def _values(values):
    return ' '.join(f'{value:.12g}' for value in values)


def build_rig_mjcf(cfg):
    """Build a self-contained scene from validated dimensions and mounting."""
    stairs, camera = cfg['stairs'], cfg['camera']
    cx, front, base = stairs['origin_m']
    width, tread, rise, count = (stairs[key] for key in ('width_m','tread_m','rise_m','steps'))
    y_min, y_max = min(-.4, front-.8), front+count*tread+.4
    local = look_at_pose(camera['position_in_rig_m'],camera['target_in_rig_m'],[0,0,1])
    gl = local[:3,:3] @ np.diag([1.,-1.,-1.])
    lines = ['<mujoco model="stairs_head_camera">', '<compiler angle="degree"/>',
             f'<visual><global offwidth="{max(128,camera["width"])}" offheight="{max(128,camera["height"])}"/>',
             '<map znear="0.005" zfar="10"/><quality offsamples="0"/>',
             '<headlight ambient="0.8 0.8 0.8" diffuse="0.2 0.2 0.2" specular="0 0 0"/></visual>',
             '<default><geom contype="0" conaffinity="0"/></default><worldbody>',
             f'<geom name="floor" type="box" pos="{_values([cx,(y_min+y_max)/2,base-.04])}" size="{_values([width/2+.7,(y_max-y_min)/2,.04])}" rgba="0.32 0.42 0.36 1"/>']
    for i in range(count):
        height = (i+1)*rise
        lines.append(f'<geom name="stair_{i}" type="box" pos="{_values([cx,front+(i+.5)*tread,base+height/2])}" size="{_values([width/2,tread/2,height/2])}" rgba="{_values([.45+.05*(i%2),.55+.05*(i%2),.70,1])}"/>')
    lines += ['<camera name="evaluation_camera" mode="fixed" fovy="'+str(camera['fovy_degrees'])+'"/>',
              '<body name="sensor_rig" mocap="true">',
              f'<camera name="head_camera" mode="fixed" pos="{_values(camera["position_in_rig_m"])}" xyaxes="{_values(np.r_[gl[:,0],gl[:,1]])}" fovy="{camera["fovy_degrees"]}"/>']
    for name, group, color in (('torso',1,[.20,.28,.38,1]),('box',2,[.80,.48,.16,1])):
        obj = cfg[name]
        lines.append(f'<geom name="{name}" group="{group}" type="box" pos="{_values(obj["position_in_rig_m"])}" size="{_values(np.array(obj["size_m"])/2)}" rgba="{_values(color)}"/>')
    lines += ['</body></worldbody></mujoco>']
    return '\n'.join(lines)


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _render_observation(mujoco, renderer, data, camera, option, depth_limit, far):
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().astype(np.float32, copy=True)
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    invalid = ~np.isfinite(depth) | (depth <= 0) | (depth >= depth_limit) | (depth >= far-max(1e-6,far*1e-5))
    depth[invalid] = 0
    rgb[invalid] = 0
    return rgb, depth, segmentation


def _geom_mask(mujoco, segmentation, ids):
    return (segmentation[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:,:,0], ids)


def capture_rig(config_path, output_dir, progress=None, *, preview=True):
    """Capture paired observations and shared evaluation-only references.

    Disable preview for dense high-resolution sequences; raw frame assets are
    still saved for streaming downstream video exports.
    """
    if type(preview) is not bool:
        raise ValueError('preview must be a boolean')
    cfg = load_rig_capture_config(config_path)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'rig capture output must be new or empty: {out}')
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; install the project's .[sim] extra") from exc
    xml = build_rig_mjcf(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
    eval_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'evaluation_camera')
    rig_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'sensor_rig')
    mocap_id = int(model.body_mocapid[rig_id])
    if model.cam_bodyid[cam_id] != rig_id or mocap_id < 0:
        raise RuntimeError('head camera must be attached to the moving mocap rig')
    geom_id = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    robot_ids, box_ids = [geom_id('torso')], [geom_id('box')]
    stair_ids = [geom_id(f'stair_{i}') for i in range(cfg['stairs']['steps'])]
    width, height = cfg['camera']['width'], cfg['camera']['height']
    k = intrinsics_from_fovy(width,height,float(model.cam_fovy[cam_id]))
    camera = {'width':width, 'height':height, 'K':k.tolist(), 'depth_trunc_m':cfg['camera']['depth_trunc_m']}
    local_camera = look_at_pose(cfg['camera']['position_in_rig_m'],cfg['camera']['target_in_rig_m'],[0,0,1])
    near, far = float(model.vis.map.znear*model.stat.extent), float(model.vis.map.zfar*model.stat.extent)
    if not 0 < near < far:
        raise ValueError('invalid MuJoCo clipping planes')
    metadata = {
        'producer':'wtumi_recon.rig_capture', 'mujoco_version':mujoco.__version__,
        'world_up':[0,0,1], 'world_unit':'metre', 'rig_forward':[0,1,0],
        'source_config':cfg['_path'], 'generated_mjcf_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'mujoco_gl_requested':os.environ.get('MUJOCO_GL','platform_default'),
        'offscreen_multisample_count':0, 'near_clip_m':near, 'far_clip_m':far,
        'T_rig_camera':local_camera.tolist(),
        'T_rig_camera_definition':'Nominal mounting pose at zero head angles; actual local poses are stored per frame',
        'head_rotation_definition':'R_rig_camera = Rz(head_yaw) @ R_nominal @ Rx(head_pitch), fixed mount translation; positive pitch raises the optical axis',
        'pose_source':'Exact simulator camera poses from mj_forward; no estimated localization',
        'mask_source':'Simulator geom IDs/type; torso and payload excluded before all map integration',
        'depth_source':'Metric camera-Z depth; invalid returns and excluded pixels are zero',
        'held_out_policy':'Environment only, shared between conditions, evaluation only',
        'capture_motion':'Kinematic mocap body waypoints with optional independent camera yaw/pitch at a fixed mount; torso and box remain rigidly attached; no physics stepping',
    }
    options = {}
    for condition in ('unoccluded','occluded','evaluation'):
        option = mujoco.MjvOption()
        option.geomgroup[1] = condition != 'evaluation'
        option.geomgroup[2] = condition == 'occluded'
        options[condition] = option
    records, preview_frames, held_masks = [], [], {}
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        # GL initialization succeeds before creating the capture directory.
        writers = {condition:SequenceWriter(out/condition,camera,cfg['scene_name'],
                   'simulator_ground_truth', {**metadata,'condition':condition})
                   for condition in ('unoccluded','occluded')}
        (out/'model_source.xml').write_text(xml, encoding='utf-8')
        (out/'capture_source.yaml').write_text(Path(cfg['_path']).read_text(), encoding='utf-8')
        for index, frame in enumerate(cfg['trajectory']):
            if progress: progress(f"Capturing paired waypoint {index+1}/{len(cfg['trajectory'])}: {frame['id']}")
            angle = np.deg2rad(frame['yaw_degrees'])/2
            data.mocap_pos[mocap_id] = frame['position_m']
            data.mocap_quat[mocap_id] = [np.cos(angle),0,0,np.sin(angle)]
            frame_camera = head_camera_pose(cfg['camera'], frame)
            _set_local_head_pose(mujoco, model, cam_id, frame_camera)
            mujoco.mj_forward(model,data)
            T_wrig = np.eye(4)
            T_wrig[:3,:3], T_wrig[:3,3] = data.xmat[rig_id].reshape(3,3), data.xpos[rig_id]
            actual = opencv_pose_from_mujoco(data.cam_xpos[cam_id],data.cam_xmat[cam_id])
            if not np.allclose(actual,T_wrig @ frame_camera,atol=1e-8,rtol=0):
                raise RuntimeError('actual head-camera pose disagrees with calibrated rig transform')
            payload_pose = np.eye(4)
            payload_pose[:3,:3], payload_pose[:3,3] = data.geom_xmat[box_ids[0]].reshape(3,3),data.geom_xpos[box_ids[0]]
            entry = {'id':frame['id'], 'timestamp_s':frame['timestamp_s'], 'T_wc':actual.tolist(),
                     'T_wrig':T_wrig.tolist(), 'T_wbox':payload_pose.tolist(),
                     'T_rig_camera':frame_camera.tolist(),
                     'head_yaw_degrees':frame['head_yaw_degrees'],
                     'head_pitch_degrees':frame['head_pitch_degrees'], 'conditions':{}}
            if preview:
                preview_image = Image.new('RGB',(width*4,height*4+72),'#111827')
                draw = ImageDraw.Draw(preview_image)
                draw.text((10,8),f"{frame['id']} | Kinematic head-camera playback | Raw / masked RGB",fill='white')
            for row,condition in enumerate(('unoccluded','occluded')):
                raw_rgb, raw_depth, segmentation = _render_observation(mujoco,renderer,data,'head_camera',options[condition],camera['depth_trunc_m'],far)
                robot_mask = _geom_mask(mujoco,segmentation,robot_ids)
                box_mask = _geom_mask(mujoco,segmentation,box_ids)
                excluded = robot_mask | box_mask
                rgb, depth = raw_rgb.copy(), raw_depth.copy()
                rgb[excluded], depth[excluded] = 0, 0
                if not np.any(depth > 0):
                    raise ValueError(f'{condition}/{frame["id"]} contains no valid environment depth')
                writers[condition].add_frame(frame['id'],'integration',rgb,depth,actual,frame['timestamp_s'])
                directory = out/condition/'frames'/frame['id']
                Image.fromarray(raw_rgb).save(directory/'raw_rgb.png')
                np.save(directory/'raw_depth_m.npy',raw_depth,allow_pickle=False)
                for name,mask in (('excluded',excluded),('box',box_mask),('robot',robot_mask)):
                    Image.fromarray(mask.astype(np.uint8)*255).save(directory/f'{name}_mask.png')
                entry['conditions'][condition] = {'raw_valid_pixels':int(np.count_nonzero(raw_depth)),
                    'valid_environment_pixels':int(np.count_nonzero(depth)), 'excluded_pixels':int(excluded.sum()),
                    'box_pixels':int(box_mask.sum()), 'robot_pixels':int(robot_mask.sum()),
                    'excluded_pixel_fraction':float(excluded.mean()),
                    'frame_directory':f"{condition}/frames/{frame['id']}"}
                if preview:
                    y = 30+row*(height*2+22)
                    draw.text((10,y),f'{condition}: raw',fill='white')
                    draw.text((width*2+10,y),'excluded pixels set to invalid',fill='white')
                    for col,image in enumerate((raw_rgb,rgb)):
                        preview_image.paste(Image.fromarray(image).resize((width*2,height*2),Image.Resampling.NEAREST), (col*width*2,y+18))
            records.append(entry)
            if preview:
                preview_frames.append(preview_image)
        for frame in cfg['held_out']:
            if progress: progress(f"Capturing evaluation-only view: {frame['id']}")
            pose = look_at_pose(frame['eye'],frame['target'],frame['up'])
            actual = set_camera_pose(mujoco,model,data,eval_id,pose)
            rgb, depth, segmentation = _render_observation(mujoco,renderer,data,'evaluation_camera',options['evaluation'],camera['depth_trunc_m'],far)
            mask = _geom_mask(mujoco,segmentation,stair_ids) & (depth > 0)
            if not mask.any():
                raise ValueError(f"held-out camera {frame['id']} sees no staircase pixels")
            for writer in writers.values():
                writer.add_frame(frame['id'],'held_out',rgb,depth,actual,frame['timestamp_s'])
            directory = out/'evaluation'/frame['id']; directory.mkdir(parents=True)
            Image.fromarray(mask.astype(np.uint8)*255).save(directory/'stairs_mask.png')
            held_masks[frame['id']] = f"evaluation/{frame['id']}/stairs_mask.png"
        for writer in writers.values():
            writer.finish()
    if preview:
        preview_frames[0].save(out/'capture_preview.gif',save_all=True,append_images=preview_frames[1:],duration=650,loop=0)
    report = {
        'format':'wtumi-rig-capture-v1', 'scene_name':cfg['scene_name'], 'stairs':cfg['stairs'], 'roi':cfg['roi'],
        'rig':{key:copy.deepcopy(cfg[key]) for key in ('camera','torso','box')},
        'integration_frame_ids':[v['id'] for v in cfg['trajectory']],
        'held_out_frame_ids':[v['id'] for v in cfg['held_out']],
        'frames':records, 'metadata':metadata,
        'paths':{'unoccluded_manifest':'unoccluded/manifest.json','occluded_manifest':'occluded/manifest.json',
                 'held_out_masks':held_masks,'model':'model_source.xml',
                 **({'preview_gif':'capture_preview.gif'} if preview else {})},
        'scope':'Kinematic sensor/payload proxy with exact simulator poses and masks; no locomotion, pose estimation, tactile sensing or hidden-depth completion.',
    }
    _save_json(out/'capture.json',report)
    return report
