"""Capture one configured scene as a portable, masked RGB-D sequence.

The camera and optional torso/payload follow kinematic waypoints. Simulator
poses and segmentation are privileged baseline inputs; no physics stepping,
pose estimation, or geometry completion occurs here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .mujoco_capture import intrinsics_from_fovy, opencv_pose_from_mujoco, set_camera_pose
from .render import look_at_pose
from .rig_capture import _geom_mask, _render_observation, _set_local_head_pose, head_camera_pose
from .sequence import SequenceWriter
from .scene_doors import configured_door_angles, scene_door_ids, set_scene_doors


def _required_id(mujoco, model, object_type, name):
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"scene model is missing required object: {name}")
    return object_id


def _body_belongs_to_rig(model, body_id, rig_id):
    """Include every nested robot link without accepting unrelated world geoms."""
    while body_id:
        if body_id == rig_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _scene_ids(mujoco, model, cfg):
    """Check generated-model contracts before opening a render context."""
    body = _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "sensor_rig")
    head = _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    evaluation = _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, "evaluation_camera")
    mocap = int(model.body_mocapid[body])
    if mocap < 0 or int(model.cam_bodyid[head]) != body:
        raise ValueError("head_camera must be attached directly to the sensor_rig mocap body")
    if int(model.cam_bodyid[evaluation]) != 0:
        raise ValueError("evaluation_camera must be attached to the world body")
    if not np.allclose(model.cam_pos[head], cfg["camera"]["position_in_rig_m"], atol=1e-8, rtol=0):
        raise ValueError("model head-camera position disagrees with the scene configuration")
    for camera_id in (head, evaluation):
        if not np.isclose(model.cam_fovy[camera_id], cfg["camera"]["fovy_degrees"], atol=1e-8, rtol=0):
            raise ValueError("model camera field of view disagrees with the scene configuration")
    result = {"body": body, "head": head, "evaluation": evaluation, "mocap": mocap}
    robot = cfg.get("robot")
    if robot is not None:
        if not isinstance(robot, dict) or robot.get("model") != "unitree_g1":
            raise ValueError("scene robot model must be unitree_g1")
        if cfg["torso"]["enabled"]:
            raise ValueError("unitree_g1 requires the proxy torso to be disabled")
    for name, group in (("torso", 1), ("box", 2)):
        geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if cfg[name]["enabled"]:
            if geom < 0:
                raise ValueError(f"scene model is missing enabled geom: {name}")
            if int(model.geom_bodyid[geom]) != body or int(model.geom_group[geom]) != group:
                raise ValueError(f"{name} must belong to sensor_rig and geom group {group}")
            result[name] = [geom]
        else:
            if geom >= 0:
                raise ValueError(f"scene model contains disabled geom: {name}")
            result[name] = []
    g1_geoms = [geom for geom in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("g1_")]
    if robot is None:
        if g1_geoms:
            raise ValueError("scene model contains G1 geometry without a configured robot")
    else:
        if not g1_geoms:
            raise ValueError("scene model is missing configured G1 geometry")
        for geom in g1_geoms:
            if int(model.geom_group[geom]) != 1 or not _body_belongs_to_rig(model, int(model.geom_bodyid[geom]), body):
                raise ValueError("every G1 geom must belong to sensor_rig descendants and geom group 1")
        if set(np.flatnonzero(model.geom_group == 1).tolist()) != set(g1_geoms):
            raise ValueError("geom group 1 must contain only the configured G1 robot links")
        # Retain the existing key so both capture and pose replay use the same
        # interface, while self masks now cover every rendered robot link.
        result["torso"] = g1_geoms
    result["doors"] = scene_door_ids(mujoco, model, cfg)
    return result


def apply_scene_pose(mujoco, model, data, cfg, frame):
    """Apply one scene waypoint and return the actual calibrated world poses.

    This function does not advance simulation time. It can also be used for
    native viewer playback, including interpolation between capture waypoints.
    """
    ids = _scene_ids(mujoco, model, cfg)
    half_angle = np.deg2rad(frame["yaw_degrees"]) / 2
    data.mocap_pos[ids["mocap"]] = frame["position_m"]
    data.mocap_quat[ids["mocap"]] = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
    local_camera = head_camera_pose(cfg["camera"], frame)
    _set_local_head_pose(mujoco, model, ids["head"], local_camera)
    set_scene_doors(mujoco, model, data, cfg)
    mujoco.mj_forward(model, data)
    world_rig = np.eye(4)
    world_rig[:3, :3] = data.xmat[ids["body"]].reshape(3, 3)
    world_rig[:3, 3] = data.xpos[ids["body"]]
    actual = opencv_pose_from_mujoco(data.cam_xpos[ids["head"]], data.cam_xmat[ids["head"]])
    if not np.allclose(actual, world_rig @ local_camera, atol=1e-8, rtol=0):
        raise RuntimeError("actual head-camera pose disagrees with the calibrated rig transform")
    poses = {"T_wc": actual, "T_wrig": world_rig, "T_rig_camera": local_camera}
    if ids["box"]:
        box = ids["box"][0]
        world_box = np.eye(4)
        world_box[:3, :3] = data.geom_xmat[box].reshape(3, 3)
        world_box[:3, 3] = data.geom_xpos[box]
        poses["T_wbox"] = world_box
    return poses


def capture_scene(cfg, model_path, output_dir, progress=None):
    """Capture a normalized scene configuration and its generated model.

    Integration frames retain raw images and simulator masks alongside masked
    RGB-D for fusion. Held-out images show only the static environment and are
    never integration inputs. Every data path in the report is relative to the
    capture directory; source paths in metadata are descriptive provenance.
    """
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"scene capture output must be new or empty: {out}")
    source_model = Path(model_path).expanduser().resolve()
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; install the project's .[sim] extra") from exc
    model = mujoco.MjModel.from_xml_path(str(source_model))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    ids = _scene_ids(mujoco, model, cfg)
    width, height = cfg["camera"]["width"], cfg["camera"]["height"]
    camera = {
        "width": width, "height": height,
        "K": intrinsics_from_fovy(width, height, float(model.cam_fovy[ids["head"]])).tolist(),
        "depth_trunc_m": cfg["camera"]["depth_trunc_m"],
    }
    near = float(model.vis.map.znear * model.stat.extent)
    far = float(model.vis.map.zfar * model.stat.extent)
    if not 0 < near < far:
        raise ValueError("invalid MuJoCo clipping planes")
    metadata = {
        "producer": "wtumi_recon.scene_capture", "mujoco_version": mujoco.__version__,
        "source_config": cfg.get("_path"), "source_model": str(source_model),
        "model_xml_sha256": hashlib.sha256(source_model.read_bytes()).hexdigest(),
        "model_hash_scope": "Generated model.xml only; does not hash external mesh or texture files",
        "world_up": [0, 0, 1], "world_unit": "metre", "rig_forward": [0, 1, 0],
        "mujoco_gl_requested": os.environ.get("MUJOCO_GL", "platform_default"),
        "offscreen_multisample_count": 0, "near_clip_m": near, "far_clip_m": far,
        "T_rig_camera": head_camera_pose(cfg["camera"], {}).tolist(),
        "T_rig_camera_definition": "Nominal zero-angle mounting pose; actual local camera pose is stored per integration frame",
        "head_rotation_definition": "R_rig_camera = Rz(head_yaw) @ R_nominal @ Rx(head_pitch), fixed mount translation; positive pitch raises the optical axis",
        "pose_source": "Exact simulator camera poses from mj_forward; no estimated localization",
        "mask_source": "Simulator geom IDs/type; all configured robot-link geoms and payload excluded before map integration",
        "robot_model": cfg["robot"]["model"] if cfg.get("robot") is not None else ("box_proxy" if cfg["torso"]["enabled"] else "none"),
        "robot_configuration": copy.deepcopy(cfg.get("robot")),
        "robot_geom_count": len(ids["torso"]),
        "robot_geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) for geom in ids["torso"]],
        "depth_source": "Metric camera-Z depth; invalid returns and excluded pixels are zero",
        "held_out_policy": "Static environment only; evaluation only",
        "capture_motion": "Kinematic mocap waypoints with independent head yaw/pitch; no physics stepping",
        "condition": "Configured scene with optional robot geometry and payload",
        "door_configuration": copy.deepcopy(cfg["environment"].get("doors", [])),
        "door_angles_degrees": configured_door_angles(cfg),
        "door_policy": "All integration and held-out views use the same configured door angles; no moving-door fusion",
    }
    observation_option = mujoco.MjvOption()
    evaluation_option = mujoco.MjvOption()
    observation_option.geomgroup[3] = 0
    evaluation_option.geomgroup[1:4] = 0
    records, held_records = [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        # Avoid leaving a new output directory when GL initialization fails.
        writer = SequenceWriter(out, camera, cfg["scene_name"], "simulator_ground_truth", metadata)
        for index, frame in enumerate(cfg["trajectory"]):
            if progress:
                progress(f"Capturing scene waypoint {index + 1}/{len(cfg['trajectory'])}: {frame['id']}")
            poses = apply_scene_pose(mujoco, model, data, cfg, frame)
            raw_rgb, raw_depth, segmentation = _render_observation(
                mujoco, renderer, data, "head_camera", observation_option, camera["depth_trunc_m"], far,
            )
            robot_mask = _geom_mask(mujoco, segmentation, ids["torso"])
            box_mask = _geom_mask(mujoco, segmentation, ids["box"])
            excluded = robot_mask | box_mask
            rgb, depth = raw_rgb.copy(), raw_depth.copy()
            rgb[excluded], depth[excluded] = 0, 0
            if not np.any(depth > 0):
                raise ValueError(f"integration frame {frame['id']} contains no valid environment depth")
            writer.add_frame(frame["id"], "integration", rgb, depth, poses["T_wc"], frame["timestamp_s"])
            relative_directory = f"frames/{frame['id']}"
            directory = out / relative_directory
            Image.fromarray(raw_rgb).save(directory / "raw_rgb.png")
            np.save(directory / "raw_depth_m.npy", raw_depth, allow_pickle=False)
            for name, mask in (("excluded", excluded), ("box", box_mask), ("robot", robot_mask)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{name}_mask.png")
            records.append({
                "id": frame["id"], "timestamp_s": frame["timestamp_s"],
                **{name: pose.tolist() for name, pose in poses.items()},
                "head_yaw_degrees": frame.get("head_yaw_degrees", 0),
                "head_pitch_degrees": frame.get("head_pitch_degrees", 0),
                "door_angles_degrees": configured_door_angles(cfg),
                "raw_valid_pixels": int(np.count_nonzero(raw_depth)),
                "valid_environment_pixels": int(np.count_nonzero(depth)),
                "excluded_pixels": int(excluded.sum()), "box_pixels": int(box_mask.sum()),
                "robot_pixels": int(robot_mask.sum()), "excluded_pixel_fraction": float(excluded.mean()),
                "frame_directory": relative_directory,
            })
        for frame in cfg["held_out"]:
            if progress:
                progress(f"Capturing evaluation-only view: {frame['id']}")
            desired = look_at_pose(frame["eye"], frame["target"], frame["up"])
            set_scene_doors(mujoco, model, data, cfg)
            actual = set_camera_pose(mujoco, model, data, ids["evaluation"], desired)
            rgb, depth, _ = _render_observation(
                mujoco, renderer, data, "evaluation_camera", evaluation_option, camera["depth_trunc_m"], far,
            )
            if not np.any(depth > 0):
                raise ValueError(f"held-out frame {frame['id']} contains no valid environment depth")
            writer.add_frame(frame["id"], "held_out", rgb, depth, actual, frame["timestamp_s"])
            held_records.append({
                "id": frame["id"], "timestamp_s": frame["timestamp_s"], "T_wc": actual.tolist(),
                "valid_environment_pixels": int(np.count_nonzero(depth)),
                "frame_directory": f"frames/{frame['id']}",
                "door_angles_degrees": configured_door_angles(cfg),
            })
        writer.finish()
    report = {
        "format": "wtumi-scene-capture-v1", "scene_name": cfg["scene_name"],
        "rig": {name: copy.deepcopy(cfg[name]) for name in ("camera", "torso", "box")}
               | ({"robot": copy.deepcopy(cfg["robot"])} if cfg.get("robot") is not None else {}),
        "integration_frame_ids": [frame["id"] for frame in cfg["trajectory"]],
        "held_out_frame_ids": [frame["id"] for frame in cfg["held_out"]],
        "frames": records, "held_out_frames": held_records, "metadata": metadata,
        "paths": {"manifest": "manifest.json", "report": "capture.json"},
        "scope": "Single configured scene with exact simulator poses and masks; no locomotion, pose estimation, tactile sensing or hidden-depth completion.",
    }
    (out / "capture.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
