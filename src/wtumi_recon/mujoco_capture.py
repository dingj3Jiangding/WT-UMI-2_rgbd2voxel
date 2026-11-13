"""MuJoCo camera capture into the portable RGB-D sequence format.

One named world-attached camera is repositioned at configured waypoints in a
static scene. No physics rollout, robot locomotion or camera-path collision
checking is implied. Importing this module does not initialize MuJoCo or GL.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import yaml

from .render import look_at_pose
from .sequence import SequenceWriter


_CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def _rigid_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("capture T_wc must be a finite homogeneous 4x4 pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError("capture T_wc rotation must be orthonormal with determinant +1")
    return pose


def opencv_pose_from_mujoco(position, rotation):
    """Convert MuJoCo camera axes (right/up/back) to OpenCV (right/down/front)."""
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3) @ _CV_TO_GL
    pose[:3, 3] = np.asarray(position, dtype=float)
    return _rigid_pose(pose)


def intrinsics_from_fovy(width, height, fovy_degrees):
    """Integer pixel-center calibration for MuJoCo's symmetric perspective view."""
    if any(type(v) is not int or v < 1 for v in (width, height)):
        raise ValueError("capture width/height must be positive integers")
    if not np.isfinite(fovy_degrees) or not 0 < fovy_degrees < 180:
        raise ValueError("camera fovy must be between 0 and 180 degrees")
    focal = height / (2 * np.tan(np.deg2rad(fovy_degrees) / 2))
    return np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]])


def load_capture_config(path):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("MuJoCo capture requires schema_version: 1")
    allowed = {"schema_version", "scene_name", "mjcf", "camera", "views"}
    if set(cfg) - allowed:
        raise ValueError(f"unsupported capture fields: {set(cfg) - allowed}")
    if not isinstance(cfg.get("scene_name"), str) or not cfg["scene_name"].strip():
        raise ValueError("capture scene_name must be a nonempty string")
    if not isinstance(cfg.get("mjcf"), str) or not cfg["mjcf"]:
        raise ValueError("capture mjcf must be a path")
    model_path = (path.parent / Path(cfg["mjcf"]).expanduser()).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    camera = cfg.get("camera")
    if not isinstance(camera, dict) or set(camera) != {"name", "width", "height", "depth_trunc_m"}:
        raise ValueError("capture camera requires name, width, height, depth_trunc_m")
    if not isinstance(camera["name"], str) or not camera["name"]:
        raise ValueError("capture camera name must be nonempty")
    intrinsics_from_fovy(camera["width"], camera["height"], 60)
    limit = camera["depth_trunc_m"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("capture depth_trunc_m must be finite and positive")
    views = cfg.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("capture views must be a nonempty list")
    normalized, ids = [], set()
    for view in views:
        if not isinstance(view, dict) or set(view) - {"id", "split", "timestamp_s", "eye", "target", "up", "T_wc"}:
            raise ValueError("invalid capture view fields")
        name = view.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) or name in ids:
            raise ValueError("capture frame IDs must be unique safe strings")
        ids.add(name)
        if view.get("split") not in {"integration", "held_out"}:
            raise ValueError("capture split must be integration or held_out")
        timestamp = view.get("timestamp_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("capture timestamp_s must be finite and nonnegative")
        if normalized and timestamp < normalized[-1]["timestamp_s"]:
            raise ValueError("capture timestamps must be nondecreasing")
        if "T_wc" in view:
            if any(key in view for key in ("eye", "target", "up")):
                raise ValueError("provide T_wc or eye/target/up, not both")
            pose = _rigid_pose(view["T_wc"])
        else:
            if not {"eye", "target"} <= set(view):
                raise ValueError("capture view requires T_wc or eye and target")
            pose = look_at_pose(view["eye"], view["target"], view.get("up", [0, 0, 1]))
        normalized.append({"id": name, "split": view["split"], "timestamp_s": float(timestamp), "T_wc": pose.tolist()})
    if {view["split"] for view in normalized} != {"integration", "held_out"}:
        raise ValueError("capture requires integration and held_out views")
    for view in normalized:
        if view["split"] == "held_out" and any(other["split"] == "integration" and np.allclose(view["T_wc"], other["T_wc"], atol=1e-8, rtol=0) for other in normalized):
            raise ValueError("held_out poses must differ from integration poses")
    cfg.update(mjcf=str(model_path), views=normalized, _path=str(path))
    return cfg


def set_camera_pose(mujoco, model, data, camera_id, T_wc):
    """Set a fixed world-body camera, then read the simulator's actual transform."""
    pose = _rigid_pose(T_wc)
    if model.cam_bodyid[camera_id] != 0 or int(model.cam_mode[camera_id]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        raise ValueError("capture camera must be fixed and attached directly to worldbody")
    gl_rotation = pose[:3, :3] @ _CV_TO_GL
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, gl_rotation.reshape(-1))
    model.cam_pos[camera_id] = pose[:3, 3]
    model.cam_quat[camera_id] = quaternion
    mujoco.mj_forward(model, data)
    actual = opencv_pose_from_mujoco(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
    if not np.allclose(actual, pose, atol=1e-8, rtol=0):
        raise RuntimeError("MuJoCo camera pose does not match requested OpenCV transform")
    return actual


def capture_mujoco(config_path, output_dir):
    """Capture a static waypoint sequence; return the saved manifest Path.

    Linux callers may select MUJOCO_GL=egl or osmesa before starting Python.
    This function does not change that environment variable or initialize a GUI.
    """
    cfg = load_capture_config(config_path)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"capture output must be new or empty: {out}")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed; install the project's simulation extra in this environment") from exc

    model = mujoco.MjModel.from_xml_path(cfg["mjcf"])
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg["camera"]["name"])
    if camera_id < 0:
        raise ValueError(f"MuJoCo model has no camera named {cfg['camera']['name']}")
    if hasattr(model, "cam_orthographic") and bool(model.cam_orthographic[camera_id]):
        raise ValueError("capture currently supports perspective cameras only")
    if hasattr(model, "cam_sensorsize") and np.any(model.cam_sensorsize[camera_id] != 0):
        raise ValueError("capture currently requires a fovy camera, not custom sensor/focal parameters")
    width, height = cfg["camera"]["width"], cfg["camera"]["height"]
    k = intrinsics_from_fovy(width, height, float(model.cam_fovy[camera_id]))
    camera = {"width": width, "height": height, "K": k.tolist(), "depth_trunc_m": float(cfg["camera"]["depth_trunc_m"])}
    near = float(model.vis.map.znear * model.stat.extent)
    far = float(model.vis.map.zfar * model.stat.extent)
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("invalid MuJoCo clipping planes")
    # Avoid intentional subpixel depth mixing. Finite-precision rasterization
    # can still disagree with analytic rays exactly at a silhouette boundary.
    model.vis.quality.offsamples = 0
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    metadata = {
        "producer": "wtumi_recon.mujoco_capture", "mujoco_version": mujoco.__version__,
        "world_up": [0, 0, 1], "world_axes": "MuJoCo world, z up", "world_unit": "metre",
        "source_mjcf_sha256": hashlib.sha256(Path(cfg["mjcf"]).read_bytes()).hexdigest(),
        "source_mjcf": cfg["mjcf"], "camera_name": cfg["camera"]["name"],
        "fovy_degrees": float(model.cam_fovy[camera_id]), "near_clip_m": near, "far_clip_m": far,
        "offscreen_multisample_count": 0,
        "mujoco_gl_requested": os.environ.get("MUJOCO_GL", "platform_default"),
        "camera_conversion": "T_wc rotation = data.cam_xmat @ diag(1,-1,-1); position = data.cam_xpos",
        "depth_source": "mujoco.Renderer depth rendering, metric camera-z; invalid/far/out-of-range values zeroed",
        "capture_motion": "Static scene, one fixed world camera repositioned at waypoints; timestamps are capture schedule, no physics stepping",
        "rgb_model": "Simulator RGB with multisampling/shadows/reflections disabled; reconstruction uses fused vertex colors, not a material/lighting renderer",
    }
    records = []
    # Test GL initialization before creating the output dataset.
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        writer = SequenceWriter(out, camera=camera, scene_name=cfg["scene_name"],
                                reference_kind="simulator_ground_truth", metadata=metadata)
        for view in cfg["views"]:
            actual = set_camera_pose(mujoco, model, data, camera_id, view["T_wc"])
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=cfg["camera"]["name"])
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            valid = np.isfinite(depth) & (depth > 0) & (depth < camera["depth_trunc_m"]) & (depth < far - max(1e-6, far*1e-5))
            depth[~valid] = 0
            rgb[~valid] = 0
            if not valid.any():
                raise ValueError(f"capture frame {view['id']} contains no valid depth")
            writer.add_frame(view["id"], view["split"], rgb, depth, actual, view["timestamp_s"])
            records.append({"id": view["id"], "split": view["split"], "timestamp_s": view["timestamp_s"],
                            "valid_depth_pixels": int(valid.sum()), "depth_min_m": float(depth[valid].min()),
                            "depth_max_m": float(depth[valid].max())})
        manifest = writer.finish()
    (out / "capture_source.yaml").write_text(Path(cfg["_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "capture_resolved.yaml").write_text(yaml.safe_dump({key: value for key, value in cfg.items() if key != "_path"}, sort_keys=False), encoding="utf-8")
    (out / "model_source.xml").write_text(Path(cfg["mjcf"]).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "capture.json").write_text(json.dumps({"manifest": "manifest.json", "scene_name": cfg["scene_name"],
        "frames": records, "metadata": metadata,
        "source_snapshot_scope": "MJCF text only; externally referenced assets are not copied"}, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest
