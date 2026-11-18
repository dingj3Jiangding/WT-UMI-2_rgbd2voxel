"""Portable scene bundles for the generic MuJoCo RGB-D workflow.

Geometry dimensions are full sizes in metres. World +Z is up and the
kinematic camera rig faces +Y. Group 3 is reserved for reconstruction overlays.
"""

from __future__ import annotations

import copy
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .config import normalize_occupancy_config
from .mujoco_capture import intrinsics_from_fovy
from .render import look_at_pose
from .rig_capture import _number, _vector, head_camera_pose, rig_transform
from .scene_doors import append_scene_doors, normalize_doors


TEMPLATES = ("room", "stairs", "doorway", "g1_stairs")
BUILTIN_SCENES = TEMPLATES
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")
_RESERVED_NAMES = {"sensor_rig", "head_camera", "evaluation_camera", "torso", "box"}
_DEFAULTS = {
    "camera": {"width": 320, "height": 240, "fovy_degrees": 65,
               "depth_trunc_m": 6, "position_in_rig_m": [0, 0, 1.6],
               "target_in_rig_m": [0, 2, .65]},
    "torso": {"enabled": True, "size_m": [.4, .3, .75],
              "position_in_rig_m": [0, 0, .8]},
    "box": {"enabled": False, "size_m": [.7, .6, .6],
            "position_in_rig_m": [0, .55, .85]},
    "reconstruction": {"backend": "uniform", "volume_length_m": 3.5,
                       "origin_m": [-1.75, -.5, -.15], "resolution": 160,
                       "sdf_trunc_m": .08},
    "occupancy": {"enabled": True, "resolution": 64, "pixel_stride": 4},
    "evaluation": {"depth_tolerance_m": .03},
    "playback": {"fps": 3},
}


class _SceneDumper(yaml.SafeDumper):
    """Keep short numeric vectors on one line for straightforward scene edits."""


def _list_yaml(dumper, values):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", values,
                                     flow_style=all(not isinstance(v, (dict, list)) for v in values))


_SceneDumper.add_representer(list, _list_yaml)


def _dump_yaml(config):
    return yaml.dump(config, Dumper=_SceneDumper, sort_keys=False)


def _mapping(value, allowed, name, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError(f"invalid {name} fields; allowed: {', '.join(sorted(allowed))}; "
                         f"required: {', '.join(sorted(required))}")


def _name(value, name):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{name} must be a safe alphanumeric/underscore/hyphen string")
    return value


def _resolve_ref(ref):
    candidate = Path(ref).expanduser()
    if str(ref) in TEMPLATES and not candidate.exists():
        candidate = Path(__file__).resolve().parents[2] / "scenes" / str(ref)
    if candidate.is_dir():
        candidate /= "scene.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"scene config not found: {candidate}")
    return candidate.resolve()


def _mesh_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("mesh.path must be a nonempty relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("mesh.path must remain inside the scene bundle")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("mesh.path must name a file inside the scene bundle")
    if path.suffix.lower() not in {".obj", ".stl", ".msh"}:
        raise ValueError("mesh.path requires a MuJoCo-supported .obj, .stl or .msh file")
    return path


def resolve_robot_assets(cfg):
    """Locate bundled or portable G1 assets without escaping a scene bundle."""
    robot = cfg.get("robot")
    if robot is None:
        return None
    if "asset_dir" not in robot:
        path = Path(__file__).resolve().parents[2] / "assets/robots/unitree_g1"
    else:
        value = robot["asset_dir"]
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("robot.asset_dir must be a relative directory inside the scene bundle")
        root = Path(cfg["_root"]).resolve()
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("robot.asset_dir must remain inside the scene bundle")
    if not path.is_dir() or not (path / "g1.xml").is_file():
        raise ValueError(f"G1 model assets are missing: {path}")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("robot asset bundle must not contain symlinks")
    return path


def _copy_robot_assets(cfg, portable, target):
    source = resolve_robot_assets(cfg)
    if source is not None:
        relative = "robot_assets/unitree_g1"
        shutil.copytree(source, target / relative)
        portable["robot"]["asset_dir"] = relative


def load_scene_config(ref):
    """Load a builtin name, bundle directory or scene YAML without starting GL."""
    path = _resolve_ref(ref)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "scene_name", "environment", "trajectory", "held_out"}
    _mapping(cfg, required | set(_DEFAULTS) | {"robot"}, "scene", required)
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("scene requires schema_version: 1")
    _name(cfg["scene_name"], "scene_name")
    for key, defaults in _DEFAULTS.items():
        value = cfg.get(key, {})
        allowed = set(defaults) | ({"probes"} if key == "occupancy" else set())
        _mapping(value, allowed, key)
        cfg[key] = {**copy.deepcopy(defaults), **value}

    _mapping(cfg["environment"], {"objects", "doors"}, "environment", {"objects"})
    objects = cfg["environment"]["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 500:
        raise ValueError("environment.objects must contain between 1 and 500 objects")
    names = set(_RESERVED_NAMES)
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("each environment object must be a mapping")
        kind = obj.get("type")
        dimensions = {"box": {"size_m"}, "sphere": {"radius_m"},
                      "cylinder": {"radius_m", "height_m"}, "mesh": {"path"}}
        if kind not in dimensions:
            raise ValueError("object.type must be box, sphere, cylinder or mesh")
        required_obj = {"name", "type", "position_m"} | dimensions[kind]
        _mapping(obj, required_obj | {"color_rgba", "euler_degrees"} |
                 ({"scale"} if kind == "mesh" else set()), "object", required_obj)
        name = _name(obj["name"], "object.name")
        if name in names or name.startswith("door_"):
            raise ValueError(f"duplicate or reserved object name: {name}")
        names.add(name)
        obj["position_m"] = _vector(obj["position_m"], f"{name}.position_m")
        obj["euler_degrees"] = _vector(obj.get("euler_degrees", [0, 0, 0]), f"{name}.euler_degrees")
        rgba = obj.get("color_rgba", [.55, .6, .65, 1])
        if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
            raise ValueError("color_rgba must contain four values in [0, 1]")
        obj["color_rgba"] = [_number(v, "color_rgba") for v in rgba]
        if any(not 0 <= v <= 1 for v in obj["color_rgba"]):
            raise ValueError("color_rgba values must be in [0, 1]")
        if kind == "box":
            obj["size_m"] = _vector(obj["size_m"], f"{name}.size_m", True)
        for key in dimensions[kind] & {"radius_m", "height_m"}:
            obj[key] = _number(obj[key], f"{name}.{key}", True)
        if kind == "mesh":
            _mesh_path(path.parent, obj["path"])
            obj["scale"] = _vector(obj.get("scale", [1, 1, 1]), f"{name}.scale", True)

    cfg["environment"]["doors"] = normalize_doors(cfg["environment"].get("doors", []))

    camera = cfg["camera"]
    for key in ("width", "height"):
        if type(camera[key]) is not int or not 8 <= camera[key] <= 2048:
            raise ValueError(f"camera.{key} must be an integer in [8, 2048]")
    for key in ("fovy_degrees", "depth_trunc_m"):
        camera[key] = _number(camera[key], f"camera.{key}", True)
    intrinsics_from_fovy(camera["width"], camera["height"], camera["fovy_degrees"])
    for key in ("position_in_rig_m", "target_in_rig_m"):
        camera[key] = _vector(camera[key], f"camera.{key}")
    head_camera_pose(camera, {})
    for name in ("box", "torso"):
        if type(cfg[name]["enabled"]) is not bool:
            raise ValueError(f"{name}.enabled must be a boolean")
        cfg[name]["size_m"] = _vector(cfg[name]["size_m"], f"{name}.size_m", True)
        cfg[name]["position_in_rig_m"] = _vector(cfg[name]["position_in_rig_m"], f"{name}.position_in_rig_m")

    if "robot" in cfg:
        from .g1_model import normalize_g1_config
        if not isinstance(cfg["robot"], dict):
            raise ValueError("robot must be a mapping")
        asset_dir = cfg["robot"].get("asset_dir")
        has_assets = "asset_dir" in cfg["robot"]
        cfg["robot"] = normalize_g1_config({k: v for k, v in cfg["robot"].items() if k != "asset_dir"})
        if has_assets:
            cfg["robot"]["asset_dir"] = asset_dir
        if cfg["torso"]["enabled"]:
            raise ValueError("set torso.enabled: false when using a robot model")
        resolve_robot_assets({**cfg, "_root": str(path.parent)})

    ids, integration_poses, previous = set(), [], -np.inf
    for split in ("trajectory", "held_out"):
        if not isinstance(cfg[split], list) or not 1 <= len(cfg[split]) <= 240:
            raise ValueError(f"{split} must contain between 1 and 240 views")
        for frame in cfg[split]:
            keys = ({"id", "timestamp_s", "position_m", "yaw_degrees"} if split == "trajectory"
                    else {"id", "timestamp_s", "eye", "target", "up"})
            optional = {"head_yaw_degrees", "head_pitch_degrees"} if split == "trajectory" else set()
            _mapping(frame, keys | optional, split, keys)
            name = _name(frame["id"], "frame.id")
            if name in ids:
                raise ValueError("frame IDs must be unique across both splits")
            ids.add(name)
            timestamp = _number(frame["timestamp_s"], "timestamp_s")
            if timestamp < 0 or timestamp <= previous:
                raise ValueError("timestamps must be nonnegative and strictly increasing, including held-outs")
            frame["timestamp_s"], previous = timestamp, timestamp
            if split == "trajectory":
                frame["position_m"] = _vector(frame["position_m"], "trajectory.position_m")
                for key in ("yaw_degrees", "head_yaw_degrees", "head_pitch_degrees"):
                    frame[key] = _number(frame.get(key, 0), f"trajectory.{key}")
                integration_poses.append(rig_transform(frame["position_m"], frame["yaw_degrees"]) @
                                         head_camera_pose(camera, frame))
            else:
                for key in ("eye", "target", "up"):
                    frame[key] = _vector(frame[key], f"held_out.{key}")
                pose = look_at_pose(frame["eye"], frame["target"], frame["up"])
                if any(np.allclose(pose, p, atol=1e-8, rtol=0) for p in integration_poses):
                    raise ValueError("held-out poses must differ from all integration poses")

    rec = cfg["reconstruction"]
    if rec["backend"] != "uniform":
        raise ValueError("quick scenes currently require uniform reconstruction")
    if type(rec["resolution"]) is not int or not 8 <= rec["resolution"] <= 256:
        raise ValueError("reconstruction.resolution must be an integer in [8, 256]")
    rec["origin_m"] = _vector(rec["origin_m"], "reconstruction.origin_m")
    for key in ("volume_length_m", "sdf_trunc_m"):
        rec[key] = _number(rec[key], f"reconstruction.{key}", True)
    cfg["occupancy"] = normalize_occupancy_config(cfg["occupancy"], "uniform")
    if cfg["occupancy"]["resolution"] > 256:
        raise ValueError("occupancy.resolution must be <= 256")
    cfg["evaluation"]["depth_tolerance_m"] = _number(cfg["evaluation"]["depth_tolerance_m"], "evaluation.depth_tolerance_m", True)
    cfg["playback"]["fps"] = _number(cfg["playback"]["fps"], "playback.fps", True)
    if cfg["playback"]["fps"] > 120:
        raise ValueError("playback.fps must be <= 120")
    cfg["_path"], cfg["_root"] = str(path), str(path.parent)
    return cfg


def create_scene(template, destination):
    """Copy a validated scene bundle to a new directory without overwriting."""
    cfg = load_scene_config(template)
    source, target = Path(cfg["_root"]), Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"scene destination already exists: {target}")
    if target.is_relative_to(source):
        raise ValueError("scene destination must not be inside its source bundle")
    portable = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    copies = []
    for obj in portable["environment"]["objects"]:
        if obj["type"] == "mesh":
            original = source / obj["path"]
            if original.is_symlink() or any(p.is_symlink() for p in original.parents if p != source and p.is_relative_to(source)):
                raise ValueError("scene mesh assets must not contain symlinks")
            copies.append((_mesh_path(source, obj["path"]), target / obj["path"]))
    target.mkdir(parents=True)
    for original, copied in copies:
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copied)
    _copy_robot_assets(cfg, portable, target)
    (target / "scene.yaml").write_text(_dump_yaml(portable), encoding="utf-8")
    return {"format": "wtumi_scene_bundle_v1", "scene_name": cfg["scene_name"],
            "output": str(target), "paths": {"config": "scene.yaml"}}


def _values(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def build_scene_mjcf(cfg, *, robot_asset_root=None):
    """Create MJCF from a normalized config with bundle-relative mesh paths."""
    camera, initial = cfg["camera"], cfg["trajectory"][0]
    root = ET.Element("mujoco", model=cfg["scene_name"])
    ET.SubElement(root, "compiler", angle="degree", autolimits="true")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(max(128, camera["width"])), offheight=str(max(128, camera["height"])))
    ET.SubElement(visual, "map", znear="0.005", zfar="10")
    ET.SubElement(visual, "quality", offsamples="0")
    ET.SubElement(visual, "headlight", ambient="0.8 0.8 0.8", diffuse="0.2 0.2 0.2", specular="0 0 0")
    ET.SubElement(ET.SubElement(root, "default"), "geom", contype="0", conaffinity="0")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    for obj in cfg["environment"]["objects"]:
        attrs = {"name": obj["name"], "type": obj["type"], "group": "0",
                 "pos": _values(obj["position_m"]), "euler": _values(obj["euler_degrees"]),
                 "rgba": _values(obj["color_rgba"])}
        if obj["type"] == "box":
            attrs["size"] = _values(np.asarray(obj["size_m"]) / 2)
        elif obj["type"] == "sphere":
            attrs["size"] = _values([obj["radius_m"]])
        elif obj["type"] == "cylinder":
            attrs["size"] = _values([obj["radius_m"], obj["height_m"] / 2])
        else:
            attrs["mesh"] = f"asset_{obj['name']}"
            ET.SubElement(assets, "mesh", name=attrs["mesh"], file=obj["path"], scale=_values(obj["scale"]))
        ET.SubElement(world, "geom", **attrs)
    append_scene_doors(world, cfg["environment"].get("doors", []))
    evaluation = cfg["held_out"][0]
    ev = look_at_pose(evaluation["eye"], evaluation["target"], evaluation["up"])
    ev_gl = ev[:3, :3] @ np.diag([1., -1., -1.])
    ET.SubElement(world, "camera", name="evaluation_camera", mode="fixed",
                  pos=_values(ev[:3, 3]), xyaxes=_values(np.r_[ev_gl[:, 0], ev_gl[:, 1]]),
                  fovy=str(camera["fovy_degrees"]))
    rig = ET.SubElement(world, "body", name="sensor_rig", mocap="true",
                        pos=_values(initial["position_m"]), euler=_values([0, 0, initial["yaw_degrees"]]))
    local = head_camera_pose(camera, initial)
    gl = local[:3, :3] @ np.diag([1., -1., -1.])
    ET.SubElement(rig, "camera", name="head_camera", mode="fixed", pos=_values(local[:3, 3]),
                  xyaxes=_values(np.r_[gl[:, 0], gl[:, 1]]), fovy=str(camera["fovy_degrees"]))
    for name, group, color in (("torso", 1, [.2, .28, .38, 1]), ("box", 2, [.8, .48, .16, 1])):
        obj = cfg[name]
        if obj["enabled"]:
            ET.SubElement(rig, "geom", name=name, type="box", group=str(group),
                          pos=_values(obj["position_in_rig_m"]), size=_values(np.asarray(obj["size_m"]) / 2),
                          rgba=_values(color))
    if "robot" in cfg:
        from .g1_model import append_g1_visual
        source = robot_asset_root or resolve_robot_assets(cfg)
        prefix = cfg["robot"].get("asset_dir", "robot_assets/unitree_g1")
        append_g1_visual(root, rig, source, prefix,
                         {k: v for k, v in cfg["robot"].items() if k != "asset_dir"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def build_scene(ref, output_dir):
    """Build a portable MJCF and normalized YAML in a new or empty directory."""
    cfg = load_scene_config(ref)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"scene build output must be new or empty: {out}")
    portable = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    copies = []
    for obj in portable["environment"]["objects"]:
        if obj["type"] == "mesh":
            src = _mesh_path(Path(cfg["_root"]), obj["path"])
            relative = Path("assets") / (obj["name"] + src.suffix.lower())
            obj["path"] = relative.as_posix()
            copies.append((src, out / relative))
    robot_assets = resolve_robot_assets(cfg)
    if robot_assets is not None:
        portable["robot"]["asset_dir"] = "robot_assets/unitree_g1"
    xml = build_scene_mjcf(portable, robot_asset_root=robot_assets)
    out.mkdir(parents=True, exist_ok=True)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _copy_robot_assets(cfg, portable, out)
    (out / "scene.yaml").write_text(_dump_yaml(portable), encoding="utf-8")
    (out / "model.xml").write_text(xml, encoding="utf-8")
    return {"format": "wtumi_scene_bundle_v1", "scene_name": cfg["scene_name"],
            "output": str(out), "paths": {"model": "model.xml", "config": "scene.yaml"},
            "config": load_scene_config(out)}
