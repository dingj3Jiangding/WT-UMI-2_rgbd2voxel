"""Configuration loading and validation for scene-agnostic runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import numpy as np


REQUIRED_SECTIONS = ("scene", "camera", "reconstruction", "evaluation")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve its scene root relative to the file."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"configuration is missing sections: {', '.join(missing)}")

    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    scene_root = Path(config["scene"].get("root", "."))
    if not scene_root.is_absolute():
        scene_root = (path.parent / scene_root).resolve()
    config["scene"]["root"] = str(scene_root)
    config["_path"] = str(path)
    return config


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Validate the M2 synthetic-scene schema and resolve all asset paths.

    Relative paths are relative to the YAML file, never the shell cwd.
    Integration and held-out views are explicit, disjoint by ID and pose.
    """
    from .render import look_at_pose
    from .tsdf import TSDFConfig
    from .sensor import normalize_sensor_config

    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    if raw.get("schema_version") == 3:
        from .sequence import RGBDSequence
        for required in ("source", "reconstruction", "evaluation"):
            if required not in raw:
                raise ValueError(f"sequence run config is missing {required}")
        source = raw["source"]
        if not isinstance(source, dict) or source.get("type") != "rgbd_sequence" or set(source) != {"type", "manifest"}:
            raise ValueError("schema 3 source requires type: rgbd_sequence and manifest")
        if not isinstance(source["manifest"], str):
            raise ValueError("source.manifest must be a path string")
        manifest = (path.parent / Path(source["manifest"]).expanduser()).resolve()
        sequence = RGBDSequence(manifest)
        derived = {"scene": {"name": sequence.scene_name, "type": "rgbd_sequence"},
                   "camera": sequence.camera, "views": sequence.views}
        for key, value in derived.items():
            if key in raw and raw[key] != value:
                raise ValueError(f"sequence {key} is defined by its manifest and cannot be overridden")
        cfg = {**raw, **derived, "source": {"type": "rgbd_sequence", "manifest": str(manifest)}, "_path": str(path)}
        cfg["world_up"] = sequence.metadata.get("world_up", [0, -1, 0])
    elif raw.get("schema_version") == 2:
        cfg = load_config(path)
        if cfg.get("source", {"type": "synthetic"}) != {"type": "synthetic"}:
            raise ValueError("schema 2 uses synthetic geometry; use schema 3 for RGB-D sequences")
        cfg["source"] = {"type": "synthetic"}
        if not isinstance(cfg["scene"].get("objects"), list) or not cfg["scene"]["objects"]:
            raise ValueError("scene.objects must be a non-empty list")
        cfg.setdefault("world_up", [0, -1, 0])
    else:
        raise ValueError("run config requires schema_version 2 (synthetic) or 3 (RGB-D sequence)")
    cfg["sensor"] = normalize_sensor_config(cfg.get("sensor"))
    world_up = np.asarray(cfg["world_up"], dtype=float)
    if world_up.shape != (3,) or not np.isfinite(world_up).all() or np.linalg.norm(world_up) < 1e-8:
        raise ValueError("world_up must be a finite nonzero vector")
    cfg["world_up"] = (world_up / np.linalg.norm(world_up)).tolist()
    for obj in cfg["scene"].get("objects", []):
        if not isinstance(obj, dict):
            raise ValueError("each scene object must be a mapping")
        if obj.get("type") == "mesh":
            asset = Path(obj["path"]).expanduser()
            obj["path"] = str((Path(cfg["_path"]).parent / asset).resolve())

    cam = cfg["camera"]
    if cam.get("convention", "opencv") != "opencv":
        raise ValueError("M2 camera convention must be opencv")
    for key in ("width", "height"):
        if not isinstance(cam.get(key), int) or cam[key] <= 0:
            raise ValueError(f"camera.{key} must be a positive integer")
    for key in ("fx", "fy", "depth_trunc_m"):
        if not np.isfinite(cam.get(key, np.nan)) or cam[key] <= 0:
            raise ValueError(f"camera.{key} must be finite and positive")
    for key in ("cx", "cy"):
        if not np.isfinite(cam.get(key, np.nan)):
            raise ValueError(f"camera.{key} must be finite")
    if "depth_scale" in cam:
        raise ValueError("M2 synthetic depths are float metres; do not set depth_scale")
    cam["K"] = [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]
    rec = cfg["reconstruction"]
    backend = rec.get("backend", "uniform")
    allowed = {"backend", "sdf_trunc_m", "volume_length_m", "resolution", "origin_m"} if backend == "uniform" else {"backend", "sdf_trunc_m", "voxel_size_m"}
    if set(rec) - allowed:
        raise ValueError(f"unsupported reconstruction fields for {backend}: {set(rec) - allowed}")
    settings = TSDFConfig(**rec)
    cfg["effective_voxel_size_m"] = settings.effective_voxel_size_m
    cfg["occupancy"] = normalize_occupancy_config(cfg.get("occupancy"), settings.backend)

    views = cfg.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("views must be a non-empty list")
    ids = set()
    for view in views:
        name = view.get("id", "")
        if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name) or name in ids:
            raise ValueError("view IDs must be unique, non-empty alphanumeric/underscore/hyphen strings")
        ids.add(name)
        if view.get("split") not in {"integration", "held_out"}:
            raise ValueError("view.split must be integration or held_out")
        if "T_wc" in view:
            if "eye" in view or "target" in view:
                raise ValueError("supply either T_wc or eye/target for a view")
            pose = np.asarray(view["T_wc"], dtype=float)
        else:
            pose = look_at_pose(view["eye"], view["target"], view.get("up", [0, -1, 0]))
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("T_wc must be a finite homogeneous 4x4 pose")
        rot = pose[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rot), 1):
            raise ValueError("T_wc rotation must be orthonormal with determinant +1")
        view["T_wc"] = pose.tolist()
        for key in ("eye", "target", "up"):
            view.pop(key, None)
    fused = [v for v in views if v["split"] == "integration"]
    held = [v for v in views if v["split"] == "held_out"]
    if not fused or not held:
        raise ValueError("M2 needs at least one integration view and one held_out view")
    for v in held:
        if any(np.allclose(v["T_wc"], t["T_wc"], atol=1e-8, rtol=0) for t in fused):
            raise ValueError("held_out poses must differ from integration poses")
    tolerance = cfg["evaluation"].get("depth_tolerance_m", 0.02)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("evaluation.depth_tolerance_m must be positive")
    cfg["evaluation"]["depth_tolerance_m"] = tolerance
    return cfg


def normalize_occupancy_config(raw=None, backend="uniform"):
    """Optional bounded evidence grid sharing the uniform TSDF world bounds."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("occupancy must be a mapping")
    allowed = {"enabled", "resolution", "pixel_stride", "probes"}
    if set(raw) - allowed:
        raise ValueError(f"unsupported occupancy fields: {set(raw) - allowed}")
    cfg = {"enabled": False, "resolution": 80, "pixel_stride": 1, "probes": [], **raw}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("occupancy.enabled must be boolean")
    if cfg["enabled"] and backend != "uniform":
        raise ValueError("occupancy currently requires uniform reconstruction bounds")
    for name, minimum in (("resolution", 2), ("pixel_stride", 1)):
        if type(cfg[name]) is not int or cfg[name] < minimum:
            raise ValueError(f"occupancy.{name} must be an integer >= {minimum}")
    if not isinstance(cfg["probes"], list):
        raise ValueError("occupancy.probes must be a list")
    names = set()
    for probe in cfg["probes"]:
        if not isinstance(probe, dict) or set(probe) != {"id", "point_m"}:
            raise ValueError("each occupancy probe requires id and point_m")
        name = probe["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ValueError("occupancy probe IDs must be unique safe alphanumeric/underscore/hyphen strings")
        names.add(name)
        point = np.asarray(probe["point_m"], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("occupancy probe point_m must contain three finite world coordinates")
        probe["point_m"] = point.tolist()
    return cfg
