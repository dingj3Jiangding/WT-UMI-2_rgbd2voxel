"""Portable, simulator-independent RGB-D sequences with explicit geometry units.

Images are lazy-loaded: constructing a sequence validates metadata and paths,
without reading held-out RGB/depth arrays into reconstruction code.
"""

from __future__ import annotations

import copy
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np
from PIL import Image


SEQUENCE_FORMAT = "wtumi-rgbd-sequence-v1"
CONVENTIONS = {
    "format": SEQUENCE_FORMAT,
    "depth_unit": "metre",
    "depth_definition": "camera_z",
    "camera_convention": "opencv",
    "pixel_centers": "integer",
}
REFERENCE_KINDS = {"simulator_ground_truth", "observed_rgbd"}


def normalize_camera(camera):
    """Return one fixed, zero-skew OpenCV pinhole calibration."""
    if not isinstance(camera, dict):
        raise ValueError("sequence camera must be a mapping")
    for name in ("width", "height"):
        value = camera.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"sequence camera.{name} must be a positive integer")
    k = np.asarray(camera.get("K"), dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("sequence camera.K must be a finite 3x3 matrix")
    if not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-10) or not np.allclose([k[0, 1], k[1, 0]], 0, rtol=0, atol=1e-10):
        raise ValueError("sequence camera.K must be zero-skew pinhole calibration")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("sequence camera focal lengths must be positive")
    limit = camera.get("depth_trunc_m")
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, Real) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("sequence camera.depth_trunc_m must be finite and positive")
    result = {
        "width": int(camera["width"]), "height": int(camera["height"]),
        "K": k.tolist(), "fx": float(k[0, 0]), "fy": float(k[1, 1]),
        "cx": float(k[0, 2]), "cy": float(k[1, 2]),
        "depth_trunc_m": float(limit), "convention": "opencv",
    }
    for name in ("fx", "fy", "cx", "cy"):
        if name in camera:
            value = camera[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not np.isclose(value, result[name], rtol=0, atol=1e-8):
                raise ValueError(f"sequence camera.{name} conflicts with K")
    if camera.get("convention", "opencv") != "opencv":
        raise ValueError("sequence camera convention must be opencv")
    return result


def _frame_geometry(frame):
    if not isinstance(frame, dict):
        raise ValueError("sequence frame must be a mapping")
    name = frame.get("id")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
        raise ValueError("frame IDs must be safe alphanumeric/underscore/hyphen strings")
    split = frame.get("split")
    if split not in {"integration", "held_out"}:
        raise ValueError("frame split must be integration or held_out")
    pose = np.asarray(frame.get("T_wc"), dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("frame T_wc must be a finite homogeneous 4x4 pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6):
        raise ValueError("frame T_wc rotation must be orthonormal with determinant +1")
    timestamp = frame.get("timestamp_s")
    if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, Real) or not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("frame timestamp_s must be finite and nonnegative")
    return {"id": name, "split": split, "T_wc": pose.tolist(), "timestamp_s": float(timestamp)}


def _check_new_view(view, existing):
    if any(v["id"] == view["id"] for v in existing):
        raise ValueError(f"duplicate frame ID: {view['id']}")
    if existing and view["timestamp_s"] < existing[-1]["timestamp_s"]:
        raise ValueError("frame timestamps must be nondecreasing in capture order")
    if any(v["split"] != view["split"] and np.allclose(v["T_wc"], view["T_wc"], rtol=0, atol=1e-8) for v in existing):
        raise ValueError("held_out poses must differ from integration poses")


def _check_complete(views):
    if not views or {view["split"] for view in views} != {"integration", "held_out"}:
        raise ValueError("sequence requires at least one integration and one held_out frame")


def _pair_arrays(rgb, depth_m, camera):
    rgb, depth = np.asarray(rgb), np.asarray(depth_m)
    shape = (camera["height"], camera["width"])
    if rgb.shape != shape + (3,) or rgb.dtype != np.uint8:
        raise ValueError("sequence RGB must be aligned HxWx3 uint8")
    if depth.shape != shape or depth.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError("sequence depth must be aligned HxW float32/float64 metres")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("sequence depth must be finite and nonnegative; use zero for invalid depth")
    with np.errstate(over='ignore'):
        depth32 = np.array(depth, dtype=np.float32, copy=True)
    if not np.isfinite(depth32).all():
        raise ValueError("sequence depth cannot be represented as finite float32 metres")
    return np.array(rgb, copy=True), depth32


def _within_dataset(root, value, suffix):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("sequence frame paths must be relative to the manifest")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"sequence frame path escapes the dataset: {value}")
    if path.suffix.lower() != suffix:
        raise ValueError(f"sequence frame path must end in {suffix}: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class RGBDSequence:
    """Validated portable sequence; frame image contents are loaded on demand.

    A missing reference_kind is treated as observed RGB-D, never as ground
    truth. Explicit reference files can differ from integration observations.
    """

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("sequence manifest must be a mapping")
        for key, expected in CONVENTIONS.items():
            if raw.get(key) != expected:
                raise ValueError(f"unsupported sequence {key}: expected {expected}")
        self.scene_name = raw.get("scene_name")
        if not isinstance(self.scene_name, str) or not self.scene_name.strip():
            raise ValueError("sequence scene_name must be a nonempty string")
        self.reference_kind = raw.get("reference_kind", "observed_rgbd")
        if self.reference_kind not in REFERENCE_KINDS:
            raise ValueError("sequence reference_kind must be simulator_ground_truth or observed_rgbd")
        self.camera = normalize_camera(raw.get("camera"))
        self.metadata = raw.get("metadata", {})
        if not isinstance(self.metadata, dict):
            raise ValueError("sequence metadata must be a mapping")
        json.dumps(self.metadata, allow_nan=False)
        frames = raw.get("frames")
        if not isinstance(frames, list):
            raise ValueError("sequence frames must be a list")
        self.views, self._records = [], {}
        for frame in frames:
            view = _frame_geometry(frame)
            _check_new_view(view, self.views)
            rgb_path = _within_dataset(self.root, frame.get("rgb"), ".png")
            depth_path = _within_dataset(self.root, frame.get("depth_m"), ".npy")
            has_rgb, has_depth = "reference_rgb" in frame, "reference_depth_m" in frame
            if has_rgb != has_depth:
                raise ValueError("reference_rgb and reference_depth_m must be supplied together")
            if has_rgb:
                ref_rgb = _within_dataset(self.root, frame["reference_rgb"], ".png")
                ref_depth = _within_dataset(self.root, frame["reference_depth_m"], ".npy")
            else:
                ref_rgb, ref_depth = rgb_path, depth_path
            self._records[view["id"]] = {"rgb": rgb_path, "depth_m": depth_path,
                                          "reference_rgb": ref_rgb, "reference_depth_m": ref_depth}
            self.views.append(view)
        _check_complete(self.views)
        self.manifest = copy.deepcopy(raw)

    def _record(self, view_or_id):
        name = view_or_id.get("id") if isinstance(view_or_id, dict) else view_or_id
        if not isinstance(name, str) or name not in self._records:
            raise KeyError(f"unknown sequence frame: {name!r}")
        return self._records[name]

    def _read_pair(self, rgb_path, depth_path):
        with Image.open(rgb_path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("sequence RGB file must be an 8-bit RGB PNG")
            rgb = np.asarray(image).copy()
        depth = np.load(depth_path, allow_pickle=False)
        if not isinstance(depth, np.ndarray):
            depth.close()
            raise ValueError("sequence depth file must contain one NPY array")
        return _pair_arrays(rgb, depth, self.camera)

    def read(self, view_or_id):
        frame = self._record(view_or_id)
        return self._read_pair(frame["rgb"], frame["depth_m"])

    def read_reference(self, view_or_id):
        frame = self._record(view_or_id)
        return self._read_pair(frame["reference_rgb"], frame["reference_depth_m"])


class SequenceWriter:
    """Write a fresh RGB-D dataset; no camera-specific dependency is required."""

    def __init__(self, output_dir, camera, scene_name, reference_kind, metadata=None):
        self.camera = normalize_camera(camera)
        if not isinstance(scene_name, str) or not scene_name.strip():
            raise ValueError("sequence scene_name must be a nonempty string")
        if reference_kind not in REFERENCE_KINDS:
            raise ValueError("sequence reference_kind must be simulator_ground_truth or observed_rgbd")
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict):
            raise ValueError("sequence metadata must be a mapping")
        json.dumps(metadata, allow_nan=False)
        self.scene_name, self.reference_kind = scene_name, reference_kind
        self.metadata = copy.deepcopy(metadata)
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists() and (not self.output_dir.is_dir() or any(self.output_dir.iterdir())):
            raise FileExistsError(f"sequence output must be new or empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames, self._closed = [], False

    def add_frame(self, id, split, rgb, depth_m, T_wc, timestamp_s,
                  reference_rgb=None, reference_depth_m=None):
        if self._closed:
            raise RuntimeError("sequence writer is already finished")
        view = _frame_geometry({"id": id, "split": split, "T_wc": T_wc, "timestamp_s": timestamp_s})
        _check_new_view(view, self._frames)
        rgb, depth = _pair_arrays(rgb, depth_m, self.camera)
        has_rgb, has_depth = reference_rgb is not None, reference_depth_m is not None
        if has_rgb != has_depth:
            raise ValueError("reference_rgb and reference_depth_m must be supplied together")
        reference = _pair_arrays(reference_rgb, reference_depth_m, self.camera) if has_rgb else None
        directory = self.output_dir / "frames" / id
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {"rgb": ("rgb.png", rgb), "depth_m": ("depth_m.npy", depth)}
        if reference is not None:
            arrays.update(reference_rgb=("reference_rgb.png", reference[0]),
                          reference_depth_m=("reference_depth_m.npy", reference[1]))
        record = dict(view)
        for key, (name, data) in arrays.items():
            path = directory / name
            with path.open("xb") as handle:
                if name.endswith(".png"):
                    Image.fromarray(data).save(handle, format="PNG")
                else:
                    np.save(handle, data, allow_pickle=False)
            record[key] = path.relative_to(self.output_dir).as_posix()
        self._frames.append(record)

    def finish(self):
        if self._closed:
            raise RuntimeError("sequence writer is already finished")
        _check_complete(self._frames)
        camera = {k: self.camera[k] for k in ("width", "height", "K", "depth_trunc_m")}
        manifest = {**CONVENTIONS, "scene_name": self.scene_name, "reference_kind": self.reference_kind,
                    "camera": camera, "metadata": self.metadata, "frames": self._frames}
        path = self.output_dir / "manifest.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        self._closed = True
        return path
