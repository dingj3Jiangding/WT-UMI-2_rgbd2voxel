"""Reproducible synthetic depth perturbations; not a physical camera model."""

from __future__ import annotations

import hashlib
from numbers import Real

import numpy as np


def normalize_sensor_config(raw=None):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("sensor must be a mapping")
    allowed = {"depth_noise_std_m", "dropout_probability", "seed"}
    if set(raw) - allowed:
        raise ValueError(f"unsupported sensor fields: {set(raw) - allowed}")
    result = {"depth_noise_std_m": 0.0, "dropout_probability": 0.0, "seed": 0, **raw}
    for name in ("depth_noise_std_m", "dropout_probability"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"sensor.{name} must be finite and nonnegative")
        result[name] = float(value)
    if result["dropout_probability"] > 1:
        raise ValueError("sensor.dropout_probability must be between 0 and 1")
    seed = result["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sensor.seed must be a nonnegative integer")
    return result


def corrupt_depth(depth_m, sensor, frame_id, depth_trunc_m):
    """Return a fresh float32 Z-depth image; invalid observations remain zero.

    Independent streams for Gaussian noise and Bernoulli missing pixels are
    stable across view order, voxel resolution and changes in noise amplitude.
    Only originally valid pixels can become observations. RGB is unchanged
    apart from masking by the caller where depth becomes invalid.
    """
    cfg = normalize_sensor_config(sensor)
    source = np.asarray(depth_m)
    if source.ndim != 2 or not isinstance(frame_id, str) or not frame_id:
        raise ValueError("depth must be 2D and frame_id a nonempty string")
    if not np.isfinite(depth_trunc_m) or depth_trunc_m <= 0:
        raise ValueError("depth_trunc_m must be finite and positive")
    depth = np.array(source, dtype=np.float32, copy=True)
    valid = np.isfinite(source) & (source > 0) & (source < depth_trunc_m)
    depth[~valid] = 0
    digest = hashlib.sha256(frame_id.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i+4], "little") for i in range(0, len(digest), 4)]
    noise_seed, drop_seed = np.random.SeedSequence([cfg["seed"], *words]).spawn(2)
    if cfg["depth_noise_std_m"]:
        noise = np.random.default_rng(noise_seed).normal(0, cfg["depth_noise_std_m"], depth.shape)
        depth[valid] += noise[valid].astype(np.float32)
    if cfg["dropout_probability"]:
        dropped = np.random.default_rng(drop_seed).random(depth.shape) < cfg["dropout_probability"]
        depth[dropped & valid] = 0
    depth[~np.isfinite(depth) | (depth <= 0) | (depth >= depth_trunc_m)] = 0
    return depth
