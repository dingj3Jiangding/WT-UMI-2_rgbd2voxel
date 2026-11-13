"""Geometry utilities with explicit camera/world conventions."""

from __future__ import annotations

import numpy as np


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth image into camera-frame points.

    Depth values are in metres. Invalid pixels are returned as NaNs.
    """

    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("depth must be (H, W) and intrinsics must be (3, 3)")

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")

    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(depth) & (depth > 0)

    points = np.full((height, width, 3), np.nan, dtype=np.float64)
    points[..., 2] = depth
    points[..., 0] = (u - cx) * depth / fx
    points[..., 1] = (v - cy) * depth / fy
    points[~valid] = np.nan
    return points


def transform_points(points: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """Transform points from camera coordinates to world coordinates.

    ``t_wc`` is a homogeneous ``(4, 4)`` camera-to-world transform.
    Invalid points containing NaNs remain invalid.
    """

    points = np.asarray(points, dtype=np.float64)
    t_wc = np.asarray(t_wc, dtype=np.float64)
    if points.shape[-1] != 3 or t_wc.shape != (4, 4):
        raise ValueError("points must end in 3 and t_wc must be (4, 4)")
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    result = np.full_like(flat, np.nan)
    if np.any(valid):
        homogeneous = np.concatenate(
            [flat[valid], np.ones((valid.sum(), 1), dtype=np.float64)], axis=1
        )
        result[valid] = (t_wc @ homogeneous.T).T[:, :3]
    return result.reshape(points.shape)


def voxel_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Convert world points to integer voxel indices."""

    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if points.shape[-1] != 3 or origin.shape != (3,):
        raise ValueError("points must end in 3 and origin must be (3,)")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return np.floor((points - origin) / voxel_size).astype(np.int64)
