"""RGB-D image loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_rgbd(
    rgb_path: str | Path,
    depth_path: str | Path,
    *,
    depth_scale: float = 1.0,
    depth_trunc_m: float = 3.0,
) -> o3d.geometry.RGBDImage:
    """Load an aligned RGB-D pair using the configured depth units."""

    color = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))
    if color.is_empty() or depth.is_empty():
        raise FileNotFoundError(f"could not read RGB-D pair: {rgb_path}, {depth_path}")
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def read_depth_array(path: str | Path, depth_scale: float = 1.0) -> np.ndarray:
    """Read depth as metres for geometry-level operations."""

    image = o3d.io.read_image(str(path))
    if image.is_empty():
        raise FileNotFoundError(path)
    return np.asarray(image).astype(np.float64) / depth_scale
