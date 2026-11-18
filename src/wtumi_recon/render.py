"""CPU pinhole ray casting: integer pixel centers, camera-z depth, mesh colors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import open3d as o3d


def look_at_pose(eye, target, up=(0, -1, 0)) -> np.ndarray:
    """Camera-to-world pose for OpenCV x-right, y-down, z-forward cameras."""
    eye, target, up = [np.asarray(v, dtype=float) for v in (eye, target, up)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (eye, target, up)):
        raise ValueError("eye, target, up must contain three finite values")
    forward = target - eye
    if np.linalg.norm(forward) < 1e-10:
        raise ValueError("eye and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-10:
        raise ValueError("up must not be parallel to view direction")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = eye
    return pose


def pinhole_rays(intrinsics, t_wc, width, height) -> o3d.core.Tensor:
    """Ray direction has camera z=1; t_hit is Z depth, not Euclidean range.

    Build rays explicitly because the Open3D helper uses half-pixel offsets,
    whereas back-projection and the TSDF baseline use integer pixel centers.
    """
    # Keep camera-only MuJoCo tools free of Open3D's bundled GLFW library.
    import open3d as o3d

    k, pose = np.asarray(intrinsics, dtype=float), np.asarray(t_wc, dtype=float)
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("intrinsics and t_wc must be finite 3x3 and 4x4 matrices")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("invalid pinhole intrinsics")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    rot = pose[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rot), 1) or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("t_wc must be a rigid camera-to-world pose")
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pixel = np.stack((u, v, np.ones_like(u)), axis=-1)
    directions = pixel @ np.linalg.inv(k).T @ rot.T
    origins = np.broadcast_to(pose[:3, 3], directions.shape)
    return o3d.core.Tensor(np.concatenate((origins, directions), axis=-1).astype(np.float32))


class MeshRGBDRenderer:
    """Cache one mesh acceleration structure; interpolate only its vertex colors."""

    def __init__(self, mesh):
        import open3d as o3d

        if not len(mesh.triangles):
            raise ValueError("cannot render a mesh without triangles")
        self.triangles = np.asarray(mesh.triangles).copy()
        self.colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else np.full((len(mesh.vertices), 3), 0.7)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(self, intrinsics, t_wc, width, height):
        hits = self.scene.cast_rays(pinhole_rays(intrinsics, t_wc, width, height))
        t = hits["t_hit"].numpy()
        valid = np.isfinite(t) & (t > 0)
        depth = np.where(valid, t, 0).astype(np.float32)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        if valid.any():
            ids = hits["primitive_ids"].numpy()[valid].astype(np.int64)
            uv = hits["primitive_uvs"].numpy()[valid]
            weights = np.column_stack((1 - uv.sum(axis=1), uv))
            colors = self.colors[self.triangles[ids]]
            interpolated = np.einsum("ni,nij->nj", weights, colors)
            rgb[valid] = np.rint(np.clip(interpolated, 0, 1) * 255).astype(np.uint8)
        return rgb, depth


def render_mesh_rgbd(mesh, intrinsics, t_wc, width, height):
    return MeshRGBDRenderer(mesh).render(intrinsics, t_wc, width, height)


def render_mesh_depth(mesh, intrinsics, t_wc, width, height):
    return render_mesh_rgbd(mesh, intrinsics, t_wc, width, height)[1]


def depth_to_preview(depth_m, max_depth_m=None):
    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0, posinf=0, neginf=0)
    if max_depth_m is not None:
        depth = np.minimum(depth, max_depth_m)
    return np.rint(np.clip(depth * 1000, 0, 65535)).astype(np.uint16)


def depth_to_visual_preview(depth_m, max_depth_m=3.0):
    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    image[valid] = np.clip((1 - depth[valid] / max_depth_m) * 255, 0, 255)
    return image
