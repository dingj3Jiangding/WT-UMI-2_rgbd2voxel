"""Static synthetic scenes made from configurable primitives or colored meshes."""

from pathlib import Path

import numpy as np
import open3d as o3d

from .render import MeshRGBDRenderer


def vector3(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


class SyntheticScene:
    def __init__(self, mesh):
        self.mesh = mesh
        self._renderer = MeshRGBDRenderer(mesh)

    @classmethod
    def from_config(cls, scene):
        objects = scene.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("scene.objects must be a non-empty list")
        combined = o3d.geometry.TriangleMesh()
        for item in objects:
            kind = item.get("type")
            if kind == "box":
                size = vector3(item.get("size_m"), "size_m")
                if np.any(size <= 0):
                    raise ValueError("box dimensions must be positive")
                mesh = o3d.geometry.TriangleMesh.create_box(*size)
            elif kind == "sphere":
                radius = float(item.get("radius_m", 0))
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("sphere radius_m must be positive")
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=24)
            elif kind == "mesh":
                path = Path(item["path"]).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(path)
                mesh = o3d.io.read_triangle_mesh(str(path))
                if not len(mesh.triangles):
                    raise ValueError(f"mesh has no triangles: {path}")
                if mesh.has_textures():
                    raise ValueError("textured meshes are unsupported; remove textures first, then use vertex colors or color_rgb")
            else:
                raise ValueError(f"unsupported object type: {kind}")
            mode = item.get("color_mode", "solid")
            if mode == "coordinate":
                vertices = np.asarray(mesh.vertices)
                lo, span = vertices.min(axis=0), np.ptp(vertices, axis=0)
                colors = 0.15 + 0.75 * (vertices - lo) / np.maximum(span, 1e-12)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif mode == "solid":
                if "color_rgb" in item or not mesh.has_vertex_colors():
                    color = vector3(item.get("color_rgb", [180, 180, 180]), "color_rgb")
                    if np.any((color < 0) | (color > 255)):
                        raise ValueError("color_rgb must be in [0,255]")
                    mesh.paint_uniform_color(color / 255)
            else:
                raise ValueError("color_mode must be solid or coordinate")
            angles = vector3(item.get("rotation_deg_xyz", [0, 0, 0]), "rotation_deg_xyz")
            mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad(angles)), center=[0, 0, 0])
            mesh.translate(vector3(item.get("translation_m", [0, 0, 0]), "translation_m"))
            combined += mesh
        return cls(combined)

    def render(self, intrinsics, t_wc, width, height):
        return self._renderer.render(intrinsics, t_wc, width, height)


class SyntheticCubeScene(SyntheticScene):
    """Compatibility wrapper for the original M1 script."""

    def __init__(self, *, size_m=0.5, minimum_corner=(0.25, 0.25, 1.0), color_rgb=(200, 50, 50)):
        scene = SyntheticScene.from_config({"objects": [{"type": "box", "size_m": [size_m]*3,
            "translation_m": minimum_corner, "color_rgb": color_rgb}]})
        super().__init__(scene.mesh)
        self.color_rgb = np.asarray(color_rgb, dtype=np.uint8)
