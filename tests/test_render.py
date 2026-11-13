import numpy as np
import open3d as o3d
import pytest

from wtumi_recon.render import look_at_pose, pinhole_rays, render_mesh_rgbd
from wtumi_recon.synthetic import SyntheticScene


def test_integer_pixel_rays_and_optical_axis_depth():
    k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
    rays = pinhole_rays(k, np.eye(4), 3, 3).numpy()
    np.testing.assert_allclose(rays[1, 1, 3:], [0, 0, 1])
    np.testing.assert_allclose(rays[1, 2, 3:], [0.5, 0, 1])
    mesh = o3d.geometry.TriangleMesh.create_box(4, 4, 0.1).translate([-2, -2, 2])
    _, depth = render_mesh_rgbd(mesh, k, np.eye(4), 3, 3)
    np.testing.assert_allclose(depth, 2, atol=1e-6)  # Not ray length at off-center pixels.


def test_rotated_camera_preserves_z_depth():
    pose = look_at_pose([0, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1)
    mesh = o3d.geometry.TriangleMesh.create_box(0.1, 4, 4).translate([2, -2, -2])
    k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
    _, depth = render_mesh_rgbd(mesh, k, pose, 3, 3)
    np.testing.assert_allclose(depth, 2, atol=1e-6)


def test_mesh_color_interpolation_and_color_changes():
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-1, -1, 2], [1, -1, 2], [0, 1, 2]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.eye(3))
    rgb, depth = render_mesh_rgbd(mesh, np.eye(3), np.eye(4), 1, 1)
    np.testing.assert_allclose(rgb[0, 0], [64, 64, 128], atol=1)
    mesh.paint_uniform_color([0, 1, 0])
    changed, changed_depth = render_mesh_rgbd(mesh, np.eye(3), np.eye(4), 1, 1)
    np.testing.assert_array_equal(changed[0, 0], [0, 255, 0])
    np.testing.assert_allclose(changed_depth, depth)


def test_configured_objects_translate_and_keep_distinct_colors():
    scene = SyntheticScene.from_config({"objects": [
        {"type": "box", "size_m": [0.5, 0.5, 0.5], "translation_m": [-0.7, -0.25, 1], "color_rgb": [255, 0, 0]},
        {"type": "sphere", "radius_m": 0.2, "translation_m": [0.4, 0, 1.2], "color_rgb": [0, 255, 0]}]})
    k = np.array([[40, 0, 31.5], [0, 40, 23.5], [0, 0, 1]], dtype=float)
    rgb, depth = scene.render(k, np.eye(4), 64, 48)
    assert np.any(np.all(rgb == [255, 0, 0], axis=-1))
    assert np.any(np.all(rgb == [0, 255, 0], axis=-1))
    assert (depth > 0).any()
    with pytest.raises(ValueError, match="parallel"):
        look_at_pose([0, 0, 0], [0, 1, 0], [0, 1, 0])
