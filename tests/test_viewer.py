"""Geometry and artifact checks for the offline reconstruction viewer."""

from html.parser import HTMLParser
import json

import numpy as np
import open3d as o3d
import pytest

from wtumi_recon.viewer import (
    SurfaceVoxels,
    camera_frustum_segments,
    export_viewer,
    load_surface_voxels,
    voxel_shell_mesh,
)


def _save_grid(path, *, tsdf=None, weight=None, color=None, **metadata):
    """An asymmetric 4^3 fixture; z is the fastest-changing flat axis."""
    values = dict(
        tsdf=np.zeros(64, dtype=np.float32) if tsdf is None else tsdf,
        weight=np.zeros(64, dtype=np.float32) if weight is None else weight,
        color=np.zeros((64, 3), dtype=np.float32) if color is None else color,
        resolution=4,
        voxel_size_m=0.5,
        sdf_trunc_m=1.0,
        origin_m=[-2.0, 3.0, 5.0],
        color_scale=255.0,
        tsdf_normalized=True,
        axis_order="xyz",
        flat_order="C",
    )
    values.update(metadata)
    np.savez_compressed(path, **values)


def test_flat_indices_map_to_world_centers_and_rgb(tmp_path):
    weight = np.zeros(64)
    weight[[1, 4, 16]] = 1
    colors = np.zeros((64, 3))
    colors[[1, 4, 16]] = [[255, 0, 0], [0, 128, 0], [0, 0, 255]]
    path = tmp_path / "grid.npz"
    _save_grid(path, weight=weight, color=colors)

    voxels = load_surface_voxels(path)

    np.testing.assert_array_equal(voxels.indices, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        voxels.centers,
        [[-1.75, 3.25, 5.75], [-1.75, 3.75, 5.25], [-1.25, 3.25, 5.25]],
    )
    np.testing.assert_allclose(voxels.colors, [[1, 0, 0], [0, 128 / 255, 0], [0, 0, 1]])
    assert voxels.observed_count == 3
    assert voxels.total_count == 64


def test_surface_band_excludes_unknown_low_weight_and_distant_cells(tmp_path):
    sdf = np.zeros(64)
    weight = np.zeros(64)
    # Index 0 has a zero TSDF but weight zero: it must remain unknown.
    weight[[1, 2, 3, 4, 16]] = [1, 0.5, 3, 1, 2]
    sdf[[1, 2, 3, 4, 16]] = [0.25, 0, np.nan, 0.75, -0.5]
    weight[5] = np.inf
    path = tmp_path / "grid.npz"
    _save_grid(path, tsdf=sdf, weight=weight)

    voxels = load_surface_voxels(path)
    np.testing.assert_array_equal(voxels.indices, [[0, 0, 1], [1, 0, 0]])
    assert voxels.observed_count == 5  # All finite positive weights, before filtering.
    np.testing.assert_array_equal(load_surface_voxels(path, min_weight=2).indices, [[1, 0, 0]])
    with pytest.raises(ValueError, match="exceeds max_voxels"):
        load_surface_voxels(path, max_voxels=1)
    with pytest.raises(ValueError, match="no observed voxels"):
        load_surface_voxels(path, min_weight=10)


@pytest.mark.parametrize("metadata", [{"axis_order": "zyx"}, {"flat_order": "F"}, {"tsdf_normalized": False}])
def test_incompatible_volume_conventions_are_rejected(tmp_path, metadata):
    path = tmp_path / "grid.npz"
    _save_grid(path, **metadata)
    with pytest.raises(ValueError, match="unsupported volume|expected normalized"):
        load_surface_voxels(path)


def _cells(indices):
    indices = np.asarray(indices)
    return SurfaceVoxels(
        indices=indices,
        colors=np.tile([0.2, 0.4, 0.6], (len(indices), 1)),
        origin=np.array([-2.0, 3.0, 5.0]),
        voxel_size=0.5,
        resolution=8,
        band_voxels=1,
        min_weight=1,
        observed_count=len(indices),
        total_count=512,
    )


def test_single_voxel_has_world_bounds_and_outward_faces():
    cells = _cells([[1, 2, 3]])
    vertices, triangles, colors = voxel_shell_mesh(cells)
    assert vertices.shape == (8, 3)
    assert triangles.shape == (12, 3)
    assert colors.shape == (12, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [-1.5, 4, 6.5])
    np.testing.assert_allclose(vertices.max(axis=0), [-1, 4.5, 7])
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners.mean(axis=1) - cells.centers[0]
    assert np.all(np.einsum("ij,ij->i", normals, outward) > 0)


def test_adjacent_voxels_remove_shared_faces_without_changing_bounds():
    vertices, triangles, _ = voxel_shell_mesh(_cells([[1, 2, 3], [2, 2, 3]]))
    assert vertices.shape == (12, 3)
    assert triangles.shape == (20, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [-1.5, 4, 6.5])
    np.testing.assert_allclose(vertices.max(axis=0), [-0.5, 4.5, 7])
    # The common face at x=-1 must not leave any internal triangles.
    assert not np.any(np.all(np.isclose(vertices[triangles, 0], -1), axis=1))


def test_camera_frustum_uses_integer_pixels_and_camera_to_world_pose():
    k = np.array([[2, 0, 1], [0, 4, 2], [0, 0, 1]], dtype=float)
    pose = np.array([[0, -1, 0, 10], [1, 0, 0, -3], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=float)
    segments = camera_frustum_segments(k, pose, width=3, height=5, depth=2)
    expected_corners = np.array([[11, -4, 7], [11, -2, 7], [9, -2, 7], [9, -4, 7]])
    assert segments.shape == (8, 2, 3)
    np.testing.assert_allclose(segments[:4, 0], np.tile([10, -3, 5], (4, 1)))
    np.testing.assert_allclose(segments[:4, 1], expected_corners)
    np.testing.assert_allclose(segments[4:, 0], expected_corners)
    np.testing.assert_allclose(segments[4:, 1], expected_corners[[1, 2, 3, 0]])


class _ViewerHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_scripts = []
        self.modes = {}
        self.payload = ""
        self.in_payload = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            if "src" in attrs:
                self.external_scripts.append(attrs["src"])
            self.in_payload = attrs.get("id") == "viewer-data"
        if tag == "button" and "data-mode" in attrs:
            self.modes[attrs["data-mode"]] = attrs

    def handle_data(self, data):
        if self.in_payload:
            self.payload += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False


def _write_result(root):
    root.mkdir()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[-1, 3, 6], [0, 3, 6], [-1, 4, 6]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    mesh.vertex_colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert o3d.io.write_triangle_mesh(str(root / "mesh.ply"), mesh)
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    cloud.colors = mesh.vertex_colors
    assert o3d.io.write_point_cloud(str(root / "pointcloud.ply"), cloud)
    weight = np.zeros(64)
    weight[[1, 4, 16]] = 1
    _save_grid(root / "tsdf_volume.npz", weight=weight)
    (root / "manifest.json").write_text(json.dumps({
        "K": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
        "width": 3, "height": 3,
        "frames": [
            {"id": "front", "split": "integration", "T_wc": np.eye(4).tolist()},
            {"id": "novel", "split": "held_out", "T_wc": np.eye(4).tolist()},
        ],
    }), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"scene": "test <scene>"}), encoding="utf-8")


def test_export_is_offline_and_preserves_source_reconstruction(tmp_path):
    root = tmp_path / "result"
    _write_result(root)
    original = {path.name: path.read_bytes() for path in root.iterdir()}

    output, metadata = export_viewer(root)

    assert output == root / "viewer.html"
    assert metadata["mesh_vertices"] == 3
    assert metadata["mesh_triangles"] == 1
    assert metadata["point_count"] == 3
    assert metadata["surface_voxels"] == 3
    assert metadata["voxel_size_m"] == 0.5
    assert [c["split"] for c in metadata["cameras"]] == ["integration", "held_out"]
    assert json.loads(output.with_suffix(".json").read_text()) == metadata
    page = output.read_text(encoding="utf-8")
    parsed = _ViewerHTML()
    parsed.feed(page)
    assert parsed.external_scripts == []
    assert set(parsed.modes) == {"0", "1", "2"}
    assert "disabled" not in parsed.modes["1"]
    assert "Plotly.newPlot" in page
    assert "plotly.js v" in page  # Actual bundled JS, not only a library call.
    assert json.loads(parsed.payload)["meta"] == metadata
    assert "test &lt;scene&gt;" in page
    for name, original_bytes in original.items():
        assert (root / name).read_bytes() == original_bytes


def test_legacy_result_without_volume_disables_voxels(tmp_path):
    root = tmp_path / "legacy"
    _write_result(root)
    (root / "tsdf_volume.npz").unlink()
    (root / "manifest.json").unlink()
    output, metadata = export_viewer(root, tmp_path / "export" / "legacy.html")
    parsed = _ViewerHTML()
    parsed.feed(output.read_text(encoding="utf-8"))
    assert "disabled" in parsed.modes["1"]
    assert metadata["surface_voxels"] is None
    assert metadata["cameras"] == []


def test_export_refuses_to_overwrite_source_sidecar_and_can_refresh_viewer(tmp_path):
    root = tmp_path / "result"
    _write_result(root)
    metrics_path = root / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        export_viewer(root, root / "metrics.html")
    assert metrics_path.read_bytes() == original_metrics
    assert not (root / "metrics.html").exists()

    output, first = export_viewer(root)
    same_output, refreshed = export_viewer(root)
    assert same_output == output
    assert first == refreshed
    assert first["format"] == "wtumi-recon-viewer-v1"


def test_saved_grid_agrees_with_open3d_world_coordinates(tmp_path):
    """Check the file convention against Open3D, not just our array fixture."""
    from wtumi_recon.tsdf import Open3DTSDFMap, TSDFConfig

    backend = Open3DTSDFMap(TSDFConfig(volume_length_m=2, resolution=4,
                                    sdf_trunc_m=1, origin_m=(-2, 3, 5)))
    values = np.zeros((64, 2))
    values[:, 0] = 1
    values[[1, 4, 16]] = [0.1, 1]
    backend._volume.inject_volume_tsdf(o3d.utility.Vector2dVector(values))
    native_points = np.asarray(backend.extract_voxel_point_cloud().points)
    assert len(native_points) == 3
    path = tmp_path / "native.npz"
    backend.save_volume(path)
    selected = load_surface_voxels(path)
    for center in selected.centers:
        assert np.any(np.all(np.isclose(native_points, center), axis=1))
    with np.load(path) as grid:
        assert grid['axis_order'].item() == 'xyz'
        assert grid['flat_order'].item() == 'C'
