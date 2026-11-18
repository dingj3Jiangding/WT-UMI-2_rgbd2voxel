"""Analytic checks of exposed geometry, masks, and occluded evidence."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.occupancy import EvidenceOccupancyGrid, FREE, OCCUPIED, UNKNOWN
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.stair_metrics import evaluate_stair_result, occupancy_progress, sample_stair_surfaces


STAIRS = {"origin_m": [0, 0, 0], "width_m": 2, "tread_m": 1, "rise_m": 1, "steps": 1}
ROI = {"min_m": [-1, 0, 0], "max_m": [1, 1, 2]}


def _write_tread_mesh(path, include_riser=False):
    import open3d as o3d

    # Two triangles cover the exact top rectangle. The optional second pair
    # covers the exposed front rectangle, without adding interior block faces.
    vertices = [[-1, 0, 1], [1, 0, 1], [1, 1, 1], [-1, 1, 1], [-1, 0, 0], [1, 0, 0]]
    triangles = [[0, 1, 2], [0, 2, 3]]
    if include_riser:
        triangles += [[4, 5, 1], [4, 1, 0]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    assert o3d.io.write_triangle_mesh(str(path), mesh)


def _result(tmp_path, reference=None, prediction=None, mask=None, grid=None):
    root = tmp_path / "result"
    root.mkdir()
    ref = np.array([[1, 2, 3], [0, 4, 5]], dtype=np.float64) if reference is None else reference
    pred = np.array([[1.25, 0, 300], [7, 4.5, 500]], dtype=np.float64) if prediction is None else prediction
    for name, data in (("references", ref), ("renders", pred)):
        directory = root / name / "test_view"
        directory.mkdir(parents=True)
        np.save(directory / "depth_m.npy", data)
    mask_path = tmp_path / "mask.png"
    pixels = np.array([[255, 255, 0], [255, 255, 0]], dtype=np.uint8) if mask is None else mask
    Image.fromarray(pixels).save(mask_path)
    (root / "manifest.json").write_text(json.dumps({"frames": [{"id": "test_view", "split": "held_out"}]}))
    if grid is None:
        grid = EvidenceOccupancyGrid([-1, 0, 0], 1, 3)
    grid.save(root / "occupancy.npz")
    _write_tread_mesh(root / "mesh.ply")
    return root, {"test_view": str(mask_path)}


def test_surface_sampling_excludes_hidden_riser_portions_and_has_expected_area_density():
    stairs = {"origin_m": [3, 5, 7], "width_m": 2, "tread_m": 1, "rise_m": .5, "steps": 3}
    samples = sample_stair_surfaces(stairs, .25)
    # Surface area = 3 * width * (tread + rise) = 9 m2;
    # all extents divide .25 exactly, so each probe represents .0625 m2.
    assert samples.shape == (144, 3)
    assert len(np.unique(samples, axis=0)) == len(samples)
    assert len(samples) * .25**2 == 9
    np.testing.assert_array_equal(samples, sample_stair_surfaces(stairs, .25))
    offset = 0
    for step in range(3):
        tread, riser = samples[offset:offset+32], samples[offset+32:offset+48]
        assert (tread[:, 2] == 7+(step+1)*.5).all()
        assert ((tread[:, 1] > 5+step) & (tread[:, 1] < 6+step)).all()
        assert (riser[:, 1] == 5+step).all()
        assert ((riser[:, 2] > 7+step*.5) & (riser[:, 2] < 7+(step+1)*.5)).all()
        assert ((samples[:, 0] > 2) & (samples[:, 0] < 4)).all()
        offset += 48


@pytest.mark.parametrize("change,spacing", [({"rise_m": np.nan}, .04), ({"steps": True}, .04),
    ({"steps": 0}, .04), ({"origin_m": [0, np.inf, 0]}, .04), ({}, 0), ({}, 1e-12)])
def test_surface_sampling_rejects_invalid_or_unbounded_probes(change, spacing):
    with pytest.raises(ValueError):
        sample_stair_surfaces({**STAIRS, **change}, spacing)


def test_masked_depth_metrics_exclude_background_and_count_missing_inliers(tmp_path):
    root, masks = _result(tmp_path)
    report = evaluate_stair_result(root, STAIRS, ROI, masks, tolerance_m=.25, sample_spacing_m=.5)
    metrics = report["held_out_depth"]["pooled"]
    assert metrics["mask_pixels"] == 4
    assert metrics["reference_valid_pixels"] == 3
    assert metrics["prediction_valid_pixels"] == 3
    assert metrics["overlap_pixels"] == 2
    assert metrics["missing_pixels"] == metrics["extra_pixels"] == 1
    assert metrics["coverage"] == pytest.approx(2/3)
    assert metrics["mean_absolute_error_m"] == .375
    assert metrics["root_mean_square_error_m"] == pytest.approx(np.sqrt((.25**2+.5**2)/2))
    assert metrics["within_tolerance_pixels"] == 1  # Exact boundary is included.
    assert metrics["within_tolerance_fraction"] == pytest.approx(1/3)
    surface = report["surface_completeness"]
    # Half the probes lie on the tread. Four more are .25 below its edge;
    # the last four lie .75 below it. This is triangle, not vertex distance.
    assert surface["sample_count"] == 16
    assert surface["covered_sample_count"] == 12
    assert surface["coverage_within_tolerance_fraction"] == .75
    assert surface["mean_nearest_mesh_distance_m"] == .25
    assert surface["median_nearest_mesh_distance_m"] == .125
    assert report["ground_truth_used_for_fusion"] is False
    json.dumps(report, allow_nan=False)


def test_complete_tread_and_riser_mesh_covers_all_surface_samples(tmp_path):
    root, masks = _result(tmp_path)
    _write_tread_mesh(root / "mesh.ply", include_riser=True)
    surface = evaluate_stair_result(root, STAIRS, ROI, masks, sample_spacing_m=.5)["surface_completeness"]
    assert surface["coverage_within_tolerance_fraction"] == 1
    assert surface["mean_nearest_mesh_distance_m"] == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("empty_mask", [True, False])
def test_empty_or_missing_predictions_have_json_safe_depth_metrics(tmp_path, empty_mask):
    root, masks = _result(tmp_path, prediction=np.zeros((2, 3)),
                          mask=np.zeros((2, 3), dtype=np.uint8) if empty_mask else None)
    metrics = evaluate_stair_result(root, STAIRS, ROI, masks, sample_spacing_m=.5)["held_out_depth"]["pooled"]
    assert metrics["overlap_pixels"] == 0
    assert metrics["mean_absolute_error_m"] is None
    assert metrics["root_mean_square_error_m"] is None
    if empty_mask:
        assert metrics["coverage"] is None
        assert metrics["within_tolerance_fraction"] is None
    else:
        assert metrics["coverage"] == metrics["within_tolerance_fraction"] == 0
    assert metrics["missing_pixels"] == (0 if empty_mask else 3)
    json.dumps(metrics, allow_nan=False)


def test_empty_mesh_is_missing_surface_not_zero_distance(tmp_path):
    root, masks = _result(tmp_path)
    (root / "mesh.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n"
        "property float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n")
    surface = evaluate_stair_result(root, STAIRS, ROI, masks, sample_spacing_m=.5)["surface_completeness"]
    assert surface["status"] == "empty_mesh"
    assert surface["coverage_within_tolerance_fraction"] == 0
    assert surface["mean_nearest_mesh_distance_m"] is None
    assert surface["distance_sample_count"] == 0


def test_translated_grid_roi_uses_centers_half_open_bounds_and_reports_outside(tmp_path):
    grid = EvidenceOccupancyGrid([10, 20, 30], 1, 2)
    grid.frames = 1
    grid.state[0, 0, 0] = FREE
    grid.free_count[0, 0, 0] = 1
    grid.state[0, 1, 1] = OCCUPIED
    grid.hit_count[0, 1, 1] = 1
    root, masks = _result(tmp_path, grid=grid)
    roi = {"min_m": [9.5, 20, 30], "max_m": [11.5, 22, 32]}
    # Infinite lattice x centers 9.5 and 10.5 lie in ROI; 11.5 is excluded.
    metrics = evaluate_stair_result(root, STAIRS, roi, masks, sample_spacing_m=.5)["occupancy_roi"]
    assert metrics["roi_cell_count"] == 8
    assert metrics["roi_in_volume_cell_count"] == metrics["roi_outside_volume_cell_count"] == 4
    assert metrics["roi_unknown_cell_count"] == 2
    assert metrics["roi_free_cell_count"] == metrics["roi_occupied_cell_count"] == 1
    assert metrics["roi_unknown_fraction"] == .5
    # Stair points are all outside this translated volume, never unknown.
    assert metrics["surface_samples_outside_volume_count"] == 16
    assert metrics["surface_samples_in_unknown_cells_count"] == 0
    assert metrics["surface_samples_outside_volume_fraction"] == 1
    outside_roi = {"min_m": [8, 20, 30], "max_m": [9, 21, 31]}
    outside = evaluate_stair_result(root, STAIRS, outside_roi, masks, sample_spacing_m=.5)["occupancy_roi"]
    assert outside["roi_cell_count"] == outside["roi_outside_volume_cell_count"] == 1
    assert outside["roi_in_volume_cell_count"] == outside["roi_unknown_cell_count"] == 0
    assert outside["roi_unknown_fraction"] is None


def test_progress_reads_integration_only_and_preserves_space_behind_occluder(tmp_path, monkeypatch):
    camera = {"width": 1, "height": 1, "K": np.eye(3).tolist(), "depth_trunc_m": 10}
    writer = SequenceWriter(tmp_path / "sequence", camera, "occluded", "simulator_ground_truth")
    pose = np.eye(4)
    pose[:3, 3] = [.5, .5, -1]
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    for name, split, depth, timestamp in (("front", "integration", 3.5, 0),
            ("held", "held_out", 9.0, 1), ("masked", "integration", 0.0, 2)):
        frame_pose = pose.copy()
        if split == "held_out":
            frame_pose[0, 3] += 1
        writer.add_frame(name, split, rgb, np.array([[depth]], dtype=np.float32), frame_pose, timestamp,
                         reference_rgb=rgb, reference_depth_m=np.array([[9]], dtype=np.float32))
    manifest = writer.finish()
    # Corrupt privileged files after their paths exist. Metadata validation can
    # stat them, but reading any contents would fail. Also track actual reads.
    for path in manifest.parent.rglob("reference_*"):
        path.write_bytes(b"MUST NOT BE READ")
    for path in (manifest.parent / "frames" / "held").iterdir():
        path.write_bytes(b"MUST NOT BE READ")
    original_read = RGBDSequence.read
    seen = []

    def checked_read(self, view):
        assert view["split"] == "integration"
        seen.append(view["id"])
        return original_read(self, view)

    def forbidden(*args, **kwargs):
        raise AssertionError("reference image read by progress evaluator")

    monkeypatch.setattr(RGBDSequence, "read", checked_read)
    monkeypatch.setattr(RGBDSequence, "read_reference", forbidden)
    roi = {"min_m": [0, 0, 0], "max_m": [1, 1, 5]}
    stairs = {"origin_m": [.5, 0, 3], "width_m": .5, "tread_m": .5, "rise_m": .5, "steps": 1}
    settings = {"origin_m": [0, 0, 0], "volume_length_m": 5, "resolution": 5, "pixel_stride": 1}
    rows, slices = occupancy_progress(manifest, settings, roi, stairs, sample_spacing_m=.25)
    assert seen == ["front", "masked"]
    assert [r["frames"] for r in rows] == [1, 2]
    assert [r["timestamp_s"] for r in rows] == [0, 2]
    np.testing.assert_array_equal(slices[0][:, 0], [FREE, FREE, OCCUPIED, UNKNOWN, UNKNOWN])
    np.testing.assert_array_equal(slices[0], slices[1])
    assert slices[0].dtype == np.uint8 and slices[0].shape == (5, 5)
    assert rows[0]["roi_known_cell_count"] == 3
    assert rows[0]["roi_unknown_cell_count"] == 2
    assert rows[0]["surface_samples_in_unknown_cells_fraction"] == 1
    assert rows[0]["surface_samples_in_free_cells_fraction"] == 0
    assert rows[0]["slice_axis"] == "x" and rows[0]["slice_coordinate_m"] == .5
    assert rows[0]["slice_horizontal_extent_m"] == rows[0]["slice_vertical_extent_m"] == [0, 5]
    assert rows[0]["ground_truth_used_for_fusion"] is False
    json.dumps(rows, allow_nan=False)


def test_rejects_nonheldout_mask_and_unbounded_grid_before_loading_observations(tmp_path):
    root, masks = _result(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        evaluate_stair_result(root, STAIRS, ROI, {"integration_frame": next(iter(masks.values()))})
    settings = {"origin_m": [0, 0, 0], "volume_length_m": 5, "resolution": 10000, "pixel_stride": 1}
    with pytest.raises(ValueError, match="grid cells"):
        occupancy_progress(Path("does-not-exist.json"), settings, ROI, STAIRS)
