"""Acceptance checks for real fusion, disjoint views and useful saved results."""

from pathlib import Path
import copy
import json

import numpy as np
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.evaluation import evaluate_rgbd
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.tsdf import Open3DTSDFMap

ROOT = Path(__file__).resolve().parents[1]


def compact_config():
    cfg = yaml.safe_load((ROOT / "configs/scenes/multiview_cube.yaml").read_text())
    cfg["camera"].update(width=120, height=90, fx=110.0, fy=110.0, cx=59.5, cy=44.5)
    cfg["reconstruction"]["resolution"] = 96
    return cfg


def test_multiview_fuses_only_integration_frames_and_reconstructs_color(tmp_path, monkeypatch):
    cfg = compact_config()
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    calls = []
    integrate = Open3DTSDFMap.integrate

    def track(self, rgbd, intrinsics, t_wc):
        calls.append(t_wc.copy())
        return integrate(self, rgbd, intrinsics, t_wc)

    monkeypatch.setattr(Open3DTSDFMap, "integrate", track)
    out = tmp_path / "result"
    report = run_reconstruction(path, out)
    resolved = load_run_config(path)
    fused = [v for v in resolved["views"] if v["split"] == "integration"]
    held = [v for v in resolved["views"] if v["split"] == "held_out"]
    assert len(calls) == len(fused) == 4
    for actual, expected in zip(calls, fused):
        np.testing.assert_allclose(actual, expected["T_wc"])
    assert not any(np.allclose(c, v["T_wc"]) for c in calls for v in held)
    assert report["integrated_frame_ids"] == [v["id"] for v in fused]
    assert report["mesh_triangles"] > 100
    result = report["summary"]["held_out"]
    assert result["coverage"] > 0.8
    assert result["rmse_m"] < 0.03
    assert result["rgb_mae_0_1"] < 0.15
    import open3d as o3d
    rgb = np.asarray(o3d.io.read_image(str(out / "renders/novel_front_right/rgb.png")))
    depth = np.load(out / "renders/novel_front_right/depth_m.npy")
    assert len(np.unique(rgb[depth > 0], axis=0)) > 20
    with np.load(out / "tsdf_volume.npz") as volume:
        assert volume["weight"].max() >= 2
        assert (volume["weight"] == 0).any()  # Unknown is not silently occupied.
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["held_out_frame_ids"] == [v["id"] for v in held]
    assert (out / "renders/novel_front_right/comparison.png").exists()
    replay = load_run_config(out / "config_resolved.yaml")
    assert replay["views"] == resolved["views"]


def test_rejects_held_out_pose_leakage_and_conflicting_voxel_size(tmp_path):
    cfg = compact_config()
    bad = copy.deepcopy(cfg)
    bad["views"][-1].update(eye=cfg["views"][0]["eye"], target=cfg["views"][0]["target"])
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="held_out poses"):
        load_run_config(path)
    cfg["reconstruction"]["voxel_size_m"] = 0.01
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unsupported reconstruction"):
        load_run_config(path)


def test_no_overlap_is_not_perfect_error_and_is_valid_json():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    result = evaluate_rgbd(rgb, np.ones((2, 2)), rgb, np.zeros((2, 2)))
    assert result["coverage"] == 0
    assert result["missing_pixels"] == 4
    assert result["mae_m"] is None and result["rgb_psnr_db"] is None
    json.dumps(result, allow_nan=False)


def test_missing_extra_and_error_have_independent_counts():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    result = evaluate_rgbd(rgb, [[1, 1], [0, 0]], rgb, [[1.1, 0], [1, 0]])
    assert result["missing_pixels"] == result["extra_pixels"] == 1
    assert result["coverage"] == 0.5
    assert result["silhouette_iou"] == pytest.approx(1/3)
    assert result["mae_m"] == pytest.approx(0.1)
    assert result["accurate_reference_fraction"] == 0


def test_mesh_asset_paths_are_relative_to_config_not_cwd(tmp_path, monkeypatch):
    import open3d as o3d
    from wtumi_recon.synthetic import SyntheticScene
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(0.5, 0.5, 0.5)
    mesh.paint_uniform_color([0.2, 0.4, 0.8])
    assert o3d.io.write_triangle_mesh(str(assets / "box.ply"), mesh)
    cfg = compact_config()
    cfg["scene"]["objects"] = [{"type": "mesh", "path": "assets/box.ply", "translation_m": [0.65, 0.65, 1]}]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path.parent)
    resolved = load_run_config(path)
    scene = SyntheticScene.from_config(resolved["scene"])
    np.testing.assert_allclose(scene.mesh.get_min_bound(), [0.65, 0.65, 1])
    np.testing.assert_allclose(scene.mesh.get_max_bound(), [1.15, 1.15, 1.5])
    np.testing.assert_allclose(np.asarray(scene.mesh.vertex_colors), np.tile([0.2, 0.4, 0.8], (8, 1)))
