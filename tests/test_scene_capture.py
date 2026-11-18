"""Generic scene capture checks, with actual rendering explicitly enabled."""

import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.scene_bundle import build_scene, load_scene_config
from wtumi_recon.scene_capture import apply_scene_pose, capture_scene
from wtumi_recon.sequence import RGBDSequence


RENDER = pytest.mark.skipif(
    os.environ.get("WTUMI_TEST_MUJOCO_RENDER") != "1",
    reason="Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL",
)


def _build(tmp_path, template="room", *, optional_geometry=True):
    cfg = load_scene_config(template)
    cfg.pop("_path", None)
    cfg.pop("_root", None)
    # These tests deliberately exercise legacy optional proxy geometry.
    # Actual G1 template self-masking is covered in test_g1_capture.py.
    cfg.pop("robot", None)
    cfg["camera"]["width"], cfg["camera"]["height"] = 80, 64
    cfg["trajectory"] = [cfg["trajectory"][index] for index in (0, len(cfg["trajectory"]) // 2, -1)]
    cfg["held_out"] = cfg["held_out"][:1]
    cfg["torso"]["enabled"] = optional_geometry
    cfg["box"]["enabled"] = optional_geometry
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    build_scene(path, tmp_path / "scene")
    return load_scene_config(path), tmp_path / "scene/model.xml"


def _mask(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


def test_existing_capture_is_preserved_before_any_model_loading(tmp_path):
    output = tmp_path / "capture"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        capture_scene({}, tmp_path / "missing.xml", output)
    assert marker.read_text(encoding="utf-8") == "existing data"


def test_camera_pose_tracks_rig_but_head_rotation_does_not_move_box(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    cfg, model_path = _build(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    original = copy.deepcopy(cfg["trajectory"][0])
    original["yaw_degrees"] = 27
    original["head_yaw_degrees"], original["head_pitch_degrees"] = 0, 0
    first = apply_scene_pose(mujoco, model, data, cfg, original)
    turned = {**original, "head_yaw_degrees": 25, "head_pitch_degrees": -12}
    second = apply_scene_pose(mujoco, model, data, cfg, turned)
    np.testing.assert_allclose(first["T_wbox"], second["T_wbox"])
    np.testing.assert_allclose(first["T_wrig"], second["T_wrig"])
    np.testing.assert_allclose(first["T_wc"][:3, 3], second["T_wc"][:3, 3])
    assert not np.allclose(first["T_wc"][:3, :3], second["T_wc"][:3, :3])
    np.testing.assert_allclose(second["T_wc"], second["T_wrig"] @ second["T_rig_camera"], atol=1e-8)
    assert data.time == 0


def test_optional_geometry_can_be_absent_without_invalid_geom_indices(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    cfg, model_path = _build(tmp_path, optional_geometry=False)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    poses = apply_scene_pose(mujoco, model, data, cfg, cfg["trajectory"][0])
    assert "T_wbox" not in poses
    assert np.isfinite(poses["T_wc"]).all()
    cfg["box"]["enabled"] = True
    with pytest.raises(ValueError, match="missing enabled geom: box"):
        apply_scene_pose(mujoco, model, data, cfg, cfg["trajectory"][0])


@pytest.mark.parametrize("mismatch", ["camera_name", "camera_mount", "camera_fovy", "box_enabled"])
def test_model_configuration_mismatch_fails_before_creating_capture(tmp_path, mismatch):
    pytest.importorskip("mujoco")
    cfg, model_path = _build(tmp_path)
    if mismatch == "camera_name":
        model_path.write_text(model_path.read_text().replace('name="head_camera"', 'name="missing_camera"'))
    elif mismatch == "camera_mount":
        cfg["camera"]["position_in_rig_m"][0] += 0.05
    elif mismatch == "camera_fovy":
        cfg["camera"]["fovy_degrees"] += 5
    else:
        cfg["box"]["enabled"] = False
    output = tmp_path / "capture"
    with pytest.raises(ValueError):
        capture_scene(cfg, model_path, output)
    assert not output.exists()


@RENDER
@pytest.mark.parametrize("template", ["room", "stairs", "doorway"])
def test_generic_templates_capture_masked_portable_sequences(tmp_path, template):
    pytest.importorskip("mujoco")
    cfg, model_path = _build(tmp_path, template)
    output = tmp_path / "capture"
    report = capture_scene(cfg, model_path, output)
    sequence = RGBDSequence(output / report["paths"]["manifest"])
    assert report["integration_frame_ids"] == [frame["id"] for frame in cfg["trajectory"]]
    assert report["held_out_frame_ids"] == [frame["id"] for frame in cfg["held_out"]]
    assert len(sequence.views) == 4
    assert not (output / "capture_preview.gif").exists()
    assert "stairs" not in report
    for record in report["frames"]:
        rgb, depth = sequence.read(record["id"])
        directory = output / record["frame_directory"]
        raw_depth = np.load(directory / "raw_depth_m.npy", allow_pickle=False)
        with Image.open(directory / "raw_rgb.png") as image:
            raw_rgb = np.asarray(image).copy()
        excluded = _mask(directory / "excluded_mask.png")
        box = _mask(directory / "box_mask.png")
        robot = _mask(directory / "robot_mask.png")
        np.testing.assert_array_equal(excluded, box | robot)
        assert (depth[excluded] == 0).all() and (rgb[excluded] == 0).all()
        np.testing.assert_array_equal(raw_depth[~excluded], depth[~excluded])
        np.testing.assert_array_equal(raw_rgb[~excluded], rgb[~excluded])
        assert record["valid_environment_pixels"] == np.count_nonzero(depth)
        assert record["box_pixels"] == box.sum()
        assert record["excluded_pixels"] == excluded.sum()
        np.testing.assert_allclose(record["T_wc"], np.array(record["T_wrig"]) @ record["T_rig_camera"], atol=1e-8)
    assert sum(frame["box_pixels"] for frame in report["frames"]) > 0
    integration_poses = [frame["T_wc"] for frame in report["frames"]]
    for frame in report["held_out_frames"]:
        assert not any(np.allclose(frame["T_wc"], pose) for pose in integration_poses)
        assert frame["valid_environment_pixels"] > 0
    # The manifest must remain usable after the complete capture is relocated.
    moved = tmp_path / "relocated_capture"
    output.rename(moved)
    relocated = RGBDSequence(moved / "manifest.json")
    for frame in relocated.views:
        assert relocated.read(frame)[1].shape == (64, 80)
    assert json.loads((moved / "capture.json").read_text())["paths"]["manifest"] == "manifest.json"


@RENDER
def test_held_out_reference_is_independent_of_optional_robot_and_box(tmp_path):
    pytest.importorskip("mujoco")
    enabled, disabled = tmp_path / "enabled", tmp_path / "disabled"
    enabled.mkdir()
    disabled.mkdir()
    enabled_cfg, enabled_model = _build(enabled, optional_geometry=True)
    disabled_cfg, disabled_model = _build(disabled, optional_geometry=False)
    captures = []
    for root, cfg, model in ((enabled, enabled_cfg, enabled_model), (disabled, disabled_cfg, disabled_model)):
        report = capture_scene(cfg, model, root / "capture")
        captures.append((report, RGBDSequence(root / "capture/manifest.json")))
    enabled_report, enabled_sequence = captures[0]
    disabled_report, disabled_sequence = captures[1]
    name = enabled_report["held_out_frame_ids"][0]
    rgb_a, depth_a = enabled_sequence.read(name)
    rgb_b, depth_b = disabled_sequence.read(name)
    np.testing.assert_array_equal(rgb_a, rgb_b)
    np.testing.assert_allclose(depth_a, depth_b, atol=2e-5, rtol=0)
    assert sum(frame["box_pixels"] for frame in enabled_report["frames"]) > 0
    for frame in disabled_report["frames"]:
        assert frame["excluded_pixels"] == 0
        assert "T_wbox" not in frame
