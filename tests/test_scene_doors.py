"""Hinge coordinates, static capture state, and portable door scene contracts."""

import os
import shutil

import numpy as np
import pytest
import yaml

from wtumi_recon.scene_bundle import build_scene, load_scene_config
from wtumi_recon.scene_capture import apply_scene_pose, capture_scene
from wtumi_recon.scene_doors import configured_door_angles, normalize_doors, scene_door_ids, set_scene_doors


RENDER = pytest.mark.skipif(os.environ.get("WTUMI_TEST_MUJOCO_RENDER") != "1",
                          reason="Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL")


def _door(**changes):
    return {"name": "passage", "hinge_position_m": [-.5, 2., 0.],
            "width_m": 1., "height_m": 2.1, "thickness_m": .045,
            "angle_degrees": 45., **changes}


def _build(tmp_path, door=None):
    cfg = {key: value for key, value in load_scene_config("stairs").items() if not key.startswith("_")}
    cfg["environment"] = {"objects": [
        {"name": "floor", "type": "box", "position_m": [0., 2., -.04], "size_m": [5., 6., .08]},
        {"name": "back", "type": "box", "position_m": [0., 4.5, 1.2], "size_m": [5., .08, 2.4]},
    ], "doors": [_door() if door is None else door]}
    cfg["camera"].update(width=80, height=64, position_in_rig_m=[0., 0., 1.5], target_in_rig_m=[0., 2., 1.])
    cfg["torso"]["enabled"] = cfg["box"]["enabled"] = False
    cfg["trajectory"] = [
        {"id": f"view_{i}", "timestamp_s": float(i), "position_m": position, "yaw_degrees": 0.}
        for i, position in enumerate(([0., 0., 0.], [-.25, 0., 0.], [.25, .1, 0.]))]
    cfg["held_out"] = [{"id": "evaluation", "timestamp_s": 3., "eye": [0., .3, 1.5],
                        "target": [0., 2., .8], "up": [0., 0., 1.]}]
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(cfg))
    report = build_scene(source, tmp_path / "built")
    return report["config"], tmp_path / "built/model.xml"


@pytest.mark.parametrize("change,match", [
    ({"width_m": 0}, "positive"), ({"thickness_m": True}, "positive"),
    ({"name": "../escape"}, "safe"), ({"angle_degrees": float("nan")}, "finite"),
    ({"angle_degrees": 120}, "range_degrees"), ({"range_degrees": [90, 0]}, "less"),
    ({"range_degrees": [0]}, "lower and upper"),
    ({"hinge_position_m": [0, 1]}, "three coordinates"),
    ({"color_rgba": [1, 2, 1, 1]}, r"\[0, 1\]"),
    ({"unknown": 3}, "optional fields"),
])
def test_invalid_door_schema_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        normalize_doors([_door(**change)])


def test_doors_require_a_list_and_unique_names():
    with pytest.raises(ValueError, match="list"):
        normalize_doors(None)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_doors([_door(), _door()])
    with pytest.raises(ValueError, match="at most 32"):
        normalize_doors([_door(name=f"door{i}") for i in range(33)])


@pytest.mark.parametrize("initial", [0., 45., 90.])
def test_standalone_xml_starts_at_selected_angle_and_overrides_use_absolute_angles(tmp_path, initial):
    mujoco = pytest.importorskip("mujoco")
    hinge, closed_yaw = [.2, 1.2, .05], 32.
    cfg, path = _build(tmp_path, _door(hinge_position_m=hinge, closed_yaw_degrees=closed_yaw,
                                     angle_degrees=initial))
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ids = scene_door_ids(mujoco, model, cfg)["passage"]
    for index, angle in enumerate((initial, 0., 45., 90.)):
        if index:
            set_scene_doors(mujoco, model, data, cfg, {"passage": angle})
        mujoco.mj_forward(model, data)
        radians = np.deg2rad(closed_yaw + angle)
        expected = np.array(hinge) + [.5 * np.cos(radians), .5 * np.sin(radians), 1.05]
        np.testing.assert_allclose(data.geom_xpos[ids["geom"]], expected, atol=1e-10)
        np.testing.assert_allclose(data.geom_xmat[ids["geom"]].reshape(3, 3)[:, 0],
                                   [np.cos(radians), np.sin(radians), 0], atol=1e-10)
        assert data.qpos[ids["qpos"]] == pytest.approx(np.deg2rad(angle))
        for name in ("door_passage_leaf", "door_passage_handle_front", "door_passage_handle_back"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert model.geom_bodyid[geom] == ids["body"] and model.geom_group[geom] == 0
    set_scene_doors(mujoco, model, data, cfg)
    assert data.qpos[ids["qpos"]] == pytest.approx(np.deg2rad(initial))


def test_door_overrides_are_atomic_validated_and_reset_for_each_capture_pose(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("door preview/capture must not step physics"))
    cfg, path = _build(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    initial = data.qpos.copy()
    for override in ({"unknown": 1}, {"passage": 115}, {"passage": True}, {"passage": float("nan")}):
        with pytest.raises(ValueError):
            set_scene_doors(mujoco, model, data, cfg, override)
        np.testing.assert_array_equal(data.qpos, initial)
    for frame in cfg["trajectory"]:
        set_scene_doors(mujoco, model, data, cfg, {"passage": 100.})
        data.qvel[:] = 1.
        apply_scene_pose(mujoco, model, data, cfg, frame)
        np.testing.assert_allclose(data.qpos, initial)
        np.testing.assert_array_equal(data.qvel, 0.)
        assert data.time == 0


@pytest.mark.parametrize("change", ["initial_angle", "hinge_position", "missing_joint", "leaf_group", "unconfigured"])
def test_door_model_mismatch_fails_before_creating_capture(tmp_path, change):
    pytest.importorskip("mujoco")
    cfg, path = _build(tmp_path)
    if change == "initial_angle":
        cfg["environment"]["doors"][0]["angle_degrees"] += 5
    elif change == "hinge_position":
        cfg["environment"]["doors"][0]["hinge_position_m"][0] += .1
    elif change == "missing_joint":
        path.write_text(path.read_text().replace('name="door_passage_hinge"', 'name="unexpected"'))
    elif change == "leaf_group":
        path.write_text(path.read_text().replace('name="door_passage_leaf" type="box" group="0"',
                                                 'name="door_passage_leaf" type="box" group="1"'))
    else:
        cfg["environment"]["doors"] = []
    with pytest.raises(ValueError, match="door"):
        capture_scene(cfg, path, tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


def test_door_scene_remains_portable_after_original_is_removed(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    cfg, path = _build(tmp_path / "source")
    moved = tmp_path / "moved"
    path.parent.rename(moved)
    shutil.rmtree(tmp_path / "source")
    loaded = load_scene_config(moved)
    model = mujoco.MjModel.from_xml_path(str(moved / "model.xml"))
    data = mujoco.MjData(model)
    assert set_scene_doors(mujoco, model, data, loaded) == {"passage": 45.}


def test_legacy_scene_without_doors_has_no_hinge_and_accepts_pose_updates(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    report = build_scene("stairs", tmp_path / "stairs")
    cfg = report["config"]
    assert cfg["environment"]["doors"] == []
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stairs/model.xml"))
    assert model.njnt == 0
    assert set_scene_doors(mujoco, model, mujoco.MjData(model), cfg) == {}
    # Pose-only callers historically provide camera/rig settings without a
    # full scene environment; absent door configuration must remain optional.
    cfg.pop("environment")
    assert configured_door_angles(cfg) == {}
    assert scene_door_ids(mujoco, model, cfg) == {}
    data = mujoco.MjData(model)
    assert set_scene_doors(mujoco, model, data, cfg) == {}
    pose = apply_scene_pose(mujoco, model, data, cfg, cfg["trajectory"][0])
    assert np.isfinite(pose["T_wc"]).all()


@RENDER
def test_capture_keeps_visible_door_as_environment_and_same_state_in_heldouts(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from wtumi_recon import scene_capture
    from wtumi_recon.rig_capture import _geom_mask
    from wtumi_recon.sequence import RGBDSequence

    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("capture must not step physics"))
    cfg, path = _build(tmp_path)
    original = scene_capture._render_observation
    masks, angles = [], []
    model = mujoco.MjModel.from_xml_path(str(path))
    leaf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "door_passage_leaf")

    def record(mj, renderer, data, *args, **kwargs):
        images = original(mj, renderer, data, *args, **kwargs)
        masks.append(_geom_mask(mj, images[2], [leaf_id]))
        angles.append(data.qpos.copy())
        return images

    monkeypatch.setattr(scene_capture, "_render_observation", record)
    report = capture_scene(cfg, path, tmp_path / "capture")
    sequence = RGBDSequence(tmp_path / "capture/manifest.json")
    assert report["metadata"]["door_angles_degrees"] == {"passage": 45.}
    records = report["frames"] + report["held_out_frames"]
    assert len(records) == len(masks) == 4
    for frame, mask, angles in zip(records, masks, angles):
        assert mask.any(), frame["id"]
        assert frame["door_angles_degrees"] == {"passage": 45.}
        np.testing.assert_allclose(angles, [np.pi / 4])
        rgb, depth = sequence.read(frame["id"])
        assert np.all(depth[mask] > 0)
        assert np.any(rgb[mask] > 0)
