"""Portable geometry, validation, and externally openable MuJoCo scenes."""

from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from wtumi_recon.mujoco_capture import opencv_pose_from_mujoco
from wtumi_recon.rig_capture import head_camera_pose, rig_transform
from wtumi_recon.scene_bundle import BUILTIN_SCENES, build_scene, create_scene, load_scene_config


def _write_config(tmp_path, edit):
    cfg = {k: v for k, v in load_scene_config("room").items() if not k.startswith("_")}
    edit(cfg)
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("name", BUILTIN_SCENES)
def test_template_build_opens_with_correct_initial_camera(tmp_path, name):
    mujoco = pytest.importorskip("mujoco")
    report = build_scene(name, tmp_path / name)
    cfg = report["config"]
    assert len(cfg["trajectory"]) == 13
    assert len(cfg["held_out"]) == 2
    assert "stairs" not in cfg
    model = mujoco.MjModel.from_xml_path(str(Path(report["output"]) / report["paths"]["model"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    rig_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sensor_rig")
    assert model.body_mocapid[rig_id] >= 0
    assert model.cam_bodyid[camera_id] == rig_id
    frame = cfg["trajectory"][0]
    expected = rig_transform(frame["position_m"], frame["yaw_degrees"]) @ head_camera_pose(cfg["camera"], frame)
    actual = opencv_pose_from_mujoco(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    for obj in cfg["environment"]["objects"]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj["name"])
        assert model.geom_group[geom_id] == 0
        if obj["type"] == "box":
            np.testing.assert_allclose(model.geom_size[geom_id], np.asarray(obj["size_m"]) / 2)
    for obj, group in (("torso", 1), ("box", 2)):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj)
        if cfg[obj]["enabled"]:
            assert geom_id >= 0 and model.geom_group[geom_id] == group
        else:
            assert geom_id == -1
    assert not np.any(model.geom_contype)
    assert not np.any(model.geom_conaffinity)
    assert set(model.geom_group) <= {0, 1, 2}


def test_nonzero_initial_pose_is_saved_in_standalone_xml(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    def edit(cfg):
        cfg["trajectory"][0].update(position_m=[.13, -.3, .08], yaw_degrees=18,
                                    head_yaw_degrees=-12, head_pitch_degrees=9)
    path = _write_config(tmp_path, edit)
    report = build_scene(path, tmp_path / "build")
    cfg = report["config"]
    model = mujoco.MjModel.from_xml_path(str(Path(report["output"]) / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    frame = cfg["trajectory"][0]
    expected = rig_transform(frame["position_m"], frame["yaw_degrees"]) @ head_camera_pose(cfg["camera"], frame)
    np.testing.assert_allclose(opencv_pose_from_mujoco(data.cam_xpos[cam], data.cam_xmat[cam]), expected, atol=1e-10)


def test_create_and_edit_geometry_preserves_template(tmp_path):
    original = load_scene_config("room")
    target = tmp_path / "custom"
    create_scene("room", target)
    copied = load_scene_config(target)
    assert copied["environment"] == original["environment"]
    cfg = yaml.safe_load((target / "scene.yaml").read_text())
    cfg["environment"]["objects"][1]["position_m"][0] += .1
    (target / "scene.yaml").write_text(yaml.safe_dump(cfg))
    changed = load_scene_config(target)
    assert changed["camera"] == original["camera"]
    assert changed["trajectory"] == original["trajectory"]
    assert changed["environment"] != original["environment"]
    assert load_scene_config("room")["environment"] == original["environment"]
    with pytest.raises(FileExistsError):
        create_scene("room", target)
    with pytest.raises(FileExistsError):
        build_scene("room", target)


def test_partial_options_use_defaults(tmp_path):
    def edit(cfg):
        # Exercise the legacy proxy defaults independently of template robots.
        cfg.pop("robot", None)
        for key in ("box", "torso", "reconstruction", "occupancy", "evaluation", "playback"):
            cfg.pop(key)
        cfg["camera"] = {"width": 160, "height": 120}
    cfg = load_scene_config(_write_config(tmp_path, edit))
    assert cfg["box"]["enabled"] is False
    assert cfg["camera"]["width"] == 160
    assert cfg["reconstruction"]["resolution"] == 160
    assert cfg["occupancy"]["enabled"]


@pytest.mark.parametrize("edit,match", [
    (lambda c: c["environment"]["objects"][0].update(size_m=[1, -1, 2]), "positive"),
    (lambda c: c["environment"]["objects"][0].update(name="torso"), "reserved"),
    (lambda c: c["environment"]["objects"][1].update(name="floor"), "duplicate"),
    (lambda c: c["environment"]["objects"][0].update(name="../bad"), "safe"),
    (lambda c: c["environment"]["objects"][0].update(color_rgba=[1, 2, 1, 1]), "color_rgba"),
    (lambda c: c["trajectory"][1].update(timestamp_s=0), "strictly increasing"),
    (lambda c: c["trajectory"][1].update(head_yaw_degrees=100), "head yaw"),
    (lambda c: c["held_out"][0].update(id=c["trajectory"][0]["id"]), "unique"),
    (lambda c: c["camera"].update(width=True), "integer"),
    (lambda c: c["reconstruction"].update(resolution=512), "resolution"),
    (lambda c: c["occupancy"].update(resolution=512), "resolution"),
    (lambda c: c["reconstruction"].update(backend="scalable"), "uniform"),
    (lambda c: c.update(unknown_section={}), "invalid scene fields"),
])
def test_invalid_scene_rejected(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_scene_config(_write_config(tmp_path, edit))


def test_held_out_cannot_repeat_integration_pose(tmp_path):
    def edit(cfg):
        frame = cfg["trajectory"][0]
        pose = rig_transform(frame["position_m"], frame["yaw_degrees"]) @ head_camera_pose(cfg["camera"], frame)
        cfg["held_out"][0].update(eye=pose[:3, 3].tolist(), target=(pose[:3, 3] + pose[:3, 2]).tolist(),
                                  up=(-pose[:3, 1]).tolist())
    with pytest.raises(ValueError, match="held-out poses"):
        load_scene_config(_write_config(tmp_path, edit))


def _mesh_scene(tmp_path, relative="shape.obj"):
    (tmp_path / "shape.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    def edit(cfg):
        # This fixture isolates custom mesh copying from vendored robot assets.
        cfg.pop("robot", None)
        cfg["environment"]["objects"] = [{"name": "shape", "type": "mesh", "path": relative,
                                           "position_m": [0, 1, 0], "scale": [.4, .4, .4]}]
    return _write_config(tmp_path, edit)


def test_mesh_bundle_can_move_without_source(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    source = tmp_path / "source"
    source.mkdir()
    path = _mesh_scene(source)
    created = tmp_path / "created"
    create_scene(path, created)
    built = tmp_path / "built"
    build_scene(created, built)
    shutil.rmtree(source)
    shutil.rmtree(created)
    moved = tmp_path / "moved"
    built.rename(moved)
    cfg = load_scene_config(moved)
    assert cfg["environment"]["objects"][0]["path"] == "assets/shape.obj"
    model = mujoco.MjModel.from_xml_path(str(moved / "model.xml"))
    assert model.nmesh == 1


@pytest.mark.parametrize("relative", ["../outside.obj", "/tmp/outside.obj"])
def test_mesh_paths_cannot_escape_bundle(tmp_path, relative):
    with pytest.raises(ValueError, match="inside the scene bundle"):
        load_scene_config(_mesh_scene(tmp_path, relative))


def test_mesh_symlink_cannot_escape_bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    path = _mesh_scene(folder)
    outside = tmp_path / "outside.obj"
    (folder / "shape.obj").rename(outside)
    (folder / "shape.obj").symlink_to(outside)
    with pytest.raises(ValueError, match="inside the scene bundle"):
        load_scene_config(path)
