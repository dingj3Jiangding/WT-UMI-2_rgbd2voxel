"""Whole-robot self masks and actual G1 capture with portable observations."""

import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.scene_capture import _scene_ids, apply_scene_pose, capture_scene
from wtumi_recon.sequence import RGBDSequence


RENDER = pytest.mark.skipif(os.environ.get("WTUMI_TEST_MUJOCO_RENDER") != "1",
                          reason="Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL")


def _contract_model(*, alteration=None):
    """A tiny nested model isolates classification from the downloaded meshes."""
    mujoco = pytest.importorskip("mujoco")
    source = '''<mujoco><worldbody>
      <camera name="evaluation_camera" fovy="65"/>
      <body name="sensor_rig" mocap="true">
        <camera name="head_camera" pos="0 0 1.3" fovy="65"/>
        <geom name="box" type="box" size=".2 .2 .2" pos="0 .5 .8" group="2"/>
        <body name="g1_base">
          <geom name="g1_torso_visual" type="box" size=".1 .1 .3" group="1"/>
          <body name="g1_left_arm" pos=".2 0 0">
            <geom name="g1_left_hand_visual" type="sphere" size=".05" group="1"/>
          </body>
          <body name="g1_right_arm" pos="-.2 0 0">
            <geom name="g1_right_hand_visual" type="sphere" size=".05" group="1"/>
          </body>
        </body>
      </body>
    </worldbody></mujoco>'''
    tree = ET.fromstring(source)
    if alteration:
        alteration(tree)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    cfg = {"camera": {"position_in_rig_m": [0, 0, 1.3], "target_in_rig_m": [0, 1, .8],
                       "fovy_degrees": 65},
           "torso": {"enabled": False}, "box": {"enabled": True},
           "robot": {"model": "unitree_g1", "pose": "carry"}}
    return mujoco, model, cfg


def test_g1_contract_selects_all_nested_links_and_preserves_rig_pose():
    mujoco, model, cfg = _contract_model()
    ids = _scene_ids(mujoco, model, cfg)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids["torso"]}
    assert names == {"g1_torso_visual", "g1_left_hand_visual", "g1_right_hand_visual"}
    assert not set(ids["torso"]) & set(ids["box"])
    data = mujoco.MjData(model)
    first = apply_scene_pose(mujoco, model, data, cfg,
                             {"position_m": [0, 0, 0], "yaw_degrees": 0})
    second = apply_scene_pose(mujoco, model, data, cfg,
                              {"position_m": [.3, -.2, 0], "yaw_degrees": 30})
    np.testing.assert_allclose(np.linalg.inv(first["T_wrig"]) @ first["T_wbox"],
                               np.linalg.inv(second["T_wrig"]) @ second["T_wbox"], atol=1e-8)
    np.testing.assert_allclose(second["T_wc"], second["T_wrig"] @ second["T_rig_camera"], atol=1e-8)
    assert data.time == 0


@pytest.mark.parametrize("case", ["proxy_enabled", "missing_g1", "wrong_group", "world_link",
                                  "unrelated_group1", "robot_not_configured"])
def test_g1_model_config_mismatches_are_rejected(case):
    def edit(tree):
        if case == "missing_g1":
            for geom in tree.findall(".//geom"):
                if geom.get("name", "").startswith("g1_"):
                    geom.set("name", "other_" + geom.get("name"))
        elif case == "wrong_group":
            tree.find(".//geom[@name='g1_left_hand_visual']").set("group", "0")
        elif case == "world_link":
            body = tree.find(".//body[@name='g1_left_arm']")
            tree.find(".//body[@name='g1_base']").remove(body)
            tree.find("worldbody").append(body)
        elif case == "unrelated_group1":
            ET.SubElement(tree.find("worldbody"), "geom", name="unrelated", type="sphere", size=".1", group="1")
    mujoco, model, cfg = _contract_model(alteration=edit)
    if case == "proxy_enabled":
        cfg["torso"]["enabled"] = True
    elif case == "robot_not_configured":
        cfg.pop("robot")
    with pytest.raises(ValueError):
        _scene_ids(mujoco, model, cfg)


def _build_g1(tmp_path, *, with_robot=True, template="g1_stairs"):
    from wtumi_recon.scene_bundle import build_scene, load_scene_config

    cfg = {k: v for k, v in load_scene_config(template).items() if not k.startswith("_")}
    cfg["camera"].update(width=96, height=80, fovy_degrees=85,
                         target_in_rig_m=[0, .5, .35])
    cfg["trajectory"] = [cfg["trajectory"][i] for i in (0, len(cfg["trajectory"]) // 2, -1)]
    cfg["held_out"] = cfg["held_out"][:1]
    if not with_robot:
        cfg.pop("robot")
        cfg["box"]["enabled"] = False
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    built = build_scene(path, tmp_path / "model")
    return built["config"], Path(built["output"]) / built["paths"]["model"]


@RENDER
@pytest.mark.parametrize("template", ["g1_stairs", "room", "doorway"])
def test_actual_g1_capture_masks_links_and_payload_and_keeps_portable_frames(tmp_path, template):
    mujoco = pytest.importorskip("mujoco")
    cfg, model_path = _build_g1(tmp_path, template=template)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    ids = _scene_ids(mujoco, model, cfg)
    assert len(ids["torso"]) > 3
    output = tmp_path / "capture"
    report = capture_scene(cfg, model_path, output)
    assert report["metadata"]["robot_model"] == "unitree_g1"
    assert report["metadata"]["robot_geom_count"] == len(ids["torso"])
    assert report["rig"]["robot"] == cfg["robot"]
    sequence = RGBDSequence(output / "manifest.json")
    robot_pixels, box_pixels = 0, 0
    for frame in report["frames"]:
        rgb, depth = sequence.read(frame["id"])
        directory = output / frame["frame_directory"]
        with Image.open(directory / "robot_mask.png") as image:
            robot = np.asarray(image) > 0
        with Image.open(directory / "box_mask.png") as image:
            box = np.asarray(image) > 0
        with Image.open(directory / "excluded_mask.png") as image:
            excluded = np.asarray(image) > 0
        assert not np.any(robot & box)
        np.testing.assert_array_equal(excluded, robot | box)
        assert not np.any(depth[excluded]) and not np.any(rgb[excluded])
        raw_depth = np.load(directory / "raw_depth_m.npy", allow_pickle=False)
        np.testing.assert_array_equal(raw_depth[~excluded], depth[~excluded])
        robot_pixels += frame["robot_pixels"]
        box_pixels += frame["box_pixels"]
    assert robot_pixels > 0
    assert (box_pixels > 0) == cfg["box"]["enabled"]
    moved = tmp_path / "moved_capture"
    output.rename(moved)
    assert len(RGBDSequence(moved / "manifest.json").views) == 4


@RENDER
@pytest.mark.parametrize("template", ["g1_stairs", "room", "doorway"])
def test_actual_g1_and_payload_do_not_enter_held_out_references(tmp_path, template):
    outputs = []
    for label, enabled in (("g1", True), ("environment", False)):
        directory = tmp_path / label
        directory.mkdir()
        cfg, model_path = _build_g1(directory, with_robot=enabled, template=template)
        report = capture_scene(cfg, model_path, directory / "capture")
        sequence = RGBDSequence(directory / "capture/manifest.json")
        outputs.append(sequence.read(report["held_out_frame_ids"][0]))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=2e-5, rtol=0)
