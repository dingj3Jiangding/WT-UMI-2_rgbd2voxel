"""Door controls change source previews while keeping saved captures fixed."""

from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest
import yaml

from wtumi_recon.scene_viewer import (
    PlaybackController, _validate_recorded_doors, apply_playback_frame,
    playback_status, prepare_scene_viewer,
)


def _doors():
    return [
        {"name": "entry", "hinge_position_m": [-.5, 1., 0.],
         "width_m": 1., "height_m": 2., "thickness_m": .05,
         "closed_yaw_degrees": 0., "angle_degrees": 30.,
         "range_degrees": [0., 95.], "color_rgba": [.6, .3, .1, 1.]},
        {"name": "side", "hinge_position_m": [.5, 2., 0.],
         "width_m": .8, "height_m": 2., "thickness_m": .05,
         "closed_yaw_degrees": 90., "angle_degrees": 0.,
         "range_degrees": [0., 110.], "color_rgba": [.6, .3, .1, 1.]},
    ]


@pytest.fixture
def door_scene(tmp_path):
    template = Path(__file__).resolve().parents[1] / "scenes/room/scene.yaml"
    cfg = yaml.safe_load(template.read_text())
    cfg["environment"]["doors"] = _doors()
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_source_door_controls_select_clamp_and_leave_other_doors_unchanged():
    doors = _doors()
    controller = PlaybackController(3, False, doors=doors)
    for _ in range(20):
        controller.handle_key(ord("O"))
    assert controller.door_angles == {"entry": 95., "side": 0.}
    controller.handle_key(ord("n"))
    controller.handle_key(ord("o"))
    assert controller.selected_door["name"] == "side"
    assert controller.door_angles == {"entry": 95., "side": 10.}
    for _ in range(20):
        controller.handle_key(ord("L"))
    assert controller.door_angles["side"] == 0.
    controller.handle_key(ord("N"))
    assert controller.selected_door["name"] == "entry"
    assert doors[0]["angle_degrees"] == 30.


@pytest.mark.parametrize("results, doors", [(True, _doors()), (False, [])])
def test_door_keys_do_nothing_for_saved_results_or_scenes_without_doors(results, doors):
    controller = PlaybackController(3, results, doors=doors)
    initial = controller.door_angles.copy()
    for key in "OoLlNn":
        controller.handle_key(ord(key))
    assert controller.door_angles == initial
    assert controller.selected_door_index == 0


def test_preview_door_angle_survives_frame_changes_and_does_not_save(door_scene, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("door preview must not step physics"))
    original = door_scene.read_bytes()
    with prepare_scene_viewer(scene_ref=door_scene) as prepared:
        controller = PlaybackController(len(prepared.frames), False,
                                        doors=prepared.config["environment"]["doors"])
        controller.handle_key(ord("O"))
        controller.handle_key(ord("O"))
        joint = mujoco.mj_name2id(prepared.model, mujoco.mjtObj.mjOBJ_JOINT, "door_entry_hinge")
        leaf = mujoco.mj_name2id(prepared.model, mujoco.mjtObj.mjOBJ_GEOM, "door_entry_leaf")
        assert joint >= 0 and leaf >= 0
        for frame in (0, len(prepared.frames) - 1, 1):
            apply_playback_frame(prepared, frame, controller.door_angles)
            assert np.isclose(prepared.data.qpos[prepared.model.jnt_qposadr[joint]], np.deg2rad(50.))
            rotation = prepared.data.geom_xmat[leaf].reshape(3, 3)
            np.testing.assert_allclose(rotation[:, 0], [np.cos(np.deg2rad(50.)), np.sin(np.deg2rad(50.)), 0.], atol=1e-8)
        left, right = playback_status(prepared, controller)
        assert "Selected door" in left and "entry: 50 deg" in right
        assert "Preview only" in right
        assert "O open +10 deg" in right and "N next door" in right
    assert door_scene.read_bytes() == original
    with prepare_scene_viewer(scene_ref=door_scene) as fresh:
        joint = mujoco.mj_name2id(fresh.model, mujoco.mjtObj.mjOBJ_JOINT, "door_entry_hinge")
        assert np.isclose(fresh.data.qpos[fresh.model.jnt_qposadr[joint]], np.deg2rad(30.))


def test_result_frame_replay_restores_fixed_door_angle_even_with_override(door_scene):
    mujoco = pytest.importorskip("mujoco")
    with prepare_scene_viewer(scene_ref=door_scene) as prepared:
        frame = prepared.frames[0]
        transform = np.eye(4)
        transform[:3, :3] = prepared.data.xmat[prepared.rig_id].reshape(3, 3)
        transform[:3, 3] = prepared.data.xpos[prepared.rig_id]
        from wtumi_recon.rig_capture import head_camera_pose
        frame["captured_poses"] = {"T_wrig": transform,
                                   "T_rig_camera": head_camera_pose(prepared.config["camera"], frame["pose"])}
        # Mark the prepared model as result playback without loading a GL
        # window or a mesh: this exercises the exact captured-pose branch.
        prepared.mesh_geom_ids = [0]
        controller = PlaybackController(len(prepared.frames), True,
                                        doors=prepared.config["environment"]["doors"])
        joint = mujoco.mj_name2id(prepared.model, mujoco.mjtObj.mjOBJ_JOINT, "door_entry_hinge")
        prepared.data.qpos[prepared.model.jnt_qposadr[joint]] = np.deg2rad(90.)
        apply_playback_frame(prepared, 0, {"entry": 70.})
        assert np.isclose(prepared.data.qpos[prepared.model.jnt_qposadr[joint]], np.deg2rad(30.))
        left, right = playback_status(prepared, controller)
        assert "Captured door angles" in left and "entry: 30 deg" in right
        assert "controls disabled" in right
        assert "O open" not in right


@pytest.mark.parametrize("angles", [None, {}, {"entry": 90., "side": 0.},
                                   {"entry": 30., "side": False},
                                   {"entry": float("nan"), "side": 0.}])
def test_saved_door_records_cannot_change_angle_or_omit_state(angles):
    record = {} if angles is None else {"door_angles_degrees": angles}
    with pytest.raises(ValueError, match="captured door angles"):
        _validate_recorded_doors(record, {"environment": {"doors": _doors()}})


def test_legacy_records_without_doors_and_fixed_door_records_are_accepted():
    _validate_recorded_doors({}, {"environment": {}})
    _validate_recorded_doors({"door_angles_degrees": {"entry": 30., "side": 0.}},
                             {"environment": {"doors": _doors()}})


def test_native_loop_applies_door_key_and_suppresses_shared_visual_shortcuts(door_scene, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    import mujoco.viewer
    from wtumi_recon.scene_viewer import launch_scene_viewer

    class Handle:
        def __init__(self, model, data):
            self.model, self.data = model, data
            self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
            self.user_scn = mujoco.MjvScene(model, maxgeom=20)
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 1
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
            self.user_scn.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] = 1
            self.running = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.running = False

        @contextmanager
        def lock(self):
            yield

        def is_running(self):
            return self.running

        def sync(self):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_entry_hinge")
            assert np.isclose(self.data.qpos[self.model.jnt_qposadr[joint]], np.deg2rad(40.))
            assert self.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] == 0
            assert self.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] == 1
            assert self.user_scn.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] == 0
            self.running = False

    def launch(model, data, key_callback=None, **kwargs):
        key_callback(ord("O"))
        return Handle(model, data)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    launch_scene_viewer(scene_ref=door_scene)
