"""Headless checks for native scene/result playback and its real geometry."""

import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wtumi_recon.mujoco_capture import opencv_pose_from_mujoco
from wtumi_recon.rig_capture import head_camera_pose, rig_transform
from wtumi_recon.scene_viewer import (
    PlaybackController, PreviewCache, VoxelCache, apply_display_mode, apply_playback_frame,
    load_display_voxels, playback_status, prepare_scene_viewer,
    update_view_camera,
)


def _volume(path, observed_count=16):
    n = 4
    tsdf = np.ones(n**3, dtype=np.float32)
    weight = np.zeros(n**3, dtype=np.float32)
    tsdf[:observed_count] = 0
    weight[:observed_count] = 1
    # An unobserved zero crossing must never become a displayed surface cell.
    tsdf[-1] = 0
    np.savez_compressed(path, tsdf=tsdf, weight=weight,
                        color=np.full((n**3, 3), 127.5, dtype=np.float32),
                        origin_m=np.array([2., 3., 4.]), resolution=n,
                        voxel_size_m=.1, sdf_trunc_m=.3, color_scale=255.,
                        tsdf_normalized=True, axis_order="xyz", flat_order="C")


def test_voxel_selection_uses_observation_weight_and_exact_xyz_centers(tmp_path):
    path = tmp_path / "volume.npz"
    _volume(path, 6)
    voxels = load_display_voxels(path, max_voxels=20)
    indices = np.column_stack(np.unravel_index(np.arange(6), (4, 4, 4)))
    np.testing.assert_allclose(voxels.centers, [2., 3., 4.] + (indices + .5) * .1)
    np.testing.assert_allclose(voxels.colors, .5)
    assert voxels.selected_count == voxels.observed_count == 6


def test_voxel_display_cap_is_deterministic_and_does_not_modify_snapshot(tmp_path):
    path = tmp_path / "volume.npz"
    _volume(path, 30)
    before = path.read_bytes()
    a, b = load_display_voxels(path, 7), load_display_voxels(path, 7)
    assert len(a.centers) == 7 and a.selected_count == 30
    np.testing.assert_array_equal(a.centers, b.centers)
    assert path.read_bytes() == before


def test_voxel_cache_retains_only_two_display_frames(tmp_path):
    paths = [tmp_path / f"{i}.npz" for i in range(3)]
    for path in paths:
        _volume(path)
    cache = VoxelCache(capacity=2, max_voxels=8)
    first = cache.get(paths[0])
    cache.get(paths[1])
    assert cache.get(paths[0]) is first
    cache.get(paths[2])
    assert list(cache.entries) == [paths[0], paths[2]]
    assert all(len(v.centers) <= 8 for v in cache.entries.values())


def test_empty_observed_surface_is_a_valid_empty_display(tmp_path):
    path = tmp_path / "empty.npz"
    _volume(path, 0)
    voxels = load_display_voxels(path)
    assert voxels.centers.shape == (0, 3)
    assert voxels.selected_count == 0


def test_controller_frame_keys_modes_camera_and_loop():
    control = PlaybackController(3, True, fps=2)
    control.handle_key(ord("["))
    assert control.frame_index == 0
    control.handle_key(ord("P"))
    control.advance(0.)
    control.advance(.49)
    assert control.frame_index == 0
    control.advance(.5)
    assert control.frame_index == 1
    control.advance(1.5)
    assert control.frame_index == 0
    control.handle_key(ord("]"))
    assert control.frame_index == 1 and not control.playing
    control.handle_key(ord("3"))
    control.handle_key(ord("C"))
    assert control.mode == 3 and control.head_camera
    control.handle_key(ord("I"))
    assert control.show_images
    scene_only = PlaybackController(1, False)
    scene_only.handle_key(ord("4"))
    scene_only.handle_key(ord("I"))
    assert scene_only.mode == 1
    assert not scene_only.show_images


@pytest.mark.parametrize("kwargs", [{}, {"scene_ref": "room", "result_dir": "unused"}])
def test_input_requires_exactly_one_scene_or_result(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        prepare_scene_viewer(**kwargs)


@pytest.fixture
def saved_run(tmp_path):
    pytest.importorskip("mujoco")
    import open3d as o3d
    from PIL import Image
    from wtumi_recon.scene_bundle import build_scene

    root = tmp_path / "run"
    root.mkdir()
    built = build_scene("room", root / "scene")
    cfg = built["config"]
    (root / "capture").mkdir()
    (root / "capture/manifest.json").write_text("{}")
    records = []
    for pose in cfg["trajectory"]:
        world_rig = rig_transform(pose["position_m"], pose["yaw_degrees"])
        # The native replay should use recorded transforms, not recomputed
        # config transforms: mimic a captured mount/rig translation offset.
        world_rig[0, 3] += .123
        world_camera = world_rig @ head_camera_pose(cfg["camera"], pose)
        records.append({"id": pose["id"], "T_wrig": world_rig.tolist(), "T_wc": world_camera.tolist()})
        raw_dir = root / "capture/frames" / pose["id"]
        raw_dir.mkdir(parents=True)
        Image.new("RGB", (320, 240), "#ab3400").save(raw_dir / "raw_rgb.png")
    (root / "capture/capture.json").write_text(json.dumps({"format": "wtumi-scene-capture-v1", "frames": records}))
    count = len(cfg["trajectory"])
    history = []
    vertices = np.array([[0., 0., 0.], [.3, 0., 0.], [0., .4, 0.], [0., 0., .5]]) + [1.2, 2.3, .7]
    for index, total in enumerate((1, count // 2, count)):
        directory = root / "history" / f"frame_{total:03d}"
        directory.mkdir(parents=True)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices + [index * .1, 0, 0])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        assert o3d.io.write_triangle_mesh(str(directory / "mesh.ply"), mesh)
        _volume(directory / "tsdf_volume.npz", 5 + index)
        comparison_dir = directory / "renders" / cfg["held_out"][0]["id"]
        comparison_dir.mkdir(parents=True)
        Image.new("RGB", (960, 640), "#0034ab").save(comparison_dir / "comparison.png")
        pose = cfg["trajectory"][total - 1]
        history.append({"id": pose["id"], "frames": total, "timestamp_s": pose["timestamp_s"],
                        "run_dir": str(directory.relative_to(root))})
    (root / "history.json").write_text(json.dumps(history))
    (root / "scene_run.json").write_text(json.dumps({"format": "wtumi-scene-run-v1", "paths": {
        "scene_config": "scene/scene.yaml", "model": "scene/model.xml",
        "capture_manifest": "capture/manifest.json", "history": "history.json",
        "final_result": history[-1]["run_dir"],
    }}))
    return root, cfg, records, vertices


def test_native_scene_only_model_replays_without_stepping_physics(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("viewer must not step physics"))
    with prepare_scene_viewer(scene_ref="room") as prepared:
        temp = Path(prepared.temporary.name)
        assert not prepared.has_results
        apply_playback_frame(prepared, len(prepared.frames) - 1)
        pose = prepared.frames[-1]["pose"]
        expected = rig_transform(pose["position_m"], pose["yaw_degrees"]) @ head_camera_pose(prepared.config["camera"], pose)
        actual = opencv_pose_from_mujoco(prepared.data.cam_xpos[prepared.camera_id], prepared.data.cam_xmat[prepared.camera_id])
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        assert temp.is_dir()
    assert not temp.exists()


def test_saved_meshes_keep_world_coordinates_and_selected_prefix(saved_run):
    import mujoco

    root, cfg, records, vertices = saved_run
    originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with prepare_scene_viewer(result_dir=root) as prepared:
        assert prepared.has_results and len(prepared.mesh_geom_ids) == 3
        geom_id = prepared.mesh_geom_ids[1]
        mesh_id = int(prepared.model.geom_dataid[geom_id])
        start, count = int(prepared.model.mesh_vertadr[mesh_id]), int(prepared.model.mesh_vertnum[mesh_id])
        native_vertices = prepared.model.mesh_vert[start:start + count]
        world_vertices = native_vertices @ prepared.data.geom_xmat[geom_id].reshape(3, 3).T + prepared.data.geom_xpos[geom_id]
        distances = np.linalg.norm(world_vertices[:, None, :] - (vertices + [.1, 0, 0])[None, :, :], axis=-1)
        assert np.max(np.min(distances, axis=0)) < 1e-6
        control = PlaybackController(3, True, frame_index=1, mode=2)
        option = mujoco.MjvOption()
        apply_display_mode(prepared, control, option)
        np.testing.assert_array_equal(option.geomgroup[:4], [0, 0, 0, 1])
        np.testing.assert_allclose(prepared.model.geom_rgba[prepared.mesh_geom_ids, 3], [0, 1, 0])
        apply_playback_frame(prepared, 2)
        actual = opencv_pose_from_mujoco(prepared.data.cam_xpos[prepared.camera_id], prepared.data.cam_xmat[prepared.camera_id])
        np.testing.assert_allclose(actual, records[-1]["T_wc"], atol=1e-8)
        control.mode = 4
        apply_display_mode(prepared, control, option)
        np.testing.assert_array_equal(option.geomgroup[:4], [1, 1, 1, 1])
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_native_voxel_mode_displays_saved_cells_and_labels_sampling(saved_run):
    import mujoco

    root, *_ = saved_run
    with prepare_scene_viewer(result_dir=root) as prepared:
        prepared.voxel_cache = VoxelCache(max_voxels=3)
        controller = PlaybackController(3, True, mode=3)
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(prepared.model, maxgeom=10)
        voxels = apply_display_mode(prepared, controller, option, scene)
        assert scene.ngeom == 3 and voxels.selected_count == 5
        np.testing.assert_array_equal(option.geomgroup[:4], 0)
        np.testing.assert_allclose(scene.geoms[0].pos, voxels.centers[0], atol=1e-6)
        left, right = playback_status(prepared, controller, voxels)
        assert "Surface cells displayed / selected" in left
        assert "3 / 5" in right and "Saved cumulative reconstruction playback" in right
        controller.mode = 1
        apply_display_mode(prepared, controller, option, scene)
        assert scene.ngeom == 0


def test_result_path_escape_rejected_before_native_model_creation(saved_run):
    root, *_ = saved_run
    path = root / "scene_run.json"
    run = json.loads(path.read_text())
    run["paths"]["model"] = "../outside.xml"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="escapes"):
        prepare_scene_viewer(result_dir=root)


def test_history_must_match_capture_and_include_final_map(saved_run):
    root, *_ = saved_run
    path = root / "history.json"
    history = json.loads(path.read_text())
    path.write_text(json.dumps(history[:-1]))
    with pytest.raises(ValueError, match="final cumulative"):
        prepare_scene_viewer(result_dir=root)


def test_native_image_insets_use_matching_saved_frame_and_prefix_with_bounded_cache(saved_run):
    root, *_ = saved_run
    with prepare_scene_viewer(result_dir=root) as prepared:
        cache = PreviewCache(capacity=2)
        previews = cache.get(prepared.frames[0], (160, 130), (280, 190))
        assert [kind for kind, _ in previews] == ["raw_rgb", "comparison"]
        raw, comparison = (pixels for _, pixels in previews)
        assert raw.shape[0] <= 130 and raw.shape[1] <= 160
        assert comparison.shape[0] <= 190 and comparison.shape[1] <= 280
        np.testing.assert_array_equal(raw[-1, -1], [171, 52, 0])
        np.testing.assert_array_equal(comparison[-1, -1], [0, 52, 171])
        for frame in prepared.frames[1:]:
            cache.get(frame, (160, 130), (280, 190))
        assert len(cache.entries) == 2
        assert all(key[0] != prepared.frames[0]["id"] for key in cache.entries)


def test_corrupt_saved_volume_rejected_before_opening_native_window(saved_run):
    root, *_ = saved_run
    path = next(root.glob("history/*/tsdf_volume.npz"))
    path.write_text("incomplete snapshot")
    with pytest.raises(ValueError):
        prepare_scene_viewer(result_dir=root)


@pytest.mark.parametrize("source", ["scene", "result"])
def test_native_preparation_does_not_load_open3d_glfw_into_viewer_process(saved_run, source):
    root, *_ = saved_run
    code = """
import sys
from wtumi_recon.scene_viewer import prepare_scene_viewer
options = {'scene_ref': 'room'} if sys.argv[1] == 'scene' else {'result_dir': sys.argv[2]}
with prepare_scene_viewer(**options):
    assert 'open3d' not in sys.modules, 'Open3D/GLFW must stay outside the native viewer process'
"""
    result = subprocess.run([sys.executable, "-c", code, source, str(root)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_shortcuts_preserve_free_orbit_but_mouse_updates_and_head_toggle_work():
    mujoco = pytest.importorskip("mujoco")
    camera = mujoco.MjvCamera()
    controller = PlaybackController(3, True)
    free = (np.array([1., 2., 3.]), 4., 125., -25.)
    # Simulate a built-in numeric shortcut changing camera and zoom.
    camera.type, camera.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 9
    camera.lookat[:], camera.distance = [0, 0, 0], .1
    free = update_view_camera(camera, controller, 2, free, False, True)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE and camera.fixedcamid == -1
    np.testing.assert_allclose(camera.lookat, [1, 2, 3])
    assert camera.distance == 4.
    # Ordinary mouse navigation becomes the next preserved free pose.
    camera.lookat[:], camera.distance = [2, 3, 4], 5.
    free = update_view_camera(camera, controller, 2, free, False, False)
    assert free[1] == 5.
    controller.head_camera = True
    free = update_view_camera(camera, controller, 2, free, False, True)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED and camera.fixedcamid == 2
    controller.head_camera = False
    update_view_camera(camera, controller, 2, free, True, True)
    np.testing.assert_allclose(camera.lookat, [2, 3, 4])
    assert camera.distance == 5.


@pytest.mark.parametrize("case", ["source", "result_images", "close_before_lock"])
def test_native_loop_never_queries_viewport_and_keeps_setters_outside_lock(saved_run, monkeypatch, case):
    import mujoco
    import mujoco.viewer
    from wtumi_recon.scene_viewer import launch_scene_viewer

    root, *_ = saved_run

    class Handle:
        def __init__(self):
            self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
            self.user_scn = None
            self.running, self.locked, self.lock_count = True, False, 0
            self.viewport_reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.running = False

        @contextmanager
        def lock(self):
            self.lock_count += 1
            if case == "close_before_lock" and self.lock_count == 2:
                self.running = False
            self.locked = True
            yield
            self.locked = False

        def is_running(self):
            return self.running

        @property
        def viewport(self):
            self.viewport_reads += 1
            pytest.fail("native playback must never query the unsafe viewport getter")

        def set_texts(self, _):
            assert not self.locked and self.running

        def set_images(self, images):
            assert not self.locked and self.running
            assert len(images) == (2 if case == "result_images" else 0)
            if images:
                assert [(rectangle.left, rectangle.bottom) for rectangle, _ in images] == [(12, 12), (360, 12)]
                assert images[0][0].width <= 320 and images[0][0].height <= 240
                assert images[1][0].width <= 640 and images[1][0].height <= 420

        def sync(self):
            self.running = False

    handle = Handle()

    def launch(*_, key_callback=None, **kwargs):
        if case == "result_images":
            key_callback(ord("I"))
        return handle

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    if case == "result_images":
        launch_scene_viewer(result_dir=root)
        assert handle.viewport_reads == 0
    else:
        launch_scene_viewer(scene_ref="room")
        assert handle.viewport_reads == 0
