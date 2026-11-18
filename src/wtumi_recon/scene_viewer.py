"""Native MuJoCo scene inspection and saved reconstruction playback.

The viewer never advances physics or reconstructs from ground-truth geometry.
It replays prescribed poses and displays the mesh/TSDF saved after each prefix
of the captured RGB-D sequence. Scene-only inspection needs no reconstruction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from .rig_capture import _set_local_head_pose


MODES = {1: "Original scene", 2: "Reconstructed mesh", 3: "Observed TSDF surface-band voxels",
         4: "Original scene + reconstructed mesh"}
KEY_HELP = "P play/pause | [ previous | ] next | C head/free camera | 1 scene | 2 mesh | 3 voxels | 4 overlay | I images"
DOOR_KEY_HELP = "O open +10 deg | L close -10 deg | N next door"


def _inside(root, value, *, directory=False):
    """Resolve an artifact path without allowing absolute or escaping paths."""
    root = Path(root).resolve()
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("viewer artifact paths must be nonempty relative paths")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"viewer artifact path escapes result directory: {value}")
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(path)
    return path


def _pose(value, name):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must be a finite rigid homogeneous 4x4 transform")
    return pose


def _validate_recorded_doors(record, config):
    """Refuse to display a static reconstruction with different door poses."""
    from .scene_doors import configured_door_angles

    expected = configured_door_angles(config)
    recorded = record.get("door_angles_degrees", {} if not expected else None)
    if (not isinstance(recorded, dict) or set(recorded) != set(expected)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) or not np.isclose(value, expected[name], atol=1e-8, rtol=0)
                   for name, value in recorded.items())):
        raise ValueError("captured door angles disagree with the static scene configuration")


@dataclass(frozen=True)
class DisplayVoxels:
    centers: np.ndarray
    colors: np.ndarray
    size_m: float
    selected_count: int
    observed_count: int


def load_display_voxels(path, max_voxels=5000, band_voxels=1.0):
    """Load real observed surface-band cells, with a deterministic display cap.

    Sampling changes visualization only. The saved TSDF is never modified.
    Cells with zero integration weight are unknown and are never displayed.
    """
    if type(max_voxels) is not int or max_voxels < 1:
        raise ValueError("max_voxels must be a positive integer")
    if not np.isfinite(band_voxels) or band_voxels <= 0:
        raise ValueError("band_voxels must be finite and positive")
    with np.load(path, allow_pickle=False) as grid:
        n = int(grid["resolution"].item())
        size, trunc = float(grid["voxel_size_m"]), float(grid["sdf_trunc_m"])
        origin = np.asarray(grid["origin_m"], dtype=float)
        if n < 2 or origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite([size, trunc]).all() or min(size, trunc) <= 0:
            raise ValueError("invalid TSDF grid metadata")
        for key, expected in (("axis_order", "xyz"), ("flat_order", "C")):
            if key in grid and str(grid[key].item()) != expected:
                raise ValueError(f"unsupported TSDF {key}")
        if "tsdf_normalized" in grid and not bool(grid["tsdf_normalized"].item()):
            raise ValueError("viewer expects normalized TSDF values")
        sdf, weight, color = grid["tsdf"], grid["weight"], grid["color"]
        if sdf.shape != (n**3,) or weight.shape != sdf.shape or color.shape != (n**3, 3):
            raise ValueError("TSDF array dimensions do not match resolution")
        observed = np.isfinite(weight) & (weight > 0)
        selected = np.flatnonzero(observed & np.isfinite(sdf) & (np.abs(sdf) * trunc <= band_voxels * size))
        count = len(selected)
        if count > max_voxels:
            selected = selected[np.linspace(0, count - 1, max_voxels, dtype=np.int64)]
        indices = np.column_stack(np.unravel_index(selected, (n, n, n), order="C"))
        scale = float(grid["color_scale"]) if "color_scale" in grid else 255.0
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(color[selected]).all():
            raise ValueError("invalid TSDF colors or scale")
        return DisplayVoxels(origin + (indices + .5) * size, np.clip(color[selected] / scale, 0, 1),
                             size, count, int(observed.sum()))


class VoxelCache:
    """Bound retained display arrays, independently of trajectory length."""

    def __init__(self, capacity=2, max_voxels=5000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self.capacity, self.max_voxels = capacity, max_voxels
        self.entries = OrderedDict()

    def get(self, path):
        path = Path(path)
        if path not in self.entries:
            self.entries[path] = load_display_voxels(path, self.max_voxels)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(path)
        return self.entries[path]


class PreviewCache:
    """Retain thumbnail arrays for at most two frames."""

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.entries = OrderedDict()

    def get(self, frame, raw_size, comparison_size):
        from PIL import Image, ImageDraw

        key = (frame["id"], tuple(raw_size), tuple(comparison_size))
        if key not in self.entries:
            images = []
            for kind, title, size in (("raw_rgb", "Captured RGB", raw_size),
                                      ("comparison", "Held-out true / voxel-rendered RGB-D", comparison_size)):
                path = frame.get(kind)
                if path is None:
                    continue
                with Image.open(path) as source:
                    thumbnail = source.convert("RGB")
                    thumbnail.thumbnail((max(1, size[0]), max(1, size[1] - 24)))
                card = Image.new("RGB", (thumbnail.width, thumbnail.height + 24), "#111827")
                card.paste(thumbnail, (0, 24))
                ImageDraw.Draw(card).text((6, 5), title, fill="white")
                images.append((kind, np.asarray(card).copy()))
            self.entries[key] = images
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return self.entries[key]


@dataclass
class PreparedScene:
    """Validated native model and playback inputs; close to remove temp files."""

    config: dict
    model: object
    data: object
    frames: list
    mesh_geom_ids: list
    camera_id: int
    rig_id: int
    mocap_id: int
    temporary: object
    original_rgba: np.ndarray
    voxel_cache: VoxelCache = field(default_factory=VoxelCache)
    preview_cache: PreviewCache = field(default_factory=PreviewCache)

    @property
    def has_results(self):
        return bool(self.mesh_geom_ids)

    def close(self):
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _source_tree(model_path, allowed_root):
    tree = ET.parse(model_path)
    root = tree.getroot()
    if root.tag != "mujoco" or root.findall(".//include"):
        raise ValueError("viewer requires a built, self-contained MuJoCo scene")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    directories = {"mesh": compiler.get("meshdir", compiler.get("assetdir", "")),
                   "texture": compiler.get("texturedir", compiler.get("assetdir", ""))}
    for asset in root.findall("./asset/*"):
        if "file" in asset.attrib:
            asset_path = (model_path.parent / directories.get(asset.tag, "") / asset.get("file")).resolve()
            if not asset_path.is_relative_to(allowed_root):
                raise ValueError("MuJoCo asset path escapes the scene/result directory")
            if not asset_path.is_file():
                raise FileNotFoundError(asset_path)
            asset.set("file", str(asset_path))
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
    return tree


_MESH_CONVERTER = """
import json
import sys
import numpy as np
import open3d as o3d

for source, destination in json.load(sys.stdin):
    mesh = o3d.io.read_triangle_mesh(source)
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))):
        raise ValueError(f"saved reconstruction mesh is empty or invalid: {source}")
    with open(destination, "w", encoding="utf-8") as stream:
        for xyz in vertices:
            stream.write("v " + " ".join(f"{v:.17g}" for v in xyz) + "\\n")
        for face in triangles:
            stream.write("f " + " ".join(str(int(v) + 1) for v in face) + "\\n")
"""


def _convert_saved_meshes(frames, temporary_directory):
    """Isolate Open3D's bundled GLFW from the native MuJoCo process on macOS."""
    pairs = [(str(frame["mesh"]), str(temporary_directory / f"mesh_{i}.obj"))
             for i, frame in enumerate(frames) if "mesh" in frame]
    if not pairs:
        return []
    result = subprocess.run([sys.executable, "-c", _MESH_CONVERTER],
                            input=json.dumps(pairs), text=True, capture_output=True)
    if result.returncode:
        raise ValueError("saved mesh conversion failed: " + result.stderr.strip())
    return [Path(destination) for _, destination in pairs]


def _append_mesh(tree, output_path, index):
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    name = f"wtumi_reconstruction_{index}"
    # Shell inertia permits the open surfaces typical of RGB-D reconstruction.
    ET.SubElement(asset, "mesh", name=name, file=str(output_path), inertia="shell")
    ET.SubElement(root.find("worldbody"), "geom", name=name, type="mesh", mesh=name,
                  group="3", contype="0", conaffinity="0", rgba="0.12 0.75 0.88 1")
    return name


def prepare_scene_viewer(scene_ref=None, result_dir=None):
    """Validate/load all playback metadata before opening a native window.

    The returned object is a context manager. Loading models and converting
    saved meshes needs no graphics context; volume arrays are cached on demand.
    """
    if (scene_ref is None) == (result_dir is None):
        raise ValueError("supply exactly one of scene_ref or result_dir")
    import mujoco
    from .scene_bundle import build_scene, load_scene_config

    temporary = tempfile.TemporaryDirectory(prefix="wtumi_scene_viewer_")
    temp = Path(temporary.name).resolve()
    try:
        if result_dir is None:
            report = build_scene(scene_ref, temp / "scene")
            config = report["config"]
            allowed_root = temp
            model_path = _inside(temp / "scene", report["paths"]["model"])
            frames = [{"id": pose["id"], "timestamp_s": pose["timestamp_s"], "pose": pose,
                       "frames": i + 1} for i, pose in enumerate(config["trajectory"])]
        else:
            allowed_root = Path(result_dir).expanduser().resolve()
            run = json.loads((allowed_root / "scene_run.json").read_text(encoding="utf-8"))
            if not isinstance(run, dict) or run.get("format") != "wtumi-scene-run-v1":
                raise ValueError("expected a quick-scene result with scene_run.json")
            paths = run.get("paths", {})
            config = load_scene_config(_inside(allowed_root, paths.get("scene_config")))
            model_path = _inside(allowed_root, paths.get("model"))
            manifest_path = _inside(allowed_root, paths.get("capture_manifest"))
            _inside(allowed_root, paths.get("final_result"), directory=True)
            capture_path = _inside(allowed_root, str(manifest_path.parent.relative_to(allowed_root) / "capture.json"))
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            if not isinstance(capture, dict) or capture.get("format") != "wtumi-scene-capture-v1":
                raise ValueError("expected a quick-scene capture.json for exact pose replay")
            records = capture.get("frames", [])
            if not isinstance(records, list) or [v.get("id") for v in records] != [v["id"] for v in config["trajectory"]]:
                raise ValueError("capture records disagree with the configured trajectory")
            captured = {}
            for record in records:
                _validate_recorded_doors(record, config)
                world_rig = _pose(record.get("T_wrig"), "capture.T_wrig")
                world_camera = _pose(record.get("T_wc"), "capture.T_wc")
                captured[record["id"]] = {"T_wrig": world_rig, "T_wc": world_camera,
                                            "T_rig_camera": np.linalg.inv(world_rig) @ world_camera}
                relative = record.get("frame_directory", f"frames/{record['id']}")
                try:
                    raw = _inside(manifest_path.parent, str(Path(relative) / "raw_rgb.png"))
                except FileNotFoundError:
                    raw = None
                captured[record["id"]]["raw_rgb"] = raw
            history = json.loads(_inside(allowed_root, paths.get("history")).read_text(encoding="utf-8"))
            if not isinstance(history, list) or not history:
                raise ValueError("scene history must be a nonempty list")
            frames, previous = [], 0
            trajectory = config["trajectory"]
            for item in history:
                count = item.get("frames") if isinstance(item, dict) else None
                if type(count) is not int or not previous < count <= len(trajectory):
                    raise ValueError("history frame counts must increase within the capture trajectory")
                pose = trajectory[count - 1]
                if item.get("id") != pose["id"] or not np.isclose(item.get("timestamp_s", -1), pose["timestamp_s"]):
                    raise ValueError("history identifiers/timestamps disagree with the capture trajectory")
                directory = _inside(allowed_root, item.get("run_dir"), directory=True)
                comparison_relative = f"renders/{config['held_out'][0]['id']}/comparison.png"
                try:
                    comparison = _inside(directory, comparison_relative)
                except FileNotFoundError:
                    comparison = None
                frames.append({"id": pose["id"], "timestamp_s": pose["timestamp_s"], "pose": pose,
                               "frames": count, "mesh": _inside(directory, "mesh.ply"),
                               "voxels": _inside(directory, "tsdf_volume.npz"),
                               "captured_poses": captured[pose["id"]],
                               "raw_rgb": captured[pose["id"]]["raw_rgb"], "comparison": comparison})
                previous = count
            if previous != len(trajectory):
                raise ValueError("scene history must include the final cumulative reconstruction")
        tree = _source_tree(model_path, allowed_root)
        converted_meshes = _convert_saved_meshes(frames, temp)
        names = [_append_mesh(tree, path, i) for i, path in enumerate(converted_meshes)]
        display_path = temp / "display.xml"
        tree.write(display_path, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(display_path))
        data = mujoco.MjData(model)
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sensor_rig")
        if camera < 0 or rig < 0 or model.cam_bodyid[camera] != rig or model.body_mocapid[rig] < 0:
            raise ValueError("scene requires head_camera attached to a mocap sensor_rig")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
        voxel_cache = VoxelCache()
        for frame in frames:
            if "voxels" in frame:
                # Fail on corrupt saved volumes before creating a GL window;
                # the cache still retains only two small display arrays.
                voxel_cache.get(frame["voxels"])
        prepared = PreparedScene(config, model, data, frames, ids, camera, rig,
                                 int(model.body_mocapid[rig]), temporary, model.geom_rgba.copy(),
                                 voxel_cache=voxel_cache)
        apply_playback_frame(prepared, 0)
        return prepared
    except BaseException:
        temporary.cleanup()
        raise


@dataclass
class PlaybackController:
    frame_count: int
    has_results: bool
    fps: float = 3.0
    frame_index: int = 0
    playing: bool = False
    mode: int = 1
    head_camera: bool = False
    show_images: bool = False
    last_tick: float | None = None
    doors: list = field(default_factory=list)
    selected_door_index: int = 0
    door_angles: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.frame_count < 1 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("playback requires frames and positive fps")
        self.door_angles = {door["name"]: float(door["angle_degrees"]) for door in self.doors}

    @property
    def selected_door(self):
        return self.doors[self.selected_door_index] if self.doors else None

    def handle_key(self, key):
        """Handle GLFW ASCII keys without touching MuJoCo's render thread."""
        if key in (ord("P"), ord("p")):
            self.playing = not self.playing
            self.last_tick = None
        elif key in (ord("["), ord("]")):
            self.frame_index = min(self.frame_count - 1, max(0, self.frame_index + (1 if key == ord("]") else -1)))
            self.playing, self.last_tick = False, None
        elif key in (ord("C"), ord("c")):
            self.head_camera = not self.head_camera
        elif key in (ord("I"), ord("i")) and self.has_results:
            self.show_images = not self.show_images
        elif self.doors and not self.has_results and key in map(ord, "OoLlNn"):
            if key in (ord("N"), ord("n")):
                self.selected_door_index = (self.selected_door_index + 1) % len(self.doors)
            else:
                door = self.selected_door
                step = 10.0 if key in (ord("O"), ord("o")) else -10.0
                self.door_angles[door["name"]] = float(np.clip(
                    self.door_angles[door["name"]] + step, *door["range_degrees"]))
        elif key in map(ord, "1234"):
            value = int(chr(key))
            if value == 1 or self.has_results:
                self.mode = value

    def advance(self, now):
        if not self.playing:
            self.last_tick = None
            return
        if self.last_tick is None:
            self.last_tick = now
        elapsed = now - self.last_tick
        steps = int((elapsed + 1e-9) * self.fps)
        if steps > 0:
            self.frame_index = (self.frame_index + steps) % self.frame_count
            self.last_tick += steps / self.fps


def apply_playback_frame(prepared, frame_index, door_angles=None):
    """Replay poses and apply preview door angles without changing artifacts.

    Saved reconstruction playback always restores the configured capture door
    state. Only a source-scene preview accepts interactive angle overrides.
    """
    import mujoco
    from .scene_capture import apply_scene_pose
    from .scene_doors import set_scene_doors

    frame = prepared.frames[frame_index]
    if "captured_poses" in frame:
        transform = frame["captured_poses"]["T_wrig"]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).reshape(9))
        prepared.data.mocap_pos[prepared.mocap_id] = transform[:3, 3]
        prepared.data.mocap_quat[prepared.mocap_id] = quat
        local = frame["captured_poses"]["T_rig_camera"]
        prepared.model.cam_pos[prepared.camera_id] = local[:3, 3]
        _set_local_head_pose(mujoco, prepared.model, prepared.camera_id, local)
        mujoco.mj_forward(prepared.model, prepared.data)
    else:
        apply_scene_pose(mujoco, prepared.model, prepared.data, prepared.config, frame["pose"])
    set_scene_doors(mujoco, prepared.model, prepared.data, prepared.config,
                    angles=None if prepared.has_results else door_angles)
    mujoco.mj_forward(prepared.model, prepared.data)
    prepared.data.time = prepared.frames[frame_index]["timestamp_s"]


def apply_display_mode(prepared, controller, option, user_scene=None):
    """Select the actual saved mesh or surface cells for the current prefix."""
    import mujoco

    if controller.mode not in MODES or (controller.mode != 1 and not prepared.has_results):
        raise ValueError("display mode is unavailable for this input")
    option.geomgroup[:] = 0
    option.geomgroup[0:3] = controller.mode in (1, 4)
    option.geomgroup[3] = controller.mode in (2, 4)
    if prepared.mesh_geom_ids:
        prepared.model.geom_rgba[prepared.mesh_geom_ids, 3] = 0
        current = prepared.mesh_geom_ids[controller.frame_index]
        prepared.model.geom_rgba[current, 3] = 0.7 if controller.mode == 4 else 1.0
    if user_scene is not None:
        user_scene.ngeom = 0
    voxels = None
    if controller.mode == 3:
        voxels = prepared.voxel_cache.get(prepared.frames[controller.frame_index]["voxels"])
        if user_scene is not None:
            if len(voxels.centers) > user_scene.maxgeom:
                raise ValueError("MuJoCo user scene has insufficient voxel display capacity")
            for index, (center, color) in enumerate(zip(voxels.centers, voxels.colors)):
                mujoco.mjv_initGeom(user_scene.geoms[index], mujoco.mjtGeom.mjGEOM_BOX,
                                   np.full(3, voxels.size_m * .47), center,
                                   np.eye(3).reshape(9), np.r_[color, 1.0].astype(np.float32))
            user_scene.ngeom = len(voxels.centers)
    return voxels


def playback_status(prepared, controller, voxels=None):
    frame = prepared.frames[controller.frame_index]
    source = "Saved cumulative reconstruction playback" if prepared.has_results else "Prescribed camera/rig pose preview"
    rows = [("Scene", prepared.config["scene_name"]), ("Playback", source),
            ("Mode", MODES[controller.mode]),
            ("Frame", f"{controller.frame_index + 1}/{controller.frame_count} | {frame['id']} | t={frame['timestamp_s']:.3f} s"),
            ("State", f"{'Playing' if controller.playing else 'Paused'} at {controller.fps:g} frames/s"),
            ("Camera", "Head camera" if controller.head_camera else "Free orbit"),
            ("Motion", "Kinematic replay; no physics stepping")]
    if prepared.has_results:
        rows.append(("Integrated observations", str(frame["frames"])))
        if controller.mode in (2, 4):
            rows.append(("Mesh display", "Uniform cyan; captured RGB stays in saved artifacts"))
    if voxels is not None:
        rows.append(("Surface cells displayed / selected", f"{len(voxels.centers):,} / {voxels.selected_count:,}"))
        rows.append(("Voxel display", "Observed surface band only; display sampled when capped"))
    rows.append(("Controls", "P play/pause | [ / ] previous/next | C head/free"))
    if prepared.has_results:
        rows.append(("Display keys", "1 scene | 2 mesh | 3 voxels | 4 overlay | I images"))
        rows.append(("Saved RGB-D insets", "Shown" if controller.show_images else "Hidden; press I to show"))
    doors = prepared.config["environment"].get("doors", [])
    if doors:
        if prepared.has_results:
            angles = ", ".join(f"{door['name']}: {door['angle_degrees']:g} deg" for door in doors)
            rows.append(("Captured door angles", angles))
            rows.append(("Door state", "Fixed to the captured scene; controls disabled"))
        else:
            selected = controller.selected_door or doors[0]
            angle = controller.door_angles.get(selected["name"], selected["angle_degrees"])
            rows.append(("Selected door", f"{selected['name']}: {angle:g} deg"))
            rows.append(("Door controls", DOOR_KEY_HELP))
            rows.append(("Door edits", "Preview only; set angle_degrees in scene.yaml before capture"))
    return "\n".join(a for a, _ in rows), "\n".join(b for _, b in rows)


def update_view_camera(camera, controller, head_id, free_camera, previous_head, restore_orbit):
    """Override native shortcut side effects while preserving mouse navigation."""
    import mujoco

    if controller.head_camera:
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = head_id
    else:
        if previous_head or restore_orbit:
            camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = free_camera
        else:
            free_camera = (camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
    return free_camera


def launch_scene_viewer(scene_ref=None, result_dir=None):
    """Open the native MuJoCo viewer (use mjpython on macOS).

    All playback snapshots are precomputed reconstruction results. Press P to
    replay camera motion and cumulative map growth; this does not run fusion.
    """
    import mujoco
    import mujoco.viewer

    with prepare_scene_viewer(scene_ref, result_dir) as prepared:
        controller = PlaybackController(len(prepared.frames), prepared.has_results,
                                        prepared.config.get("playback", {}).get("fps", 3.0),
                                        doors=prepared.config["environment"].get("doors", []))
        keys = SimpleQueue()
        print(KEY_HELP if prepared.has_results else "P play/pause | [ previous | ] next | C head/free camera", flush=True)
        if controller.doors and not prepared.has_results:
            print(DOOR_KEY_HELP + " (preview only; edit scene.yaml to save)", flush=True)
        print("Saved-result/pose playback only; no live reconstruction or physics stepping.", flush=True)
        with mujoco.viewer.launch_passive(prepared.model, prepared.data, key_callback=keys.put,
                                          show_left_ui=False, show_right_ui=False) as handle:
            with handle.lock():
                if not handle.is_running():
                    return
                mujoco.mjv_defaultFreeCamera(prepared.model, handle.cam)
                handle.cam.azimuth, handle.cam.elevation = 125, -25
                free_camera = (handle.cam.lookat.copy(), handle.cam.distance, handle.cam.azimuth, handle.cam.elevation)
            last_state, last_head, image_state = None, False, None
            while handle.is_running():
                started = time.monotonic()
                restore_camera = False
                while True:
                    try:
                        key = keys.get_nowait()
                        controller.handle_key(key)
                        # MuJoCo can bind number keys to geom-group toggles.
                        # Reassert our mode even when the selected mode repeats.
                        if key in map(ord, "1234"):
                            last_state = None
                        if key in map(ord, "1234PpIiCcOoLlNn[]"):
                            restore_camera = True
                    except Empty:
                        break
                controller.advance(started)
                state = (controller.frame_index, controller.mode, tuple(controller.door_angles.items()))
                with handle.lock():
                    # is_running can change between the loop condition and
                    # acquiring the lock. Stop touching shared state once
                    # the native window has begun shutting down.
                    if not handle.is_running():
                        break
                    # These MuJoCo defaults share C/P/I with our documented
                    # controls. Keep their unrelated visual effects disabled.
                    for flag in (mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
                                 mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
                                 mujoco.mjtVisFlag.mjVIS_INERTIA):
                        handle.opt.flags[flag] = 0
                    if controller.doors:
                        # O/N/L also toggle perturbation markers, island
                        # colors and additive rendering in the native UI.
                        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
                        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
                        if handle.user_scn is not None:
                            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] = 0
                    if state != last_state:
                        apply_playback_frame(prepared, controller.frame_index, controller.door_angles)
                        voxels = apply_display_mode(prepared, controller, handle.opt, handle.user_scn)
                        last_state = state
                    # MuJoCo also uses number keys for camera selection. Keep
                    # the last mouse-controlled orbit through our shortcuts.
                    free_camera = update_view_camera(handle.cam, controller, prepared.camera_id,
                                                     free_camera, last_head, restore_camera)
                    last_head = controller.head_camera
                # Overlay setters synchronize internally; do not invoke them
                # while holding the viewer state lock (lock-order deadlock).
                if not handle.is_running():
                    break
                if hasattr(handle, "set_texts"):
                    left, right = playback_status(prepared, controller, voxels)
                    handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, None, left, right))
                if not handle.is_running():
                    break
                if hasattr(handle, "set_images"):
                    next_image_state = (controller.show_images, controller.frame_index)
                    if next_image_state != image_state:
                        overlays = []
                        if controller.show_images:
                            # Fixed thumbnail bounds avoid MuJoCo's native
                            # viewport getter, which races during window close
                            # and can block when used under the state lock.
                            previews = prepared.preview_cache.get(prepared.frames[controller.frame_index],
                                (320, 240), (640, 420))
                            for kind, pixels in previews:
                                h, w = pixels.shape[:2]
                                left = 12 if kind == "raw_rgb" else 360
                                overlays.append((mujoco.MjrRect(left, 12, w, h), pixels))
                        if not handle.is_running():
                            break
                        handle.set_images(overlays)
                        image_state = next_image_state
                handle.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - started)))
