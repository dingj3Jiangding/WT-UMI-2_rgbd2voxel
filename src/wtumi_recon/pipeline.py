"""Configuration-driven static-scene RGB-D -> TSDF -> mesh -> RGB-D."""

from __future__ import annotations

import copy
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
import yaml

from .config import load_run_config
from .diagnostics import save_comparison
from .evaluation import evaluate_rgbd, summarize_views
from .render import render_mesh_rgbd
from .synthetic import SyntheticScene
from .sensor import corrupt_depth
from .occupancy import EvidenceOccupancyGrid
from .occupancy_diagnostics import save_occupancy_slices
from .tsdf import Open3DTSDFMap, TSDFConfig


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_rgbd(directory, rgb, depth):
    directory.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(directory / "rgb.png"), o3d.geometry.Image(np.ascontiguousarray(rgb))):
        raise IOError(f"failed to write RGB to {directory}")
    np.save(directory / "depth_m.npy", depth)


def make_rgbd(rgb, depth_m, depth_trunc_m):
    """Preserve float32 optical-axis depth; do not quantize through millimetres."""
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
        o3d.geometry.Image(np.ascontiguousarray(depth_m, dtype=np.float32)),
        depth_scale=1.0, depth_trunc=depth_trunc_m, convert_rgb_to_intensity=False,
    )


class ReconstructionState:
    """Process-local cumulative mapping state for immutable sequence prefixes.

    Supply one fresh instance per observation history. Each successful call must
    strictly extend the previous prefix. Source file identity and metadata are
    checked before reuse; do not modify source assets while a run is executing.
    If a call fails after mapping starts, discard the state and replay afresh.
    """

    def __init__(self):
        self._settings = None
        self._frames = []
        self._backend = None
        self._occupancy = None
        self._integration_seconds = 0.0
        self._occupancy_seconds = 0.0
        self._occupancy_frames = []
        self._unfinished = False


def _prefix_identity(cfg, sequence):
    if sequence is None:
        raise ValueError("prefix_state requires an immutable RGB-D sequence source")
    settings = {key: cfg[key] for key in
                ("camera", "reconstruction", "occupancy", "sensor", "world_up")}
    settings["scene_name"] = sequence.scene_name
    settings["reference_kind"] = sequence.reference_kind
    records = {row["id"]: row for row in sequence.manifest["frames"]}
    frames = []
    for view in cfg["views"]:
        if view["split"] != "integration":
            continue
        record = records[view["id"]]
        assets = {}
        for key in ("rgb", "depth_m", "reference_rgb", "reference_depth_m"):
            if key not in record:
                continue
            path = (sequence.root / record[key]).resolve()
            stat = path.stat()
            assets[key] = (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                           stat.st_mtime_ns, stat.st_ctime_ns)
        frames.append({"view": copy.deepcopy(view), "assets": assets})
    return copy.deepcopy(settings), frames


def run_reconstruction(config_path, output_dir, *, prefix_state=None, compact=False, snapshot_volume=False):
    """Reconstruct once, or extend a process-local cumulative sequence map.

    Compact snapshots evaluate/export held-out views, mesh and occupancy only;
    the final call should use compact=False to retain the complete result API.
    Set snapshot_volume=True to retain dense uniform TSDF data during compact
    playback export as well. This does not fuse any additional observations.
    """
    started = perf_counter()
    cfg = load_run_config(config_path)
    sequence = None
    if cfg["source"]["type"] == "rgbd_sequence":
        from .sequence import RGBDSequence
        sequence = RGBDSequence(cfg["source"]["manifest"])
        scene = None
    else:
        scene = SyntheticScene.from_config(cfg["scene"])
    if type(compact) is not bool:
        raise ValueError("compact must be a boolean")
    if type(snapshot_volume) is not bool:
        raise ValueError("snapshot_volume must be a boolean")
    identity, prefix_frames = None, None
    if prefix_state is not None:
        if not isinstance(prefix_state, ReconstructionState):
            raise TypeError("prefix_state must be a ReconstructionState")
        if prefix_state._unfinished:
            raise ValueError("prefix state belongs to a failed/in-progress run; create a fresh state")
        identity, prefix_frames = _prefix_identity(cfg, sequence)
        if prefix_state._settings is not None:
            previous = prefix_state._frames
            if identity != prefix_state._settings:
                raise ValueError("prefix mapping settings or calibration changed")
            if len(prefix_frames) <= len(previous) or prefix_frames[:len(previous)] != previous:
                raise ValueError("prefix must strictly extend unchanged observations, poses and source files")
    settings = TSDFConfig(**cfg["reconstruction"])
    if settings.backend == "uniform" and scene is not None:
        lo, hi = np.asarray(settings.origin_m), np.asarray(settings.origin_m) + settings.volume_length_m
        if np.any(scene.mesh.get_min_bound() < lo) or np.any(scene.mesh.get_max_bound() > hi):
            raise ValueError("scene geometry exceeds uniform volume bounds; adjust origin_m/volume_length_m")
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty; choose a new --output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_source.yaml").write_text(Path(cfg["_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    resolved = {k: v for k, v in cfg.items() if k != "_path"}
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    cam = cfg["camera"]
    k = np.asarray(cam["K"], dtype=float)
    width, height = cam["width"], cam["height"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    occupancy_cfg = cfg["occupancy"]
    extending = prefix_state is not None and prefix_state._settings is not None
    backend = prefix_state._backend if extending else Open3DTSDFMap(settings)
    occupancy = (prefix_state._occupancy if extending else
        EvidenceOccupancyGrid(settings.origin_m, settings.volume_length_m / occupancy_cfg["resolution"],
                              occupancy_cfg["resolution"]) if occupancy_cfg["enabled"] else None)
    occupancy_seconds = prefix_state._occupancy_seconds if extending else 0.0
    occupancy_frames = copy.deepcopy(prefix_state._occupancy_frames) if extending else []
    references, observations, frame_records = {}, {}, []
    integrated_ids = [frame["view"]["id"] for frame in prefix_state._frames] if extending else []
    prior_ids = set(integrated_ids)
    reference_kind = sequence.reference_kind if sequence is not None else "clean_synthetic_ground_truth"
    source_metadata = sequence.metadata if sequence is not None else {"producer": "open3d_synthetic"}
    if sequence is not None:
        (out / "source_manifest.json").write_text(Path(cfg["source"]["manifest"]).read_text(encoding="utf-8"), encoding="utf-8")

    def clipped_pair(rgb, depth):
        rgb, depth = np.array(rgb, dtype=np.uint8, copy=True), np.array(depth, dtype=np.float32, copy=True)
        depth[(depth <= 0) | (depth >= cam["depth_trunc_m"]) | ~np.isfinite(depth)] = 0
        rgb[depth == 0] = 0
        return rgb, depth

    def read_observation(view, reference=False):
        if sequence is not None:
            pair = sequence.read_reference(view) if reference else sequence.read(view)
        else:
            pair = scene.render(k, np.asarray(view["T_wc"]), width, height)
        return clipped_pair(*pair)
    integration_seconds = prefix_state._integration_seconds if extending else 0.0
    if prefix_state is not None:
        prefix_state._unfinished = True
    # Only integration views reach the TSDF API. Held-out images are generated later.
    for view in cfg["views"]:
        if view["split"] != "integration" or view["id"] in prior_ids:
            continue
        pose = np.asarray(view["T_wc"])
        rgb, depth = read_observation(view)
        if not np.any(depth > 0):
            raise ValueError(f"integration view {view['id']} observes no valid geometry")
        if not compact:
            references[view["id"]] = read_observation(view, reference=True) if sequence is not None else (rgb, depth)
        input_depth = corrupt_depth(depth, cfg["sensor"], view["id"], cam["depth_trunc_m"])
        input_rgb = rgb.copy()
        input_rgb[input_depth == 0] = 0
        if not np.any(input_depth > 0):
            raise ValueError(f"integration view {view['id']} has no valid observations after sensor perturbation")
        if not compact:
            observations[view["id"]] = (input_rgb, input_depth)
        tick = perf_counter()
        backend.integrate(make_rgbd(input_rgb, input_depth, cam["depth_trunc_m"]), intrinsic, pose)
        integration_seconds += perf_counter() - tick
        if occupancy is not None:
            tick = perf_counter()
            evidence = occupancy.integrate(input_depth, k, pose, cam["depth_trunc_m"],
                                            pixel_stride=occupancy_cfg["pixel_stride"])
            occupancy_seconds += perf_counter() - tick
            occupancy_frames.append({"id": view["id"], **evidence})
        integrated_ids.append(view["id"])

    mesh = backend.extract_mesh()
    if not len(mesh.triangles):
        raise RuntimeError("TSDF produced an empty mesh. Check bounds/poses; scalable returned empty in this local build. Try uniform.")
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh):
        raise IOError("failed to save reconstruction mesh")
    if not compact:
        o3d.io.write_point_cloud(str(out / "pointcloud.ply"), backend.extract_point_cloud())
    volume_path = None
    if settings.backend == "uniform" and (not compact or snapshot_volume):
        backend.save_volume(out / "tsdf_volume.npz")
        volume_path = "tsdf_volume.npz"

    occupancy_report = None
    if occupancy is not None:
        occupancy.save(out / "occupancy.npz")
        if not compact:
            save_occupancy_slices(out / "occupancy_slices.png", occupancy)
        labels = {-1: "outside", 0: "unknown", 1: "free", 2: "occupied"}
        probes = []
        for probe in occupancy_cfg["probes"]:
            code = int(occupancy.state_at(np.asarray([probe["point_m"]]))[0])
            probes.append({**probe, "state": labels[code], "state_code": code})
        occupancy_report = {
            **occupancy.summary(), "volume": "occupancy.npz", "slices": None if compact else "occupancy_slices.png",
            "origin_m": list(settings.origin_m), "voxel_size_m": occupancy.voxel_size_m,
            "resolution": occupancy_cfg["resolution"], "pixel_stride": occupancy_cfg["pixel_stride"],
            "integration_seconds": occupancy_seconds, "integrated_frame_ids": integrated_ids.copy(),
            "frame_evidence": occupancy_frames, "probes": probes,
            "semantics": "Static ray evidence: unknown=no observed ray evidence; free=traversed by a valid measured ray before its hit cell; occupied=contains a measured endpoint. Occupied wins conflicts. These labels do not certify the entire voxel as collision-free or fill solid interiors. Invalid/missing depth adds no evidence.",
        }
        write_json(out / "occupancy.json", occupancy_report)

    rows = []
    for view in cfg["views"]:
        name, split = view["id"], view["split"]
        if compact and split == "integration":
            continue
        pose = np.asarray(view["T_wc"])
        if split == "held_out":
            rgb, depth = read_observation(view, reference=True)
        else:
            if name not in references:
                references[name] = read_observation(view, reference=True)
                source_rgb, source_depth = read_observation(view)
                input_depth = corrupt_depth(source_depth, cfg["sensor"], name, cam["depth_trunc_m"])
                source_rgb[input_depth == 0] = 0
                observations[name] = (source_rgb, input_depth)
            rgb, depth = references[name]
        if not (depth > 0).any():
            raise ValueError(f"evaluation view {name} observes no valid geometry")
        pred_rgb, pred_depth = render_mesh_rgbd(mesh, k, pose, width, height)
        clipped = (pred_depth <= 0) | (pred_depth >= cam["depth_trunc_m"]) | ~np.isfinite(pred_depth)
        pred_depth[clipped], pred_rgb[clipped] = 0, 0
        # Keep actual fusion inputs separate from clean ground truth. The held-out
        # input remains a reference-only image and never reaches integrate().
        input_rgb, input_depth = observations[name] if split == "integration" else (rgb, depth)
        save_rgbd(out / "inputs" / name, input_rgb, input_depth)
        save_rgbd(out / "references" / name, rgb, depth)
        save_rgbd(out / "renders" / name, pred_rgb, pred_depth)
        tolerance = cfg["evaluation"]["depth_tolerance_m"]
        metrics = evaluate_rgbd(rgb, depth, pred_rgb, pred_depth, tolerance)
        metrics.update(id=name, split=split)
        rows.append(metrics)
        frame_records.append({"id": name, "split": split, "T_wc": view["T_wc"],
                              "rgb": f"inputs/{name}/rgb.png", "depth_m": f"inputs/{name}/depth_m.npy",
                              "input_role": "fusion_observation" if split == "integration" else "evaluation_reference_only",
                              "reference_rgb": f"references/{name}/rgb.png",
                              "reference_depth_m": f"references/{name}/depth_m.npy",
                              **({"timestamp_s": view["timestamp_s"]} if "timestamp_s" in view else {})})
        valid, pred_valid = depth > 0, pred_depth > 0
        error = np.full(depth.shape, np.nan, dtype=np.float32)
        both = valid & pred_valid
        error[both] = np.abs(pred_depth[both] - depth[both])
        np.save(out / "renders" / name / "depth_abs_error_m.npy", error)
        for mask_name, mask in (("missing", valid & ~pred_valid), ("extra", pred_valid & ~valid)):
            o3d.io.write_image(str(out / "renders" / name / f"{mask_name}.png"), o3d.geometry.Image(mask.astype(np.uint8) * 255))
        save_comparison(out / "renders" / name / "comparison.png", rgb, depth, pred_rgb, pred_depth,
                        f"{cfg['scene'].get('name', 'scene')} | {name} | {split}", tolerance)
    report = {
        "schema_version": 2, "scene": cfg["scene"].get("name", "scene"),
        "sensor": cfg["sensor"], "evaluation_reference": reference_kind,
        "data_source": {**cfg["source"], "metadata": source_metadata},
        "sensor_model": "Optional independent Gaussian camera-z noise and Bernoulli dropout on integration observations only; seed plus SHA256 frame ID, separate PCG64 streams. Not a calibrated camera model.",
        "integrated_frame_ids": integrated_ids,
        "held_out_frame_ids": [v["id"] for v in cfg["views"] if v["split"] == "held_out"],
        "backend": settings.backend, "effective_voxel_size_m": settings.effective_voxel_size_m,
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
        "volume": volume_path, "occupancy": occupancy_report, "views": rows,
        "summary": {split: summarize_views([r for r in rows if r["split"] == split]) for split in ("integration", "held_out")},
        "metric_scope": "Depth and RGB errors use depth-overlap pixels only; coverage, missing/extra and inlier fraction report missing or incorrect geometry. RGB PSNR null with exact_match=true means infinite PSNR.",
        "export_scope": "held_out_snapshot" if compact else "complete",
        "incremental_prefix": prefix_state is not None,
        "newly_integrated_frame_ids": [name for name in integrated_ids if name not in prior_ids],
        "timing_scope": "total_seconds measures this call's loading, new fusion, evaluation and exports; excludes final manifest/metrics writes and viewer export. Integration timings are cumulative across the prefix when incremental_prefix is true.",
        "integration_seconds": integration_seconds, "total_seconds": perf_counter() - started,
        "versions": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "open3d": o3d.__version__},
    }
    write_json(out / "manifest.json", {"depth_unit": "metre", "depth_definition": "camera_z", "pixel_centers": "integer u,v", "camera_convention": "opencv x-right y-down z-forward", "K": cam["K"], "width": width, "height": height, "world_up": cfg["world_up"], "frames": frame_records})
    write_json(out / "metrics.json", report)
    if prefix_state is not None:
        prefix_state._settings, prefix_state._frames = identity, prefix_frames
        prefix_state._backend, prefix_state._occupancy = backend, occupancy
        prefix_state._integration_seconds = integration_seconds
        prefix_state._occupancy_seconds = occupancy_seconds
        prefix_state._occupancy_frames = copy.deepcopy(occupancy_frames)
        prefix_state._unfinished = False
    return report
