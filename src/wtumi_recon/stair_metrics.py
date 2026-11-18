"""Bounded, evaluation-only staircase and sampled-ray occupancy diagnostics.

Ground-truth surfaces and held-out masks never enter occupancy integration.
Depth errors are conditional on valid overlap; completeness denominators retain
missing predictions. Occupancy states are coarse ray evidence, not solid fill
or whole-cell collision clearance.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .occupancy import EvidenceOccupancyGrid, FREE, OCCUPIED, UNKNOWN, OCCUPANCY_SEMANTICS, load_occupancy
from .sequence import RGBDSequence


MAX_SURFACE_SAMPLES = 1_000_000
MAX_GRID_CELLS = 256**3
MAX_ROI_LATTICE_CELLS = 256**3
MAX_DEPTH_PIXELS = 4096**2
MAX_SNAPSHOT_BYTES = 128 * 1024**2


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def _resolution(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("resolution must be a positive integer")
    if int(value)**3 > MAX_GRID_CELLS:
        raise ValueError(f"occupancy exceeds {MAX_GRID_CELLS} grid cells")
    return int(value)


def sample_stair_surfaces(stairs, spacing_m=0.04):
    """Sample only exposed treads and front risers, ordered by ascending step.

    Step i's tread is at base+(i+1)*rise; its front riser spans heights
    base+i*rise to base+(i+1)*rise at y=front+i*tread. Interiors, side walls,
    bottoms, and the back wall are excluded. Each rectangle is divided into
    ceil(length/spacing) equal cells per axis, sampled at their centers, so
    spacing is at most spacing_m and edges are never double-counted. Returned
    points are equally weighted probes, not exact surface-area quadrature.
    """
    if not isinstance(stairs, dict):
        raise ValueError("stairs must be a mapping")
    origin = _vector(stairs.get("origin_m"), "stairs.origin_m")
    width, tread, rise = [_positive(stairs.get(k), f"stairs.{k}")
                          for k in ("width_m", "tread_m", "rise_m")]
    steps = stairs.get("steps")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("stairs.steps must be a positive integer")
    spacing = _positive(spacing_m, "spacing_m")
    ratios = np.asarray([width, tread, rise]) / spacing
    if not np.isfinite(ratios).all() or (ratios > MAX_SURFACE_SAMPLES).any():
        raise ValueError("stair surface sample limit exceeded")
    nx, ny, nz = (max(1, int(math.ceil(v))) for v in ratios)
    count = int(steps) * nx * (ny + nz)
    if count > MAX_SURFACE_SAMPLES:
        raise ValueError(f"stair surface sample limit is {MAX_SURFACE_SAMPLES}")
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = np.array([origin + [-width / 2, 0, 0],
                           origin + [width / 2, int(steps) * tread, int(steps) * rise]])
    if not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("stair bounds must be finite and representable")
    x = origin[0] + ((np.arange(nx) + 0.5) / nx - 0.5) * width
    local_y = (np.arange(ny) + 0.5) * (tread / ny)
    local_z = (np.arange(nz) + 0.5) * (rise / nz)
    result = np.empty((count, 3), dtype=np.float64)
    offset = 0
    for i in range(int(steps)):
        n = nx * ny
        result[offset:offset+n] = np.column_stack((np.repeat(x, ny),
            np.tile(origin[1] + i*tread + local_y, nx), np.full(n, origin[2] + (i+1)*rise)))
        offset += n
        n = nx * nz
        result[offset:offset+n] = np.column_stack((np.repeat(x, nz),
            np.full(n, origin[1] + i*tread), np.tile(origin[2] + i*rise + local_z, nx)))
        offset += n
    return result


def _roi_bounds(roi):
    if not isinstance(roi, dict):
        raise ValueError("roi must be a mapping")
    lower = _vector(roi.get("min_m"), "roi.min_m")
    upper = _vector(roi.get("max_m"), "roi.max_m")
    if not (upper > lower).all():
        raise ValueError("roi.max_m must exceed roi.min_m on every axis")
    return lower, upper


def _roi_selection(origin, size, resolution, roi):
    """Count centers in a half-open ROI on the grid's infinite lattice."""
    lower, upper = _roi_bounds(roi)
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = (np.stack([lower, upper]) - origin) / size - 0.5
    if not np.isfinite(bounds).all():
        raise ValueError("ROI lattice coordinates must be finite")
    starts = [math.ceil(v) for v in bounds[0]]
    stops = [math.ceil(v) for v in bounds[1]]
    total = math.prod(b-a for a, b in zip(starts, stops))
    if total > MAX_ROI_LATTICE_CELLS:
        raise ValueError(f"ROI exceeds {MAX_ROI_LATTICE_CELLS} lattice cells")
    starts = [min(resolution, max(0, x)) for x in starts]
    stops = [min(resolution, max(0, x)) for x in stops]
    return tuple(slice(a, b) for a, b in zip(starts, stops)), total


def _sample_states(state, origin, size, samples):
    with np.errstate(over="ignore", invalid="ignore"):
        positions = (samples - origin) / size
    inside = np.isfinite(positions).all(axis=1) & (positions >= 0).all(axis=1) & (positions < state.shape[0]).all(axis=1)
    labels = np.full(len(samples), -1, dtype=np.int8)
    indices = np.floor(positions[inside]).astype(np.int64)
    labels[inside] = state[tuple(indices.T)]
    return labels


def _occupancy_diagnostics(state, origin, size, roi, samples):
    selection, total = _roi_selection(origin, size, state.shape[0], roi)
    selected = state[selection]
    counts = np.bincount(selected.reshape(-1), minlength=3)
    inside = int(selected.size)
    report = {"roi_cell_count": total, "roi_in_volume_cell_count": inside,
              "roi_outside_volume_cell_count": total-inside}
    for name, code in (("unknown", UNKNOWN), ("free", FREE), ("occupied", OCCUPIED)):
        count = int(counts[code])
        report[f"roi_{name}_cell_count"] = count
        report[f"roi_{name}_fraction"] = count/inside if inside else None
    known = int(counts[FREE] + counts[OCCUPIED])
    report.update(roi_known_cell_count=known, roi_known_fraction=known/inside if inside else None)
    labels = _sample_states(state, origin, size, samples)
    n = len(labels)
    outside = int(np.count_nonzero(labels == -1))
    report.update(surface_sample_count=n, surface_samples_in_volume_count=n-outside,
                  surface_samples_outside_volume_count=outside,
                  surface_samples_outside_volume_fraction=outside/n if n else None)
    for name, code in (("unknown", UNKNOWN), ("free", FREE), ("occupied", OCCUPIED)):
        count = int(np.count_nonzero(labels == code))
        report[f"surface_samples_in_{name}_cells_count"] = count
        report[f"surface_samples_in_{name}_cells_fraction"] = count/n if n else None
    known = int(np.count_nonzero(labels > UNKNOWN))
    report.update(surface_samples_known_count=known, surface_samples_known_fraction=known/n if n else None)
    return report


def _masked_depth_metrics(reference, prediction, mask, tolerance):
    if reference.ndim != 2 or prediction.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("mask, reference depth, and rendered depth must have aligned HxW shapes")
    if reference.size > MAX_DEPTH_PIXELS:
        raise ValueError("depth image pixel limit exceeded")
    ref_valid = mask & np.isfinite(reference) & (reference > 0)
    pred_valid = mask & np.isfinite(prediction) & (prediction > 0)
    overlap = ref_valid & pred_valid
    error = prediction[overlap].astype(np.float64) - reference[overlap].astype(np.float64)
    total, n = int(ref_valid.sum()), int(overlap.sum())
    accurate = int((np.abs(error) <= tolerance).sum())
    scale = float(np.max(np.abs(error))) if n else 0
    mean_error = float(np.mean(np.abs(error)/scale))*scale if scale else 0.0
    rms_error = float(np.sqrt(np.mean((error/scale)**2)))*scale if scale else 0.0
    return {"status": "ok" if n else "no_overlap" if total else "no_valid_reference",
            "mask_pixels": int(mask.sum()), "reference_valid_pixels": total,
            "prediction_valid_pixels": int(pred_valid.sum()), "overlap_pixels": n,
            "missing_pixels": int((ref_valid & ~pred_valid).sum()),
            "extra_pixels": int((pred_valid & ~ref_valid).sum()),
            "within_tolerance_pixels": accurate, "coverage": n/total if total else None,
            "mean_absolute_error_m": mean_error if n else None,
            "root_mean_square_error_m": rms_error if n else None,
            "within_tolerance_fraction": accurate/total if total else None,
            "tolerance_m": tolerance}


def _pooled_depth_metrics(rows, tolerance):
    keys = ("mask_pixels", "reference_valid_pixels", "prediction_valid_pixels", "overlap_pixels",
            "missing_pixels", "extra_pixels", "within_tolerance_pixels")
    result = {key: sum(row[key] for row in rows) for key in keys}
    n, total = result["overlap_pixels"], result["reference_valid_pixels"]
    rms_scale = max((r["root_mean_square_error_m"] for r in rows if r["overlap_pixels"]), default=0)
    pooled_rms = rms_scale * math.sqrt(sum((r["root_mean_square_error_m"]/rms_scale)**2 * (r["overlap_pixels"]/n)
                    for r in rows if r["overlap_pixels"])) if rms_scale else 0.0
    result.update(view_count=len(rows), tolerance_m=tolerance, coverage=n/total if total else None,
                  within_tolerance_fraction=result["within_tolerance_pixels"]/total if total else None,
                  mean_absolute_error_m=sum(r["mean_absolute_error_m"]*(r["overlap_pixels"]/n) for r in rows
                                            if r["overlap_pixels"]) if n else None,
                  root_mean_square_error_m=pooled_rms if n else None)
    return result


def _surface_completeness(mesh_path, samples, tolerance, spacing):
    import open3d as o3d

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not np.isfinite(vertices).all():
        raise ValueError("reconstructed mesh contains nonfinite vertices")
    n = len(samples)
    distances = np.empty(0, dtype=np.float32)
    if len(vertices) and len(triangles):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        chunks = []
        for offset in range(0, n, 100_000):
            query = samples[offset:offset+100_000].astype(np.float32)
            if not np.isfinite(query).all():
                raise ValueError("stair sample coordinates exceed mesh query precision")
            chunks.append(scene.compute_distance(o3d.core.Tensor(query)).numpy())
        distances = np.concatenate(chunks)
        if not np.isfinite(distances).all():
            raise ValueError("mesh distance query returned nonfinite distances")
    covered = int(np.count_nonzero(distances <= tolerance))
    return {"status": "ok" if len(distances) else "empty_mesh", "sample_count": n,
            "covered_sample_count": covered,
            "coverage_within_tolerance_fraction": covered/n if n else None,
            "mean_nearest_mesh_distance_m": float(distances.mean()) if len(distances) else None,
            "median_nearest_mesh_distance_m": float(np.median(distances)) if len(distances) else None,
            "distance_sample_count": int(len(distances)), "tolerance_m": tolerance,
            "sample_spacing_max_m": float(spacing), "mesh_triangle_count": len(triangles),
            "scope": "One-way exposed-stair-sample to closest reconstructed triangle distance. "
                     "Completeness-aware: missing regions remain in the coverage denominator. "
                     "Does not measure false-positive mesh geometry or exact surface-area coverage."}


def evaluate_stair_result(result_dir, stairs, roi, held_out_masks, tolerance_m=0.03, sample_spacing_m=0.04):
    """Evaluate common held-out stair masks, exposed surfaces, and fixed ROI.

    Any nonzero single-channel PNG pixel belongs to the mask. Error statistics
    use valid overlap only; within_tolerance_fraction divides by all valid
    masked reference pixels, including missing predictions. Empty denominators
    yield None. ROI includes cell centers in [min_m,max_m); outside-volume
    lattice centers are reported separately. Surface state fractions divide by
    all surface samples, including separately reported out-of-volume samples.
    """
    tolerance = _positive(tolerance_m, "tolerance_m")
    samples = sample_stair_surfaces(stairs, sample_spacing_m)
    _roi_bounds(roi)
    root = Path(result_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    held_out_ids = {f["id"] for f in manifest["frames"] if f["split"] == "held_out"}
    if not isinstance(held_out_masks, dict):
        raise ValueError("held_out_masks must map frame IDs to absolute PNG paths")
    rows = []
    for name, mask_value in sorted(held_out_masks.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) or name not in held_out_ids:
            raise ValueError(f"mask ID is not a held-out result frame: {name!r}")
        mask_path = Path(mask_value).expanduser()
        if not mask_path.is_absolute():
            raise ValueError("held-out mask paths must be absolute")
        with Image.open(mask_path) as image:
            if image.format != "PNG" or image.mode not in ("1", "L", "I", "I;16"):
                raise ValueError("held-out masks must be single-channel PNGs")
            if image.width * image.height > MAX_DEPTH_PIXELS:
                raise ValueError("mask image pixel limit exceeded")
            mask = np.asarray(image) != 0
        reference = np.load(root / "references" / name / "depth_m.npy", allow_pickle=False, mmap_mode="r")
        prediction = np.load(root / "renders" / name / "depth_m.npy", allow_pickle=False, mmap_mode="r")
        row = _masked_depth_metrics(reference, prediction, mask, tolerance)
        rows.append({"id": name, **row})
    # Inspect the scalar before loading the dense evidence arrays.
    with np.load(root / "occupancy.npz", allow_pickle=False) as data:
        _resolution(data["resolution"].item())
    occupancy = load_occupancy(root / "occupancy.npz")
    diagnostics = _occupancy_diagnostics(occupancy["state"], occupancy["origin_m"],
                                         occupancy["voxel_size_m"], roi, samples)
    diagnostics.update(roi={"min_m": list(map(float, roi["min_m"])), "max_m": list(map(float, roi["max_m"]))},
                       roi_fraction_denominator="in-volume ROI cell centers",
                       surface_sample_fraction_denominator="all exposed-surface samples",
                       voxel_size_m=occupancy["voxel_size_m"], semantics=OCCUPANCY_SEMANTICS)
    return {"schema_version": 1, "distance_unit": "metre", "ground_truth_used_for_fusion": False,
            "ground_truth_scope": "Evaluation only. This evaluator does not fuse, mutate occupancy, or modify reconstruction files.",
            "held_out_depth": {"views": rows, "pooled": _pooled_depth_metrics(rows, tolerance),
                "scope": "Common single-channel stair masks only. Depth errors use overlap; coverage and "
                         "within-tolerance fraction use all valid masked reference pixels, including missing predictions."},
            "surface_completeness": _surface_completeness(root / "mesh.ply", samples, tolerance, sample_spacing_m),
            "occupancy_roi": diagnostics}


def occupancy_progress(sequence_manifest, occupancy_settings, roi, stairs, sample_spacing_m=0.04):
    """Accumulate integration observations once; never read held-out/reference images.

    Returns one row and uint8 Z-by-Y snapshot per integration frame. Snapshots
    cover the entire fixed grid at the cell containing the ROI center X; use
    origin='lower' when plotting. If this plane lies outside the volume, raise
    instead of silently substituting another plane. Each row includes axes,
    world extents, cell index, and actual slice-center coordinate. Surface
    samples are queried after integration and never supplied to integrate().
    """
    if not isinstance(occupancy_settings, dict):
        raise ValueError("occupancy_settings must be a mapping")
    origin = _vector(occupancy_settings.get("origin_m"), "occupancy_settings.origin_m")
    length = _positive(occupancy_settings.get("volume_length_m"), "volume_length_m")
    n = _resolution(occupancy_settings.get("resolution"))
    stride = occupancy_settings.get("pixel_stride", 1)
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    lower, upper = _roi_bounds(roi)
    samples = sample_stair_surfaces(stairs, sample_spacing_m)
    size = length/n
    if not np.isfinite(origin + length).all() or not ((origin + length) > origin).all() or size <= 0:
        raise ValueError("occupancy volume bounds must be finite and representable")
    _roi_selection(origin, size, n, roi)
    slice_index = math.floor(((lower[0]/2 + upper[0]/2) - origin[0])/size)
    if not 0 <= slice_index < n:
        raise ValueError("ROI center-X slice is outside the occupancy volume")
    sequence = RGBDSequence(sequence_manifest)
    views = [view for view in sequence.views if view["split"] == "integration"]
    if len(views)*n*n > MAX_SNAPSHOT_BYTES:
        raise ValueError("occupancy progress snapshot memory limit exceeded")
    if sequence.camera["width"]*sequence.camera["height"] > MAX_DEPTH_PIXELS:
        raise ValueError("sequence depth image pixel limit exceeded")
    grid = EvidenceOccupancyGrid(origin, size, n)
    rows, snapshots = [], []
    for view in views:
        _, depth = sequence.read(view)
        grid.integrate(depth, sequence.camera["K"], view["T_wc"], sequence.camera["depth_trunc_m"], int(stride))
        diagnostics = _occupancy_diagnostics(grid.state, origin, size, roi, samples)
        rows.append({"id": view["id"], "timestamp_s": float(view["timestamp_s"]), "frames": grid.frames,
                     **diagnostics, "slice_axis": "x", "slice_index": slice_index,
                     "slice_coordinate_m": float(origin[0] + (slice_index+0.5)*size),
                     "slice_horizontal_axis": "y", "slice_vertical_axis": "z",
                     "slice_horizontal_extent_m": [float(origin[1]), float(origin[1]+length)],
                     "slice_vertical_extent_m": [float(origin[2]), float(origin[2]+length)],
                     "roi_fraction_denominator": "in-volume ROI cell centers",
                     "surface_sample_fraction_denominator": "all exposed-surface samples",
                     "ground_truth_used_for_fusion": False})
        snapshots.append(grid.state[slice_index, :, :].T.copy())
    return rows, snapshots
