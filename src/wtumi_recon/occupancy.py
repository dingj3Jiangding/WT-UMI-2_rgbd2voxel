"""Surface-point helper and bounded ray-based occupancy evidence for static scenes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import voxel_indices


@dataclass
class SparseOccupancyMap:
    """Sparse occupied-voxel set using the project's world-frame convention.

    This compatibility helper records surface voxels only. Use
    EvidenceOccupancyGrid for bounded free/occupied/unknown evidence.
    """

    origin: np.ndarray
    voxel_size: float
    _occupied: set[tuple[int, int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64)
        if self.origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")

    def integrate_points(self, points_world: np.ndarray) -> None:
        flat = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1)
        if not np.any(valid):
            return
        indices = voxel_indices(flat[valid], self.origin, self.voxel_size)
        for index in indices:
            self._occupied.add(tuple(int(value) for value in index))

    @property
    def occupied_indices(self) -> np.ndarray:
        if not self._occupied:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(sorted(self._occupied), dtype=np.int64)


UNKNOWN, FREE, OCCUPIED = 0, 1, 2
OCCUPANCY_FORMAT = "wtumi-occupancy-evidence-v1"
OCCUPANCY_SEMANTICS = (
    "Static, sampled-ray evidence: unknown=0, free=1, occupied=2. "
    "A valid ray marks intersected cells before its endpoint free and its endpoint cell occupied. "
    "No updates behind a return or from invalid/missing depth. Occupied evidence wins conflicts. "
    "Free means traversed by a sampled ray, not that every part of the cell is empty; "
    "occupied cells are observed surface returns, not filled object interiors. "
    "This is not a calibrated occupancy probability or collision-safety guarantee."
)


def _validate_grid(origin_m, voxel_size_m, resolution):
    origin = np.asarray(origin_m, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("occupancy origin_m must contain three finite values")
    if isinstance(voxel_size_m, (bool, np.bool_)) or not np.isscalar(voxel_size_m):
        raise ValueError("occupancy voxel_size_m must be finite and positive")
    try:
        size = float(voxel_size_m)
    except (ValueError, TypeError) as exc:
        raise ValueError("occupancy voxel_size_m must be finite and positive") from exc
    if not np.isfinite(size) or size <= 0:
        raise ValueError("occupancy voxel_size_m must be finite and positive")
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(resolution, (int, np.integer)) or resolution < 1:
        raise ValueError("occupancy resolution must be a positive integer")
    return origin, size, int(resolution)


def _summary(state, free_count, hit_count, frames, valid_rays):
    counts = np.bincount(state.reshape(-1), minlength=3)
    return {
        "unknown_cells": int(counts[UNKNOWN]), "free_cells": int(counts[FREE]),
        "occupied_cells": int(counts[OCCUPIED]),
        "conflict_cells": int(np.count_nonzero((free_count > 0) & (hit_count > 0))),
        "total_cells": int(state.size), "frames": int(frames), "valid_rays": int(valid_rays),
        "evidence_policy": "occupied_evidence_wins", "semantics": OCCUPANCY_SEMANTICS,
    }


class EvidenceOccupancyGrid:
    """Finite static-scene evidence grid, with per-frame cell counts.

    Geometry uses the half-open volume [origin, origin+N*size). Coordinates and
    array axes are xyz (C order). A ray ending exactly on an internal plane
    belongs to floor(endpoint), independently of its direction. Traversal
    counts only positive-length cell intersections; exact corner/edge ties
    advance together. Hit evidence dominates free evidence and conflicts stay
    visible in the counts. No temporal clearing or probabilistic inference is
    performed.
    """

    def __init__(self, origin_m, voxel_size_m, resolution):
        self.origin_m, self.voxel_size_m, self.resolution = _validate_grid(origin_m, voxel_size_m, resolution)
        shape = (self.resolution,) * 3
        self.state = np.zeros(shape, dtype=np.uint8)
        self.free_count = np.zeros(shape, dtype=np.uint32)
        self.hit_count = np.zeros(shape, dtype=np.uint32)
        self.frames = 0
        self.valid_rays = 0

    def state_at(self, points_world):
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points_world must have shape (...,3)")
        flat = points.reshape(-1, 3)
        grid = (flat - self.origin_m) / self.voxel_size_m
        inside = np.isfinite(grid).all(axis=1) & (grid >= 0).all(axis=1) & (grid < self.resolution).all(axis=1)
        result = np.full(len(flat), -1, dtype=np.int8)
        indices = np.floor(grid[inside]).astype(np.int64)
        result[inside] = self.state[tuple(indices.T)]
        return result.reshape(points.shape[:-1])

    def integrate(self, depth_m, K, T_wc, depth_trunc_m, pixel_stride=1):
        """Integrate valid optical-axis depth rays; invalid samples do nothing.

        One cell can get at most one hit count and one free count per frame,
        even when many rays visit it. Counts are saturated uint32 values.
        """
        depth = np.asarray(depth_m)
        k, pose = np.asarray(K, dtype=float), np.asarray(T_wc, dtype=float)
        if depth.ndim != 2 or not depth.size:
            raise ValueError("occupancy depth_m must be a nonempty 2D image")
        if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("occupancy K must be a finite pinhole 3x3 matrix")
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("occupancy T_wc must be a finite homogeneous 4x4 matrix")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("occupancy T_wc rotation must be orthonormal with determinant +1")
        if isinstance(pixel_stride, (bool, np.bool_)) or not isinstance(pixel_stride, (int, np.integer)) or pixel_stride < 1:
            raise ValueError("occupancy pixel_stride must be a positive integer")
        if not np.isscalar(depth_trunc_m) or not np.isfinite(depth_trunc_m) or depth_trunc_m <= 0:
            raise ValueError("occupancy depth_trunc_m must be finite and positive")
        try:
            inverse_k = np.linalg.inv(k)
        except np.linalg.LinAlgError as exc:
            raise ValueError("occupancy K must be invertible") from exc
        vv, uu = np.mgrid[0:depth.shape[0]:pixel_stride, 0:depth.shape[1]:pixel_stride]
        sampled = depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(sampled) & (sampled > 0) & (sampled < depth_trunc_m)
        pixels = np.column_stack((uu[valid], vv[valid], np.ones(int(valid.sum()))))
        camera_rays = pixels @ inverse_k.T
        # K^-1 [u,v,1] has z=1, so depth remains optical-axis distance.
        endpoints = (camera_rays * sampled[valid, None]) @ rotation.T + pose[:3, 3]
        if not np.isfinite(endpoints).all():
            raise ValueError("occupancy ray endpoints are not finite")
        start = (pose[:3, 3] - self.origin_m) / self.voxel_size_m
        end = (endpoints - self.origin_m) / self.voxel_size_m
        free, hits = self._trace_segments(start, end)
        cap = np.iinfo(np.uint32).max
        free_flat, hit_flat = self.free_count.reshape(-1), self.hit_count.reshape(-1)
        free_flat[free & (free_flat < cap)] += 1
        hit_flat[hits & (hit_flat < cap)] += 1
        self.state[self.free_count > 0] = FREE
        self.state[self.hit_count > 0] = OCCUPIED
        self.frames += 1
        self.valid_rays += len(end)
        return {
            "sampled_pixels": int(sampled.size), "valid_rays": int(len(end)),
            "free_evidence_cells": int(free.sum()), "hit_evidence_cells": int(hits.sum()),
            "frame_conflict_cells": int((free & hits).sum()),
        }

    def _trace_segments(self, start, end):
        """Vectorized slab clipping and Amanatides-Woo traversal in grid units."""
        n = self.resolution
        free, hits = np.zeros(n**3, dtype=bool), np.zeros(n**3, dtype=bool)
        if not len(end):
            return free, hits
        endpoint_inside = (end >= 0).all(axis=1) & (end < n).all(axis=1)
        endpoint_flat = np.full(len(end), -1, dtype=np.int64)
        endpoint_indices = np.floor(end[endpoint_inside]).astype(np.int64)
        endpoint_flat[endpoint_inside] = np.ravel_multi_index(endpoint_indices.T, (n, n, n))
        hits[endpoint_flat[endpoint_inside]] = True
        direction = end - start
        parallel = direction == 0
        # A ray on the volume's excluded maximum face never enters it.
        outside_parallel = np.any(parallel & ((start < 0) | (start >= n)), axis=1)
        low = np.full_like(direction, -np.inf)
        high = np.full_like(direction, np.inf)
        np.divide(-start, direction, out=low, where=~parallel)
        np.divide(n-start, direction, out=high, where=~parallel)
        near, far = np.minimum(low, high), np.maximum(low, high)
        entry = np.maximum(0, near.max(axis=1))
        exit_time = np.minimum(1, far.min(axis=1))
        active = (exit_time > entry) & ~outside_parallel
        if not active.any():
            return free, hits
        direction, entry, exit_time, endpoint_flat = direction[active], entry[active], exit_time[active], endpoint_flat[active]
        position = np.clip(start + direction * entry[:, None], 0, n)
        # Select the cell just after entry, including entry from a maximum face.
        toward = np.where(direction > 0, np.inf, np.where(direction < 0, -np.inf, position))
        indices = np.floor(np.nextafter(position, toward)).astype(np.int64)
        signs = np.sign(direction).astype(np.int64)
        boundary = indices + (direction > 0)
        next_time = np.full_like(direction, np.inf)
        delta_time = np.full_like(direction, np.inf)
        np.divide(boundary - start, direction, out=next_time, where=direction != 0)
        np.divide(1.0, np.abs(direction), out=delta_time, where=direction != 0)
        current_time = entry
        for _ in range(3*n + 3):
            inside = (indices >= 0).all(axis=1) & (indices < n).all(axis=1)
            if not inside.all():
                indices, signs, next_time, delta_time, current_time, exit_time, endpoint_flat = [
                    x[inside] for x in (indices, signs, next_time, delta_time, current_time, exit_time, endpoint_flat)
                ]
            if not len(indices):
                break
            crossing = next_time.min(axis=1)
            cell_exit = np.minimum(crossing, exit_time)
            flat = np.ravel_multi_index(indices.T, (n, n, n))
            seen = (cell_exit > current_time) & (flat != endpoint_flat)
            free[flat[seen]] = True
            continuing = crossing < exit_time
            if not continuing.any():
                break
            indices, signs, next_time, delta_time, crossing, exit_time, endpoint_flat = [
                x[continuing] for x in (indices, signs, next_time, delta_time, crossing, exit_time, endpoint_flat)
            ]
            # Merge numerical representations of the same edge/corner crossing.
            tolerance = 16*np.finfo(float).eps*np.maximum(1.0, np.abs(crossing))
            tied = np.abs(next_time - crossing[:, None]) <= tolerance[:, None]
            indices += signs * tied
            next_time[tied] += delta_time[tied]
            current_time = crossing
        else:
            raise RuntimeError("occupancy ray traversal exceeded grid crossing bound")
        return free, hits

    def summary(self):
        result = _summary(self.state, self.free_count, self.hit_count, self.frames, self.valid_rays)
        result.update(origin_m=self.origin_m.tolist(), voxel_size_m=self.voxel_size_m, resolution=self.resolution)
        return result

    def save(self, path):
        np.savez_compressed(
            path, format=OCCUPANCY_FORMAT, state=self.state,
            free_count=self.free_count, hit_count=self.hit_count,
            origin_m=self.origin_m, voxel_size_m=self.voxel_size_m, resolution=self.resolution,
            frames=self.frames, valid_rays=self.valid_rays,
            axis_order="xyz", flat_order="C", unknown_code=UNKNOWN, free_code=FREE, occupied_code=OCCUPIED,
            evidence_policy="occupied_evidence_wins", semantics=OCCUPANCY_SEMANTICS,
        )


def load_occupancy(path):
    """Load a bounded evidence snapshot, validating conventions and state counts."""
    with np.load(path, allow_pickle=False) as data:
        expected = {"format": OCCUPANCY_FORMAT, "axis_order": "xyz", "flat_order": "C",
                    "unknown_code": UNKNOWN, "free_code": FREE, "occupied_code": OCCUPIED,
                    "evidence_policy": "occupied_evidence_wins"}
        for key, value in expected.items():
            if key not in data or data[key].shape != () or data[key].item() != value:
                raise ValueError(f"unsupported occupancy {key}")
        origin, size, n = _validate_grid(data["origin_m"], data["voxel_size_m"].item(), data["resolution"].item())
        arrays = {key: data[key].copy() for key in ("state", "free_count", "hit_count")}
        for key, array in arrays.items():
            if array.shape != (n, n, n) or not np.issubdtype(array.dtype, np.integer) or (array < 0).any():
                raise ValueError(f"invalid occupancy {key} array")
        state, free_count, hit_count = arrays["state"], arrays["free_count"], arrays["hit_count"]
        expected_state = np.where(hit_count > 0, OCCUPIED, np.where(free_count > 0, FREE, UNKNOWN))
        if not np.array_equal(state, expected_state):
            raise ValueError("occupancy state disagrees with evidence counts")
        frames, valid_rays = data["frames"].item(), data["valid_rays"].item()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (frames, valid_rays)):
            raise ValueError("invalid occupancy frame/ray counters")
        if (free_count > frames).any() or (hit_count > frames).any():
            raise ValueError("occupancy evidence count exceeds frame count")
    summary = _summary(state, free_count, hit_count, frames, valid_rays)
    summary.update(origin_m=origin.tolist(), voxel_size_m=size, resolution=n)
    return {**arrays, "origin_m": origin, "voxel_size_m": size, "resolution": n,
            "frames": frames, "valid_rays": valid_rays, "summary": summary}
