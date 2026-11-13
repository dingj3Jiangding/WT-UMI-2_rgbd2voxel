"""Independent analytic checks of free/occupied/unknown ray evidence."""

import numpy as np
import pytest

from wtumi_recon.occupancy import EvidenceOccupancyGrid, FREE, OCCUPIED, UNKNOWN, load_occupancy


def _pose(position=(.5, .5, -1), rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = position
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def test_front_ray_marks_free_hit_and_preserves_occluded_space():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 5)
    info = grid.integrate([[3.5]], np.eye(3), _pose(), 10)
    np.testing.assert_array_equal(grid.state[0, 0], [FREE, FREE, OCCUPIED, UNKNOWN, UNKNOWN])
    assert (grid.state[:, 1:] == UNKNOWN).all()
    assert info['valid_rays'] == 1 and info['hit_evidence_cells'] == 1
    assert grid.summary()['free_cells'] == 2
    assert grid.summary()['occupied_cells'] == 1
    assert grid.state_at([.5, .5, 2.5]) == OCCUPIED
    np.testing.assert_array_equal(grid.state_at([[-.1, 0, 0], [5, 0, 0], [np.nan, 0, 0]]), [-1, -1, -1])


def test_rotated_camera_and_off_axis_depth_is_optical_z_not_ray_length():
    # Camera +Z points world +X; u=1 has camera ray (1,0,1), depth 2.5.
    rotation = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    grid = EvidenceOccupancyGrid([-3, 0, -3], 1, 7)
    grid.integrate([[0, 2.5]], np.eye(3), _pose((0, .5, 0), rotation), 10)
    assert grid.state_at([2.5, .5, -2.5]) == OCCUPIED
    assert grid.state_at([1.5, .5, -1.5]) == FREE
    assert grid.state_at([3.5, .5, -3.5]) == -1
    assert grid.summary()['occupied_cells'] == 1


def test_corner_crossings_skip_zero_length_side_cells():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    # Pixel (u=1,v=1), identity K -> diagonal (1,1,1).
    depth = np.zeros((2, 2)); depth[1, 1] = 2.0
    grid.integrate(depth, np.eye(3), _pose((.5, .5, .5)), 10)
    assert grid.state[0, 0, 0] == grid.state[1, 1, 1] == FREE
    assert grid.state[2, 2, 2] == OCCUPIED
    assert np.count_nonzero(grid.state) == 3


def test_invalid_depth_has_no_evidence_including_max_range():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    info = grid.integrate([[0, -1, np.nan, np.inf, 5, 6]], np.eye(3), _pose(), 5)
    assert info['sampled_pixels'] == 6 and info['valid_rays'] == 0
    assert not grid.state.any()
    assert not grid.free_count.any() and not grid.hit_count.any()
    assert grid.summary()['frames'] == 1


def test_valid_endpoint_beyond_volume_carves_only_intersected_segment():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    grid.integrate([[7]], np.eye(3), _pose(), 10)
    np.testing.assert_array_equal(grid.state[0, 0], [FREE]*4)
    assert grid.summary()['occupied_cells'] == 0
    assert grid.summary()['free_cells'] == 4


def test_max_boundary_parallel_ray_does_not_enter_volume():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    grid.integrate([[3]], np.eye(3), _pose((4, .5, -1)), 10)
    assert not grid.state.any()


def test_entering_from_max_face_and_internal_endpoint_plane_ownership():
    back = np.diag([-1, 1, -1])
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    grid.integrate([[2]], np.eye(3), _pose((.5, .5, 4), back), 10)
    np.testing.assert_array_equal(grid.state[0, 0], [UNKNOWN, UNKNOWN, OCCUPIED, FREE])
    # Same endpoint plane, opposite direction, is owned by the same floor cell.
    forward = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    forward.integrate([[3]], np.eye(3), _pose(), 10)
    np.testing.assert_array_equal(forward.state[0, 0], [FREE, FREE, OCCUPIED, UNKNOWN])


def test_endpoint_on_min_face_has_hit_without_intersection_length():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    grid.integrate([[1]], np.eye(3), _pose(), 10)
    np.testing.assert_array_equal(grid.state[0, 0], [OCCUPIED, UNKNOWN, UNKNOWN, UNKNOWN])
    assert grid.summary()['free_cells'] == 0


def test_endpoint_on_max_face_is_outside_and_previous_cells_are_free():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    grid.integrate([[5]], np.eye(3), _pose(), 10)
    np.testing.assert_array_equal(grid.state[0, 0], [FREE]*4)


def test_frame_counts_are_unique_per_cell_and_occupied_wins_order_independently():
    k = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]])
    grids = []
    for depths in ([2.5, 3.5], [3.5, 2.5]):
        grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
        for d in depths:
            grid.integrate([[d, d]], k, _pose(), 10)
        grids.append(grid)
        assert grid.free_count[0, 0, 0] == 2  # Two frames, not four rays.
        assert grid.hit_count[0, 0, 1] == 1
        assert grid.free_count[0, 0, 1] == 1
        assert grid.state[0, 0, 1] == OCCUPIED
        assert grid.summary()['conflict_cells'] == 1
    for name in ('state', 'free_count', 'hit_count'):
        np.testing.assert_array_equal(getattr(grids[0], name), getattr(grids[1], name))


def test_same_frame_conflict_is_retained_and_unknown_is_not_free():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    k = np.diag([1000., 1000., 1.])
    grid.integrate([[2.5, 3.5]], k, _pose(), 10)
    assert grid.hit_count[0, 0, 1] == grid.free_count[0, 0, 1] == 1
    assert grid.state[0, 0, 1] == OCCUPIED
    assert grid.state[0, 0, 3] == UNKNOWN


def test_stride_samples_only_selected_pixels():
    grid = EvidenceOccupancyGrid([0, 0, 0], 1, 4)
    info = grid.integrate(np.full((3, 5), 2.5), np.diag([1000., 1000., 1.]), _pose(), 10, pixel_stride=2)
    assert info['sampled_pixels'] == info['valid_rays'] == 6
    assert grid.hit_count.max() == 1


def test_save_load_preserves_arrays_and_rejects_inconsistent_state(tmp_path):
    grid = EvidenceOccupancyGrid([0, 0, 0], .5, 8)
    grid.integrate([[2.5]], np.eye(3), _pose(), 10)
    path = tmp_path / 'occupancy.npz'
    grid.save(path)
    loaded = load_occupancy(path)
    for name in ('state', 'free_count', 'hit_count'):
        np.testing.assert_array_equal(loaded[name], getattr(grid, name))
    assert loaded['summary'] == grid.summary()
    with np.load(path) as data:
        values = {k: data[k] for k in data.files}
    values['state'] = values['state'].copy()
    values['state'][0, 0, 0] = OCCUPIED
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='disagrees'):
        load_occupancy(path)


@pytest.mark.parametrize('kwargs', [
    {'origin_m':[0, 0, np.nan]}, {'voxel_size_m':0}, {'voxel_size_m':True},
    {'resolution':0}, {'resolution':2.5}, {'resolution':True},
])
def test_bad_grid_metadata_is_rejected(kwargs):
    fields = {'origin_m':[0,0,0], 'voxel_size_m':1, 'resolution':4, **kwargs}
    with pytest.raises(ValueError):
        EvidenceOccupancyGrid(**fields)
