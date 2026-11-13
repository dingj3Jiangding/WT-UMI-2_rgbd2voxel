import numpy as np

from wtumi_recon.geometry import backproject_depth, transform_points, voxel_indices


def test_backproject_center_and_offset_pixel():
    depth = np.ones((2, 3), dtype=np.float32)
    k = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    points = backproject_depth(depth, k)
    np.testing.assert_allclose(points[0, 1], [0.0, -0.25, 1.0])
    np.testing.assert_allclose(points[0, 2], [0.5, -0.25, 1.0])


def test_camera_to_world_transform():
    points = np.array([[0.1, 0.2, 1.0], [np.nan, 0.0, 1.0]])
    t_wc = np.eye(4)
    t_wc[:3, 3] = [1.0, 2.0, 3.0]
    transformed = transform_points(points, t_wc)
    np.testing.assert_allclose(transformed[0], [1.1, 2.2, 4.0])
    assert np.isnan(transformed[1]).all()


def test_voxel_indices():
    points = np.array([[0.13, 0.27, 1.04], [0.0, 0.0, 0.0]])
    indices = voxel_indices(points, np.zeros(3), 0.01)
    np.testing.assert_array_equal(indices, [[13, 27, 104], [0, 0, 0]])
