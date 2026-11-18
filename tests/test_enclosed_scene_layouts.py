"""Check enclosure, access clearance and camera placement in room templates."""

from itertools import product

import numpy as np
import pytest

from wtumi_recon.rig_capture import head_camera_pose, rig_transform
from wtumi_recon.scene_bundle import build_scene, load_scene_config
from wtumi_recon.scene_presets import resample_scene_config


@pytest.fixture(scope="module")
def g1_geometry(tmp_path_factory):
    """Measure the actual compiled link meshes in each rig's local frame."""
    mujoco = pytest.importorskip("mujoco")
    result = {}
    root = tmp_path_factory.mktemp("room_g1_geometry")
    for template in ("room", "doorway"):
        built = build_scene(template, root / template)
        model = mujoco.MjModel.from_xml_path(str(root / template / "model.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sensor_rig")
        points = []
        geoms = np.flatnonzero(model.geom_group == 1)
        assert len(geoms) == 35
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso") == -1
        for geom in geoms:
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = (model.mesh_vert[first:first + count] @ data.geom_xmat[geom].reshape(3, 3).T
                     + data.geom_xpos[geom])
            points.append((world - data.xpos[rig]) @ data.xmat[rig].reshape(3, 3))
        points = np.concatenate(points)
        result[template] = {"radius": np.linalg.norm(points[:, :2], axis=1).max(),
                            "lower": points.min(axis=0), "upper": points.max(axis=0)}
    return result


def _boxes(config):
    return {obj['name']: obj for obj in config['environment']['objects']}


def _bounds(obj):
    center, half_size = np.asarray(obj['position_m']), np.asarray(obj['size_m']) / 2
    return center - half_size, center + half_size


def _interior(config):
    boxes = _boxes(config)
    return (np.array([_bounds(boxes['left_wall'])[1][0], _bounds(boxes['front_wall'])[1][1], 0.]),
            np.array([_bounds(boxes['right_wall'])[0][0], _bounds(boxes['back_wall'])[0][1], 2.4]))


def _carried_footprint_radius(config, geometry):
    """Bound every robot and payload point at every intermediate yaw."""
    radius = geometry['radius']
    if config['box']['enabled']:
        half_size = np.asarray(config['box']['size_m'])[:2] / 2
        corners = (np.asarray(list(product((-1., 1.), repeat=2))) * half_size
                   + np.asarray(config['box']['position_in_rig_m'])[:2])
        radius = max(radius, np.linalg.norm(corners, axis=1).max())
    return radius


def _segment_intersects_box(start, end, lower, upper):
    """Slab intersection for a complete path segment, including its endpoints."""
    enter, leave = 0., 1.
    for origin, delta, low, high in zip(start, end - start, lower, upper):
        if abs(delta) < 1e-12:
            if origin < low or origin > high:
                return False
            continue
        first, last = sorted(((low - origin) / delta, (high - origin) / delta))
        enter, leave = max(enter, first), min(leave, last)
        if enter > leave:
            return False
    return True


def _door_corners(door, angle):
    radians = np.deg2rad(door['closed_yaw_degrees'] + angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    points = np.array(list(product((0., door['width_m']),
                                  (-door['thickness_m'] / 2, door['thickness_m'] / 2))))
    return points @ rotation.T + np.asarray(door['hinge_position_m'])[:2]


def _rectangles_overlap(first, second):
    """Separating-axis test; exact edge contact is allowed."""
    directions = [np.array([1., 0.]), np.array([0., 1.])]
    for edge in (first[1] - first[0], first[2] - first[0]):
        directions.append(np.array([-edge[1], edge[0]]))
    for axis in directions:
        if np.linalg.norm(axis) < 1e-12:
            continue
        axis = axis / np.linalg.norm(axis)
        left, right = first @ axis, second @ axis
        if min(left.max(), right.max()) - max(left.min(), right.min()) <= 1e-9:
            return False
    return True


@pytest.mark.parametrize('template', ['room', 'doorway'])
def test_rectangular_templates_have_four_joined_full_height_walls(template):
    config = load_scene_config(template)
    boxes = _boxes(config)
    low, high = _interior(config)
    np.testing.assert_allclose(high[:2] - low[:2], [6.4, 10.])
    floor_low, floor_high = _bounds(boxes['floor'])
    assert floor_high[2] == pytest.approx(0.)
    for name in ('front_wall', 'back_wall', 'left_wall', 'right_wall'):
        wall_low, wall_high = _bounds(boxes[name])
        np.testing.assert_allclose([wall_low[2], wall_high[2]], [0., 2.4])
        assert np.all(wall_low[:2] >= floor_low[:2] - 1e-12)
        assert np.all(wall_high[:2] <= floor_high[:2] + 1e-12)
    for side in ('left_wall', 'right_wall'):
        side_low, side_high = _bounds(boxes[side])
        np.testing.assert_allclose([side_low[1], side_high[1]], [low[1], high[1]])
        for end in ('front_wall', 'back_wall'):
            end_low, end_high = _bounds(boxes[end])
            assert end_low[0] <= side_low[0] + 1e-12
            assert end_high[0] >= side_high[0] - 1e-12


def test_room_partition_is_centered_half_length_with_two_bypass_gaps(g1_geometry):
    config = load_scene_config('room')
    low, high = _interior(config)
    wall = _boxes(config)['center_partition']
    lower, upper = _bounds(wall)
    np.testing.assert_allclose(wall['position_m'][:2], (low[:2] + high[:2]) / 2)
    assert wall['size_m'][1] == pytest.approx((high[1] - low[1]) / 2)
    np.testing.assert_allclose([lower[2], upper[2]], [0., high[2]])
    footprint_radius = _carried_footprint_radius(config, g1_geometry['room'])
    assert lower[1] - low[1] > 2 * footprint_radius
    assert high[1] - upper[1] > 2 * footprint_radius
    positions = np.array([frame['position_m'][:2] for frame in config['trajectory']])
    assert positions[:, 0].min() < lower[0] and positions[:, 0].max() > upper[0]
    for start, end in zip(positions[:-1], positions[1:]):
        assert not _segment_intersects_box(start, end, lower[:2] - footprint_radius,
                                          upper[:2] + footprint_radius)


def test_doorway_partition_reaches_side_walls_and_door_fits_opening():
    config = load_scene_config('doorway')
    boxes, (low, high) = _boxes(config), _interior(config)
    left_low, left_high = _bounds(boxes['partition_left'])
    right_low, right_high = _bounds(boxes['partition_right'])
    header_low, header_high = _bounds(boxes['partition_header'])
    np.testing.assert_allclose([left_low[0], right_high[0]], [low[0], high[0]])
    np.testing.assert_allclose([left_high[0], right_low[0]], [-.5, .5])
    np.testing.assert_allclose([header_low[0], header_high[0]], [-.5, .5])
    assert header_low[2] == pytest.approx(2.05)
    for name in ('partition_left', 'partition_right', 'partition_header'):
        assert boxes[name]['position_m'][1] == pytest.approx((low[1] + high[1]) / 2)
    door, = config['environment']['doors']
    closed = _door_corners(door, 0.)
    assert closed[:, 0].min() > left_high[0]
    assert closed[:, 0].max() < right_low[0]
    assert door['hinge_position_m'][2] > 0.
    assert door['hinge_position_m'][2] + door['height_m'] < header_low[2]
    assert door['range_degrees'][0] <= 0. < door['angle_degrees'] <= door['range_degrees'][1]


def test_door_can_swing_through_entire_range_without_intersecting_jambs():
    config = load_scene_config('doorway')
    boxes = _boxes(config)
    door, = config['environment']['doors']
    obstacles = []
    for name in ('partition_left', 'partition_right'):
        lower, upper = _bounds(boxes[name])
        obstacles.append(np.array(list(product((lower[0], upper[0]), (lower[1], upper[1])))))
    for angle in np.linspace(*door['range_degrees'], 111):
        leaf = _door_corners(door, angle)
        assert not any(_rectangles_overlap(leaf, obstacle) for obstacle in obstacles), angle


@pytest.mark.parametrize('template', ['room', 'doorway'])
def test_pace30_cameras_and_g1_payload_footprints_remain_inside_room(template, g1_geometry):
    config = resample_scene_config(load_scene_config(template))
    assert len(config['trajectory']) == 121
    low, high = _interior(config)
    radius = _carried_footprint_radius(config, g1_geometry[template])
    for frame in config['trajectory']:
        rig = rig_transform(frame['position_m'], frame['yaw_degrees'])
        camera = rig @ head_camera_pose(config['camera'], frame)
        assert np.all(camera[:3, 3] > low) and np.all(camera[:3, 3] < high)
        center = rig[:2, 3]
        # The circle encloses G1 links and the carried box at every yaw.
        assert np.all(center - radius > low[:2])
        assert np.all(center + radius < high[:2])
        if template == 'doorway':
            front_face = _bounds(_boxes(config)['partition_left'])[0][1]
            assert center[1] + radius < front_face
            # Independently interpolated position/yaw remains aimed at the
            # opening, though intermediate views need not track its exact center.
            direction = camera[:2, 2] / np.linalg.norm(camera[:2, 2])
            target = np.array([0., _boxes(config)['partition_header']['position_m'][1]])
            desired = target - camera[:2, 3]
            desired /= np.linalg.norm(desired)
            assert direction @ desired > np.cos(np.deg2rad(3.))
    for frame in config['held_out']:
        eye = np.asarray(frame['eye'])
        assert np.all(eye > low) and np.all(eye < high)


@pytest.mark.parametrize('template', ['room', 'doorway'])
def test_reconstruction_bounds_cover_enclosure_and_all_door_angles(template):
    config = load_scene_config(template)
    origin = np.asarray(config['reconstruction']['origin_m'])
    maximum = origin + config['reconstruction']['volume_length_m']
    for obj in config['environment']['objects']:
        lower, upper = _bounds(obj)
        assert np.all(lower >= origin) and np.all(upper <= maximum)
    for door in config['environment'].get('doors', []):
        for angle in np.linspace(*door['range_degrees'], 111):
            corners = _door_corners(door, angle)
            assert np.all(corners >= origin[:2]) and np.all(corners <= maximum[:2])
        assert origin[2] <= door['hinge_position_m'][2]
        assert door['hinge_position_m'][2] + door['height_m'] <= maximum[2]


@pytest.mark.parametrize('template', ['room', 'doorway'])
def test_enlarged_rooms_keep_g1_real_scale_and_camera_above_head(template, g1_geometry):
    config = load_scene_config(template)
    assert config['robot']['model'] == 'unitree_g1'
    assert config['robot']['pose'] == 'carry'
    assert config['torso']['enabled'] is False
    assert config['box']['enabled'] is True
    reference = load_scene_config('g1_stairs')
    assert config['robot'] == reference['robot']
    assert config['box'] == reference['box']
    geometry = g1_geometry[template]
    # The room grows, but the G1 retains its roughly 1.32 m physical height.
    assert 1.2 < geometry['upper'][2] - geometry['lower'][2] < 1.4
    assert 0 <= geometry['lower'][2] < .01
    assert geometry['upper'][0] - geometry['lower'][0] < 1.
    assert geometry['upper'][2] < config['camera']['position_in_rig_m'][2] < 1.45
    assert all(frame['head_yaw_degrees'] == frame['head_pitch_degrees'] == 0
               for frame in config['trajectory'])
    low, high = _interior(config)
    assert config['camera']['depth_trunc_m'] > np.linalg.norm(high - low)
