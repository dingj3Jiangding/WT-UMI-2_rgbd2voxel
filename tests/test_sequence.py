"""Portable capture metadata, lazy image reads and strict geometry conventions."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.geometry import backproject_depth, transform_points
from wtumi_recon.sequence import RGBDSequence, SequenceWriter


def _camera():
    return {'width': 3, 'height': 2, 'K': [[4., 0., 1.], [0., 5., .5], [0., 0., 1.]], 'depth_trunc_m': 4.0}


def _data():
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)*10
    depth = np.array([[0, 1, 1.2], [1.5, 2, 2.5]], dtype=np.float32)
    return rgb, depth


def _write(root, separate_reference=False):
    rgb, depth = _data()
    writer = SequenceWriter(root, _camera(), 'portable_room', 'simulator_ground_truth',
                            metadata={'producer': 'analytic_test', 'world_up': '+y'})
    pose = np.eye(4)
    reference = {'reference_rgb': np.full_like(rgb, 120), 'reference_depth_m': np.ones_like(depth)} if separate_reference else {}
    writer.add_frame('front', 'integration', rgb, depth, pose, 0., **reference)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [1, 2, 3]
    writer.add_frame('novel', 'held_out', rgb, depth.astype(np.float64), pose, .05)
    return writer.finish()


def _edit(path, transform):
    data = json.loads(path.read_text())
    transform(data)
    path.write_text(json.dumps(data))


def test_roundtrip_preserves_geometry_rgb_depth_and_capture_order(tmp_path):
    path = _write(tmp_path/'source')
    dataset = RGBDSequence(path)
    assert dataset.reference_kind == 'simulator_ground_truth'
    assert dataset.scene_name == 'portable_room'
    assert dataset.camera['fx'] == 4 and dataset.camera['fy'] == 5
    assert dataset.metadata['producer'] == 'analytic_test'
    assert [v['id'] for v in dataset.views] == ['front', 'novel']
    assert [v['timestamp_s'] for v in dataset.views] == [0., .05]
    rgb, depth = dataset.read(dataset.views[1])
    expected_rgb, expected_depth = _data()
    np.testing.assert_array_equal(rgb, expected_rgb)
    np.testing.assert_array_equal(depth, expected_depth)
    assert depth.dtype == np.float32
    camera_points = backproject_depth(depth, np.array(dataset.camera['K']))
    points = transform_points(camera_points, np.array(dataset.views[1]['T_wc']))
    # Pixel (2,1): camera point (0.625,0.25,2.5); rotate around Z then translate.
    np.testing.assert_allclose(points[1, 2], [.75, 2.625, 5.5])
    rgb[:] = 0
    np.testing.assert_array_equal(dataset.read('novel')[0], expected_rgb)


def test_dataset_can_be_relocated_without_editing_paths(tmp_path, monkeypatch):
    path = _write(tmp_path/'source')
    shutil.copytree(path.parent, tmp_path/'moved')
    monkeypatch.chdir(tmp_path.parent)
    moved = RGBDSequence(tmp_path/'moved/manifest.json')
    np.testing.assert_array_equal(moved.read('front')[1], _data()[1])
    assert all(not Path(frame['rgb']).is_absolute() for frame in moved.manifest['frames'])


def test_separate_reference_and_safe_missing_provenance_fallback(tmp_path):
    path = _write(tmp_path/'source', separate_reference=True)
    dataset = RGBDSequence(path)
    assert dataset.read('front')[1][0, 0] == 0
    np.testing.assert_array_equal(dataset.read_reference('front')[1], 1)
    np.testing.assert_array_equal(dataset.read_reference('front')[0], 120)
    np.testing.assert_array_equal(dataset.read_reference('novel')[1], dataset.read('novel')[1])
    _edit(path, lambda m: m.pop('reference_kind'))
    assert RGBDSequence(path).reference_kind == 'observed_rgbd'


def test_metadata_validation_does_not_load_held_out_images(tmp_path, monkeypatch):
    path = _write(tmp_path/'source')
    opened = []
    original = np.load

    def track(filename, *args, **kwargs):
        opened.append(Path(filename))
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(np, 'load', track)
    # A broken held-out array does not get read by metadata validation or front fusion read.
    held_path = path.parent/'frames/novel/depth_m.npy'
    held_path.write_bytes(b'broken depth content')
    dataset = RGBDSequence(path)
    assert opened == []
    dataset.read('front')
    assert opened == [path.parent/'frames/front/depth_m.npy']
    with pytest.raises(ValueError):
        dataset.read('novel')


@pytest.mark.parametrize('key,value', [
    ('depth_unit', 'millimetre'), ('depth_definition', 'range'),
    ('camera_convention', 'opengl'), ('pixel_centers', 'half_integer'),
    ('format', 'unrecognized'), ('reference_kind', 'assumed_truth'),
])
def test_incompatible_geometry_or_reference_semantics_are_rejected(tmp_path, key, value):
    path = _write(tmp_path/'source')
    _edit(path, lambda m: m.update({key: value}))
    with pytest.raises(ValueError):
        RGBDSequence(path)


def test_frame_path_traversal_and_symlink_escape_are_rejected(tmp_path):
    path = _write(tmp_path/'source')
    original = path.read_bytes()
    outside = tmp_path/'outside.png'
    Image.fromarray(_data()[0]).save(outside)
    _edit(path, lambda m: m['frames'][0].update(rgb='../outside.png'))
    with pytest.raises(ValueError, match='escapes'):
        RGBDSequence(path)
    path.write_bytes(original)
    (path.parent/'external.png').symlink_to(outside)
    _edit(path, lambda m: m['frames'][0].update(rgb='external.png'))
    with pytest.raises(ValueError, match='escapes'):
        RGBDSequence(path)


@pytest.mark.parametrize('corruption', ['duplicate_id', 'same_pose', 'reflection', 'backward_time', 'missing_pair', 'skew', 'mismatched_focal'])
def test_malformed_metadata_is_rejected(tmp_path, corruption):
    path = _write(tmp_path/'source')
    def corrupt(m):
        if corruption == 'duplicate_id': m['frames'][1]['id'] = 'front'
        elif corruption == 'same_pose': m['frames'][1]['T_wc'] = m['frames'][0]['T_wc']
        elif corruption == 'reflection': m['frames'][0]['T_wc'][0][0] = -1
        elif corruption == 'backward_time': m['frames'][1]['timestamp_s'] = -1
        elif corruption == 'missing_pair': m['frames'][0]['reference_rgb'] = m['frames'][0]['rgb']
        elif corruption == 'skew': m['camera']['K'][0][1] = .2
        elif corruption == 'mismatched_focal': m['camera']['fx'] = 8
    _edit(path, corrupt)
    with pytest.raises(ValueError):
        RGBDSequence(path)


@pytest.mark.parametrize('bad_depth', [
    np.ones((2, 3), dtype=np.uint16), np.ones((3, 2), dtype=np.float32),
    np.full((2, 3), np.nan, dtype=np.float32), np.full((2, 3), -1., dtype=np.float32),
    np.full((2, 3), 1e100, dtype=np.float64),
])
def test_depth_storage_rejects_units_shapes_and_invalid_samples(tmp_path, bad_depth):
    path = _write(tmp_path/'source')
    np.save(path.parent/'frames/front/depth_m.npy', bad_depth)
    dataset = RGBDSequence(path)
    with pytest.raises(ValueError):
        dataset.read('front')


def test_rgb_png_mode_must_be_rgb_without_implicit_conversion(tmp_path):
    path = _write(tmp_path/'source')
    Image.fromarray(np.zeros((2, 3, 4), dtype=np.uint8)).save(path.parent/'frames/front/rgb.png')
    with pytest.raises(ValueError, match='RGB PNG'):
        RGBDSequence(path).read('front')


def test_writer_never_overwrites_and_rejects_bad_frame_before_writing_it(tmp_path):
    path = _write(tmp_path/'source')
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        SequenceWriter(path.parent, _camera(), 'other', 'observed_rgbd')
    assert path.read_bytes() == original
    writer = SequenceWriter(tmp_path/'fresh', _camera(), 'room', 'observed_rgbd')
    rgb, depth = _data()
    with pytest.raises(ValueError):
        writer.add_frame('../escape', 'integration', rgb, depth, np.eye(4), 0)
    with pytest.raises(ValueError):
        writer.add_frame('bad', 'integration', rgb.astype(float), depth, np.eye(4), 0)
    assert list(writer.output_dir.iterdir()) == []
    writer.add_frame('front', 'integration', rgb, depth, np.eye(4), 0)
    before = (writer.output_dir/'frames/front/depth_m.npy').read_bytes()
    with pytest.raises(ValueError, match='duplicate'):
        writer.add_frame('front', 'integration', rgb, depth, np.eye(4), 1)
    assert (writer.output_dir/'frames/front/depth_m.npy').read_bytes() == before
    with pytest.raises(ValueError, match='held_out'):
        writer.finish()
    assert not (writer.output_dir/'manifest.json').exists()
    novel = np.eye(4); novel[0, 3] = .2
    writer.add_frame('novel', 'held_out', rgb, depth, novel, 1)
    writer.finish()
    with pytest.raises(RuntimeError):
        writer.add_frame('extra', 'integration', rgb, depth, novel, 2)


def test_writer_rejects_invalid_camera_before_creating_directory(tmp_path):
    camera = _camera(); camera['width'] = True
    with pytest.raises(ValueError):
        SequenceWriter(tmp_path/'bad', camera, 'room', 'observed_rgbd')
    assert not (tmp_path/'bad').exists()
