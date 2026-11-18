"""Measured prefix playback, source image selection and offline export checks."""

from html.parser import HTMLParser
import json

import numpy as np
import open3d as o3d
from PIL import Image
import pytest

from wtumi_recon.memory_visuals import export_memory_visuals
from wtumi_recon.occupancy import EvidenceOccupancyGrid
from wtumi_recon.sequence import SequenceWriter


CASES = ('always_occluded', 'observe_then_occluded')


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.data = ''
        self.external_scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.in_data = attrs.get('id') == 'memory-history-data'
            if 'src' in attrs:
                self.external_scripts.append(attrs['src'])

    def handle_data(self, value):
        if self.in_data:
            self.data += value

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_data = False


def _fixture(tmp_path):
    root = tmp_path / 'output'
    camera = {'width': 8, 'height': 8, 'K': [[8, 0, 3.5], [0, 8, 3.5], [0, 0, 1]], 'depth_trunc_m': 5.}
    histories, sequences = {}, {}
    for column, case in enumerate(CASES):
        sequence = root / 'sequences' / case
        writer = SequenceWriter(sequence, camera, 'stairs', 'simulator_ground_truth')
        rows = []
        for index, name in enumerate(('first', 'second')):
            pose = np.eye(4)
            pose[0, 3] = index * .2
            # The mixed first view is bright green; masked box views are blue.
            raw_color = (20, 220, 40) if case == 'observe_then_occluded' and index == 0 else (30, 40, 210)
            raw = np.full((8, 8, 3), raw_color, dtype=np.uint8)
            rgb = raw.copy()
            rgb[4:] = 0
            depth = np.ones((8, 8), dtype=np.float32)
            writer.add_frame(name, 'integration', rgb, depth, pose, float(index))
            Image.fromarray(raw).save(sequence / 'frames' / name / 'raw_rgb.png')

            run = root / 'history' / case / str(index+1)
            (run / 'renders' / 'novel').mkdir(parents=True)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector([[0, 0, index+1], [1+column, 0, index+1], [0, 1, index+1]])
            mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((3, 3), .5))
            assert o3d.io.write_triangle_mesh(str(run / 'mesh.ply'), mesh)
            grid = EvidenceOccupancyGrid([0, 0, 0], .5, 4)
            grid.frames = index+1
            grid.hit_count[2, index, index] = 1
            grid.state[2, index, index] = 2
            grid.save(run / 'occupancy.npz')
            Image.new('RGB', (100, 70), raw_color).save(run / 'renders' / 'novel' / 'comparison.png')
            rows.append({'id': name, 'frames': index+1, 'phase': 'occluded',
                         'run_dir': run.relative_to(root).as_posix(),
                         'stair_depth_coverage': .2+.1*index,
                         'surface_coverage_within_tolerance_fraction': .4+.1*index,
                         'roi_unknown_fraction': .8-.1*index})
        pose = np.eye(4)
        pose[1, 3] = .3
        writer.add_frame('novel', 'held_out', np.zeros((8, 8, 3), dtype=np.uint8),
                         np.ones((8, 8), dtype=np.float32), pose, 3.)
        sequences[case] = writer.finish()
        histories[case] = rows
    return root, histories, sequences


def test_actual_prefix_meshes_source_switch_and_offline_playback(tmp_path):
    root, histories, sequences = _fixture(tmp_path)
    originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    paths = export_memory_visuals(root, histories, sequences, preobserve_frames=1)
    assert all((root / path).is_file() for path in paths.values())
    page = _Page()
    page.feed((root / paths['history_viewer']).read_text())
    data = json.loads(page.data)
    assert page.external_scripts == []
    assert len(data['frames']) == 2
    # The second frame must come from its own mesh, not a repeated final/first map.
    assert data['frames'][0]['traces'][0]['z'] == [1., 1., 1.]
    assert data['frames'][1]['traces'][0]['z'] == [2., 2., 2.]
    assert len(data['frames'][0]['traces'][1]['x']) == 1
    assert len(data['frames'][1]['traces'][1]['x']) == 2
    assert data['ranges'][0][1] > 2  # Both scenes share bounds covering the wider map.
    assert data['frames'][0]['source_labels'][1].startswith('No box')
    assert data['frames'][1]['source_labels'][1].startswith('Box present')
    assert 'Play' in (root / paths['history_viewer']).read_text()
    for key in ('capture_animation', 'occupancy_animation', 'reconstruction_animation'):
        with Image.open(root / paths[key]) as animation:
            assert animation.n_frames == 2
            assert animation.info['loop'] == 0
    with Image.open(root / paths['capture_animation']) as animation:
        animation.seek(0)
        pixels = animation.convert('RGB')
        assert pixels.getpixel((180, 200)) == (30, 40, 210)
        assert pixels.getpixel((540, 200)) == (20, 220, 40)
        animation.seek(1)
        assert animation.convert('RGB').getpixel((540, 200)) == (30, 40, 210)
    for path, data in originals.items():
        assert path.read_bytes() == data


def test_rejects_nonmatching_prefixes_before_export(tmp_path):
    root, histories, sequences = _fixture(tmp_path)
    histories['observe_then_occluded'][1]['frames'] = 1
    with pytest.raises(ValueError, match='prefix lengths'):
        export_memory_visuals(root, histories, sequences, preobserve_frames=1)
    assert not (root / 'capture_history.gif').exists()


def test_rejects_history_paths_outside_result(tmp_path):
    root, histories, sequences = _fixture(tmp_path)
    histories['always_occluded'][0]['run_dir'] = '../outside'
    with pytest.raises(ValueError, match='inside the output'):
        export_memory_visuals(root, histories, sequences, preobserve_frames=1)


@pytest.mark.parametrize('count', [0, 2, True])
def test_switch_must_leave_both_phases(tmp_path, count):
    root, histories, sequences = _fixture(tmp_path)
    with pytest.raises(ValueError, match='both pre-observation'):
        export_memory_visuals(root, histories, sequences, preobserve_frames=count)


def test_mp4_streams_actual_frames_and_gallery_uses_video(tmp_path, monkeypatch):
    from wtumi_recon import video
    root, histories, sequences = _fixture(tmp_path)
    encoded = {}
    monkeypatch.setattr(video, 'require_video_encoder', lambda: {'codec': 'libx264'})

    def record_video(path, frames, fps):
        values = []
        for frame in frames:
            # Inspect content at the raw camera's corresponding scaled location.
            values.append((frame.size, frame.getpixel((1080, 400))))
        encoded[path.name] = values
        path.write_bytes(b'video test stub')
        width, height = values[0][0]
        return {'codec': 'H.264', 'width': width, 'height': height,
                'fps': fps, 'frame_count': len(values), 'duration_s': len(values)/fps}

    monkeypatch.setattr(video, 'save_mp4', record_video)
    paths = export_memory_visuals(root, histories, sequences, preobserve_frames=1,
                                  playback={'fps': 15, 'scale': 2, 'mp4': True})
    assert encoded['capture_history.mp4'] == [((1440, 1240), (20, 220, 40)),
                                              ((1440, 1240), (30, 40, 210))]
    assert encoded['reconstruction_history.mp4'][0][0] == (2560, 1000)
    assert encoded['occupancy_history.mp4'][0][0] == (1600, 956)
    page = (root/'index.html').read_text()
    assert page.count('<video controls loop playsinline') == 3
    assert '_history.gif' not in page
    assert '15 frames per second' in page
    assert not list(root.glob('*.gif'))
    meta = json.loads((root/paths['animation_metadata']).read_text())
    assert meta['capture_timestamps_s'] == [0, 1]
    assert all(value['frame_count'] == 2 and value['fps'] == 15 for value in meta['animations'].values())
    history = _Page()
    history.feed((root/'history_viewer.html').read_text())
    assert json.loads(history.data)['target_fps'] == 15


def test_scaled_frames_retain_high_resolution_camera_detail(tmp_path):
    from wtumi_recon.memory_visuals import _capture_frames, _validate
    root, histories, sequences = _fixture(tmp_path)
    # This 640-pixel raw image has single-pixel stripes. A pipeline that first
    # downsamples to the legacy 340-pixel panel would destroy this detail.
    stripes = np.zeros((480, 640, 3), np.uint8)
    stripes[:, ::2] = 255
    for case in CASES:
        Image.fromarray(stripes).save(root/'sequences'/case/'frames/first/raw_rgb.png')
    loaded, views, _, _ = _validate(root, histories, sequences, 1)
    frame = next(_capture_frames(loaded, views, 1, 2))
    row = np.asarray(frame)[400, 58:650, 0]
    assert np.count_nonzero(np.abs(np.diff(row.astype(int))) > 50) > 400


@pytest.mark.parametrize('playback', [{'fps': 0}, {'fps': float('nan')}, {'fps': True},
                                      {'scale': 3}, {'scale': True}, {'mp4': 'yes'}])
def test_invalid_playback_does_not_export(tmp_path, playback):
    root, histories, sequences = _fixture(tmp_path)
    with pytest.raises(ValueError, match='playback'):
        export_memory_visuals(root, histories, sequences, 1, playback)
    assert not (root/'index.html').exists()
