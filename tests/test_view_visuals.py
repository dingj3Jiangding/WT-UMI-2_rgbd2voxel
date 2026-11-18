"""M8 shares positions while preserving different measured head directions."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.view_visuals import CASES, export_view_visuals
from test_memory_visuals import _fixture as memory_fixture, _Page


def _fixture(tmp_path):
    root, old_histories, old_sequences = memory_fixture(tmp_path)
    histories, sequences = {}, {}
    for old, new in zip(old_histories, CASES):
        old_sequence = Path(old_sequences[old])
        new_sequence = old_sequence.parent.with_name(new)
        old_sequence.parent.rename(new_sequence)
        (root/'history'/old).rename(root/'history'/new)
        histories[new] = old_histories[old]
        for row in histories[new]:
            row['run_dir'] = row['run_dir'].replace('/'+old+'/', '/'+new+'/')
        sequences[new] = new_sequence/'manifest.json'
    path = sequences['head_scan']
    manifest = json.loads(path.read_text())
    angle = np.deg2rad(25)
    rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    view = manifest['frames'][1]
    pose = np.array(view['T_wc'])
    pose[:3, :3] = rotation
    view['T_wc'] = pose.tolist()
    path.write_text(json.dumps(manifest))
    return root, histories, sequences


def test_head_directions_and_actual_prefixes_in_shared_playback(tmp_path):
    root, histories, sequences = _fixture(tmp_path)
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    paths = export_view_visuals(root, histories, sequences, {'mp4': False, 'scale': 1, 'fps': 15})
    assert all((root/path).is_file() for path in paths.values())
    page = _Page()
    text = (root/paths['history_viewer']).read_text()
    page.feed(text)
    data = json.loads(page.data)
    assert data['format'] == 'wtumi-view-history-viewer-v1'
    assert data['case_ids'] == list(CASES)
    assert data['labels'] == ['Fixed head camera', 'Independent head scan']
    assert 'preobserve_frames' not in data
    assert 'pre-observation' not in text
    assert 'Visual memory' not in text
    assert page.external_scripts == []
    assert len(data['frames']) == 2
    assert data['frames'][0]['traces'][0]['z'] == [1., 1., 1.]
    assert data['frames'][1]['traces'][0]['z'] == [2., 2., 2.]
    # Same camera centers and path, but the second optical-axis segment changes.
    traces = data['frames'][1]['traces']
    assert traces[1]['x'] == traces[3]['x']
    fixed = np.array([traces[4][axis] for axis in ('x', 'y', 'z')]).T
    scan = np.array([traces[5][axis] for axis in ('x', 'y', 'z')]).T
    np.testing.assert_array_equal(fixed[0], scan[0])
    assert not np.allclose(fixed[1], scan[1])
    assert traces[5]['name'] == 'Current optical axis'
    gallery = (root/paths['gallery']).read_text()
    assert 'M8' in gallery and 'Independent head scan' in gallery
    assert 'observe first' not in gallery.lower()
    assert 'runs/fixed_head/viewer.html' in gallery
    for case in CASES:
        assert 'Box present throughout' in data['frames'][1]['source_labels'][CASES.index(case)]
    for key in ('capture_animation', 'reconstruction_animation', 'occupancy_animation'):
        with Image.open(root/paths[key]) as image:
            assert image.n_frames == 2
    for path, content in original.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize('mismatch', ['camera_center', 'timestamp', 'held_out_pose'])
def test_rejects_nonshared_capture_contract_before_export(tmp_path, mismatch):
    root, histories, sequences = _fixture(tmp_path)
    path = sequences['head_scan']
    manifest = json.loads(path.read_text())
    if mismatch == 'camera_center':
        manifest['frames'][1]['T_wc'][0][3] += .1
    elif mismatch == 'timestamp':
        manifest['frames'][1]['timestamp_s'] += .1
    else:
        manifest['frames'][-1]['T_wc'][0][3] += .1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='shared'):
        export_view_visuals(root, histories, sequences, {'mp4': False})
    assert not (root/'capture_history.gif').exists()


def test_mp4_export_decodes_all_three_actual_streams(tmp_path):
    ffmpeg = pytest.importorskip('imageio_ffmpeg')
    root, histories, sequences = _fixture(tmp_path)
    paths = export_view_visuals(root, histories, sequences, {'mp4': True, 'scale': 1, 'fps': 15})
    metadata = json.loads((root/paths['animation_metadata']).read_text())
    gallery = (root/paths['gallery']).read_text()
    assert gallery.count('<video ') == 3
    assert '.gif' not in gallery
    for name in ('capture', 'reconstruction', 'occupancy'):
        path = root/paths[name+'_animation']
        reader = ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        decoded_metadata = next(reader)
        frames = list(reader)
        assert decoded_metadata['fps'] == 15
        assert len(frames) == 2
        assert frames[0] != frames[1]
        record = metadata['animations'][name]
        assert tuple((record['width'], record['height'])) == decoded_metadata['size']
        assert record['frame_count'] == 2
