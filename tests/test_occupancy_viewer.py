"""Offline occupancy export and slice-control correctness, without a browser."""

import base64
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import open3d as o3d
import pytest

from wtumi_recon.occupancy import EvidenceOccupancyGrid
from wtumi_recon.viewer import export_viewer, _VIEWER_SCRIPT


class PayloadParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.payload = ''
        self.modes = []
        self.external_scripts = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'script':
            self.active = attrs.get('id') == 'viewer-data'
            if 'src' in attrs:
                self.external_scripts.append(attrs['src'])
        if tag == 'button' and 'data-mode' in attrs:
            self.modes.append(attrs['data-mode'])

    def handle_data(self, text):
        if self.active:
            self.payload += text

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False


def result_with_occupancy(root):
    root.mkdir()
    mesh = o3d.geometry.TriangleMesh.create_box(.2, .2, .2)
    mesh.paint_uniform_color([.3, .5, .8])
    assert o3d.io.write_triangle_mesh(str(root / 'mesh.ply'), mesh)
    grid = EvidenceOccupancyGrid([0, 0, 0], .5, 4)
    pose = np.eye(4)
    pose[:3, 3] = [.25, 1.25, -.5]
    grid.integrate(np.array([[1.0]], dtype=np.float32), np.eye(3), pose, 3)
    grid.save(root / 'occupancy.npz')
    (root / 'manifest.json').write_text(json.dumps({
        'K': np.eye(3).tolist(), 'width': 1, 'height': 1,
        'frames': [{'id': 'front', 'split': 'integration', 'T_wc': pose.tolist()}],
    }))
    return grid


def test_occupancy_payload_preserves_xyz_states_and_separates_mode_from_cameras(tmp_path):
    root = tmp_path / 'result'
    grid = result_with_occupancy(root)
    original = (root / 'occupancy.npz').read_bytes()
    output, meta = export_viewer(root)
    parsed = PayloadParser()
    parsed.feed(output.read_text())
    assert parsed.external_scripts == []
    assert parsed.modes == ['0', '1', '2', '3']
    payload = json.loads(parsed.payload)
    assert payload['mode_trace_indices'] == {'0': [0], '1': [1], '2': [2], '3': [3]}
    assert payload['camera_indices'] == [4]
    state = np.frombuffer(base64.b64decode(payload['occupancy']['state_base64']), dtype=np.uint8).reshape((4,4,4))
    np.testing.assert_array_equal(state, grid.state)
    assert set(np.unique(state)) == {0, 1, 2}
    assert meta['occupancy']['counts'] == grid.summary()
    assert meta['occupancy']['voxel_size_m'] == .5
    assert meta['occupancy']['initial_slice_axis'] == 'y'
    assert meta['occupancy']['initial_slice_index'] == 2
    assert meta['occupancy']['slice_cells'] == 16
    assert 'state_base64' not in meta['occupancy']  # Sidecar stays a readable summary.
    assert 'whole-cell' in meta['occupancy']['semantics']
    assert 'wtumi-plot-ready' in output.read_text()
    assert (root / 'occupancy.npz').read_bytes() == original


def test_occupancy_slice_limit_is_explicit(tmp_path):
    root = tmp_path / 'result'
    result_with_occupancy(root)
    with pytest.raises(ValueError, match='occupancy slice has 16 cells'):
        export_viewer(root, max_voxels=15)
    assert not (root / 'viewer.html').exists()


def test_slice_javascript_with_mocked_plotly(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Optional Node.js source-level control test; no browser required')
    script = tmp_path / 'viewer-controls.js'
    script.write_text(_VIEWER_SCRIPT)
    result = subprocess.run([node, str(Path(__file__).with_name('viewer_controls_check.cjs')), str(script)],
                            capture_output=True, text=True, check=True)
    assert 'mocked controls pass' in result.stdout
