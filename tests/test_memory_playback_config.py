"""Reject invalid expensive playback jobs before rendering or writing outputs."""
import pytest
import yaml

from wtumi_recon.memory_demo import load_memory_demo_config, run_memory_demo
from test_memory_demo import _setup, _capture, CASES


@pytest.mark.parametrize('overrides', [
    {'incremental_prefixes': 1},
    {'playback': None},
    {'playback': {'fps': 0}},
    {'playback': {'fps': float('inf')}},
    {'playback': {'fps': True}},
    {'playback': {'scale': 3}},
    {'playback': {'mp4': 'yes'}},
    {'playback': {'unsupported': 1}},
])
def test_invalid_playback_precedes_capture(tmp_path, monkeypatch, overrides):
    from wtumi_recon import rig_capture
    path, cfg = _setup(tmp_path)
    cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('invalid playback reached capture'))
    with pytest.raises(ValueError):
        run_memory_demo(path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_video_preflight_precedes_capture(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture, video
    path, cfg = _setup(tmp_path)
    cfg['playback'] = {'mp4': True}
    path.write_text(yaml.safe_dump(cfg))
    def fail():
        raise ImportError('video encoder not installed')
    monkeypatch.setattr(video, 'require_video_encoder', fail)
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('missing encoder reached capture'))
    with pytest.raises(ImportError, match='encoder'):
        run_memory_demo(path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_incremental_memory_export_and_final_counts(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture
    from wtumi_recon.tsdf import Open3DTSDFMap
    import json
    path, cfg = _setup(tmp_path)
    cfg['incremental_prefixes'] = True
    path.write_text(yaml.safe_dump(cfg))
    assert load_memory_demo_config(path)['incremental_prefixes']
    monkeypatch.setattr(rig_capture, 'capture_rig', _capture)
    calls = []
    original = Open3DTSDFMap.integrate
    def track(self, *args):
        calls.append(1)
        return original(self, *args)
    monkeypatch.setattr(Open3DTSDFMap, 'integrate', track)
    result = run_memory_demo(path, tmp_path/'output')
    assert len(calls) == 4  # Two observations per condition, each fused once.
    assert result['incremental_prefixes']
    for case in CASES:
        out = tmp_path/'output'
        assert (out/'runs'/case/'tsdf_volume.npz').is_file()
        prefix = out/'history'/case/'frame_001'
        assert not (prefix/'tsdf_volume.npz').exists()
        assert (prefix/'mesh.ply').is_file()
        assert (prefix/'occupancy.npz').is_file()
        assert (prefix/'renders'/'evaluation_front_left'/'comparison.png').is_file()
        metrics = json.loads((out/'runs'/case/'metrics.json').read_text())
        assert metrics['integrated_frame_ids'] == ['approach', 'near']
