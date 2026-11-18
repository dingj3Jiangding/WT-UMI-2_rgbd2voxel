"""The dense preset preserves the scene and renders new poses at a fixed cadence."""

from pathlib import Path

import numpy as np
import yaml

from wtumi_recon.memory_demo import load_memory_demo_config
from wtumi_recon.rig_capture import load_rig_capture_config


ROOT = Path(__file__).resolve().parents[1]


def test_dense_capture_preserves_keyframes_scene_and_held_out_views():
    base = load_rig_capture_config(ROOT / 'configs/capture/stairs_rig.yaml')
    dense = load_rig_capture_config(ROOT / 'configs/capture/stairs_rig_hq.yaml')
    assert dense['camera']['width'] == 640
    assert dense['camera']['height'] == 480
    for key in ('scene_name', 'stairs', 'box', 'torso', 'roi', 'held_out'):
        assert dense[key] == base[key]
    for key in ('fovy_degrees', 'depth_trunc_m', 'position_in_rig_m', 'target_in_rig_m'):
        assert dense['camera'][key] == base['camera'][key]
    frames = dense['trajectory']
    assert len(frames) == len({row['id'] for row in frames}) == 61
    np.testing.assert_allclose(np.diff([row['timestamp_s'] for row in frames]), 1/15, atol=1e-14)
    assert frames[::15] == base['trajectory']
    for index, row in enumerate(frames):
        segment = min(index // 15, 3)
        fraction = (index - segment * 15) / 15
        first, last = base['trajectory'][segment:segment+2]
        expected = np.asarray(first['position_m']) * (1-fraction) + np.asarray(last['position_m']) * fraction
        np.testing.assert_allclose(row['position_m'], expected, atol=1e-12)
        np.testing.assert_allclose(row['yaw_degrees'], first['yaw_degrees'] * (1-fraction) + last['yaw_degrees'] * fraction, atol=1e-12)


def test_hq_demo_has_a_two_second_clear_prefix_and_matched_occupancy_ray_spacing():
    cfg = load_memory_demo_config(ROOT / 'configs/demos/stairs_memory_hq.yaml')
    base = yaml.safe_load((ROOT / 'configs/demos/stairs_memory.yaml').read_text())
    assert cfg['preobserve_frames'] == 30
    frames = cfg['capture']['trajectory']
    assert all(row['timestamp_s'] < 2 for row in frames[:30])
    assert frames[30]['timestamp_s'] == 2
    assert cfg['incremental_prefixes'] is True
    assert cfg['playback'] == {'fps': 15, 'scale': 2, 'mp4': True}
    assert cfg['reconstruction']['resolution'] == base['reconstruction']['resolution']
    assert cfg['occupancy']['pixel_stride'] / cfg['capture']['camera']['width'] == base['occupancy']['pixel_stride'] / 160
