"""Short real-render checks of the 30 fps camera-to-voxel video path."""

import json
import os

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.scene_bundle import load_scene_config
from wtumi_recon.scene_pipeline import run_scene
from wtumi_recon.scene_presets import resample_scene_config


@pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                    reason='Enable actual MuJoCo rendering for this short smoke test')
@pytest.mark.parametrize('scene', ['room', 'doorway', 'g1_stairs'])
def test_pace30_fresh_rgbd_prefixes_encode_at_actual_30_fps(tmp_path, scene):
    ffmpeg = pytest.importorskip('imageio_ffmpeg')
    config = resample_scene_config(load_scene_config(scene))
    assert len(config['trajectory']) == 121
    # Keep production camera resolution and capture interval, while bounding
    # local smoke-test cost. The full three-scene run belongs on PACE.
    config['trajectory'] = config['trajectory'][:4]
    config['reconstruction']['resolution'] = 64
    config['occupancy']['resolution'] = 16
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = tmp_path / 'run'
    report = run_scene(path, result)
    capture = json.loads((result / 'capture/capture.json').read_text())
    comparison = json.loads((result / 'camera_voxel.json').read_text())
    assert report['integration_frames'] == comparison['frame_count'] == 4
    assert report['held_out_frames'] == 2
    assert comparison['playback_fps'] == 30
    ids = [row['id'] for row in capture['frames']]
    np.testing.assert_allclose([row['timestamp_s'] for row in capture['frames']], np.arange(4) / 30)
    images = []
    for index, row in enumerate(comparison['frames']):
        assert row['id'] == ids[index] and row['observations'] == index + 1
        with Image.open(result / row['raw_rgb']) as image:
            assert image.size == (640, 480)
            images.append(np.asarray(image).copy())
        metrics = json.loads((result / row['tsdf_volume']).with_name('metrics.json').read_text())
        assert metrics['integrated_frame_ids'] == ids[:index + 1]
    assert any(not np.array_equal(images[0], image) for image in images[1:])
    reader = ffmpeg.read_frames(str(result / report['paths']['camera_voxel_mp4']))
    video = next(reader)
    assert video['fps'] == 30
    assert video['size'] == (1280, 622)
    assert sum(1 for _ in reader) == 4
    assert comparison['video']['duration_s'] == pytest.approx(4 / 30)
