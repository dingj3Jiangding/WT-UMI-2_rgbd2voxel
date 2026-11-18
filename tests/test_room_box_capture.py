"""Real-render checks of the default head camera and carried-box occlusion."""

import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.scene_bundle import build_scene, load_scene_config
from wtumi_recon.scene_capture import capture_scene
from wtumi_recon.scene_presets import resample_scene_config
from wtumi_recon.sequence import RGBDSequence


RENDER = pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                          reason='Enable actual MuJoCo rendering for carried-box validation')


def _capture(root, config):
    root.mkdir()
    path = root / 'scene.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    built = build_scene(path, root / 'model')
    report = capture_scene(built['config'], Path(built['output']) / 'model.xml', root / 'capture')
    return report, RGBDSequence(root / 'capture/manifest.json')


def _mask(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


@RENDER
@pytest.mark.parametrize('template', ['room', 'doorway'])
def test_room_defaults_occlude_and_mask_payload_without_changing_held_outs(tmp_path, template):
    pytest.importorskip('mujoco')
    original = load_scene_config(template)
    config = resample_scene_config(original, width=96, height=72)
    # Sample start, middle and end of the actual pace30 schedule without
    # altering the camera mount, tilt, field of view or environment geometry.
    config['trajectory'] = [config['trajectory'][index] for index in (0, 60, 120)]
    carried, sequence = _capture(tmp_path / 'carried', config)
    pixels = config['camera']['width'] * config['camera']['height']
    fractions = []
    for frame in carried['frames']:
        directory = tmp_path / 'carried/capture' / frame['frame_directory']
        box = _mask(directory / 'box_mask.png')
        robot = _mask(directory / 'robot_mask.png')
        excluded = _mask(directory / 'excluded_mask.png')
        np.testing.assert_array_equal(excluded, box | robot)
        assert not np.any(box & robot)
        rgb, depth = sequence.read(frame['id'])
        assert not np.any(depth[excluded]) and not np.any(rgb[excluded])
        raw_depth = np.load(directory / 'raw_depth_m.npy', allow_pickle=False)
        np.testing.assert_array_equal(depth[~excluded], raw_depth[~excluded])
        assert frame['box_pixels'] == int(box.sum())
        fraction = frame['box_pixels'] / pixels
        # The box must occlude a meaningful image area while leaving enough
        # environmental evidence for mapping with this unchanged head mount.
        assert .1 < fraction < .8
        assert frame['valid_environment_pixels'] / pixels > .15
        assert 'T_wbox' in frame
        fractions.append({'frame': frame['id'], 'box_fraction': fraction,
                          'excluded_fraction': frame['excluded_pixels'] / pixels,
                          'environment_fraction': frame['valid_environment_pixels'] / pixels})
    environment = copy.deepcopy(config)
    environment.pop('robot')
    environment['box']['enabled'] = False
    reference, reference_sequence = _capture(tmp_path / 'environment', environment)
    for frame in carried['held_out_frame_ids']:
        rgb, depth = sequence.read(frame)
        clean_rgb, clean_depth = reference_sequence.read(frame)
        np.testing.assert_array_equal(rgb, clean_rgb)
        np.testing.assert_allclose(depth, clean_depth, atol=2e-5, rtol=0)
    assert reference['held_out_frame_ids'] == carried['held_out_frame_ids']
    evidence = {'scene': template, 'camera_width': 96, 'camera_height': 72,
                'scope': 'Three actual rendered pace30 poses; default camera geometry retained.',
                'frames': fractions, 'held_outs_match_environment_only': True}
    (tmp_path / 'occlusion_validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))
