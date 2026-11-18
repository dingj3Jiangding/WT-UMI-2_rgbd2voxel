"""Verify single-scene orchestration and portable prefix result contracts."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from wtumi_recon.scene_bundle import load_scene_config
from wtumi_recon.scene_pipeline import run_scene
from wtumi_recon.sequence import RGBDSequence
from wtumi_recon.config import load_run_config


def test_existing_scene_run_is_preserved(tmp_path):
    (tmp_path / 'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError, match='new or empty'):
        run_scene('room', tmp_path)
    assert (tmp_path / 'keep.txt').read_text() == 'keep'


@pytest.mark.skipif(os.environ.get('WTUMI_TEST_MUJOCO_RENDER') != '1',
                    reason='Set WTUMI_TEST_MUJOCO_RENDER=1 with working MuJoCo GL')
@pytest.mark.parametrize('template', ['room', 'stairs', 'doorway'])
def test_scene_pipeline_real_capture_prefixes_and_portable_replay(tmp_path, template):
    cfg = {k: v for k, v in load_scene_config(template).items() if not k.startswith('_')}
    cfg['camera'].update(width=80, height=64)
    cfg['reconstruction']['resolution'] = 64
    cfg['occupancy']['resolution'] = 16
    cfg['trajectory'] = [cfg['trajectory'][i] for i in (0, len(cfg['trajectory']) // 2, -1)]
    cfg['held_out'] = cfg['held_out'][:1]
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    out = tmp_path / 'run'
    result = run_scene(path, out)
    assert result['format'] == 'wtumi-scene-run-v1'
    assert result['integration_frames'] == 3
    sequence = RGBDSequence(out / result['paths']['capture_manifest'])
    ids = [v['id'] for v in sequence.views if v['split'] == 'integration']
    held = [v['id'] for v in sequence.views if v['split'] == 'held_out']
    history = json.loads((out / 'history.json').read_text())
    assert len(history) == 3
    for i, row in enumerate(history):
        directory = out / row['run_dir']
        metrics = json.loads((directory / 'metrics.json').read_text())
        assert metrics['integrated_frame_ids'] == ids[:i + 1]
        assert metrics['newly_integrated_frame_ids'] == ids[i:i + 1]
        assert not set(metrics['integrated_frame_ids']) & set(held)
        for filename in ('mesh.ply', 'occupancy.npz', 'tsdf_volume.npz'):
            assert (directory / filename).is_file()
        assert row['held_out']['overlap_pixels'] > 0
        with np.load(directory / 'tsdf_volume.npz') as volume:
            assert np.any(volume['weight'] > 0)
    for filename in ('index.html', 'inputs.gif', 'reconstruction.gif', 'reconstruction/viewer.html'):
        assert (out / filename).is_file()
    # Delete the source layout and resolve everything from a moved result.
    moved = tmp_path / 'moved'
    shutil.move(out, moved)
    path.unlink()
    RGBDSequence(moved / result['paths']['capture_manifest'])
    load_scene_config(moved / result['paths']['scene_config'])
    replay = load_run_config(moved / 'configs/prefix_003.yaml')
    assert Path(replay['source']['manifest']).is_file()
