"""One captured sequence feeds all resolutions; no MuJoCo/GL needed here."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from wtumi_recon import mujoco_capture
from wtumi_recon.config import load_run_config
from wtumi_recon.demo import load_demo_config, run_sim_demo
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.synthetic import SyntheticScene


def _setup(tmp_path):
    root = tmp_path/'project'
    (root/'configs/capture').mkdir(parents=True)
    (root/'configs/demos').mkdir()
    (root/'assets').mkdir()
    (root/'assets/room.xml').write_text('<mujoco><worldbody><camera name="rgbd"/></worldbody></mujoco>')
    capture = {
        'schema_version': 1, 'scene_name': 'tiny_room', 'mjcf': '../../assets/room.xml',
        'camera': {'name': 'rgbd', 'width': 64, 'height': 48, 'depth_trunc_m': 3.0},
        'views': [
            {'id': 'front', 'split': 'integration', 'timestamp_s': 0., 'eye': [.5,.5,-.3], 'target': [.5,.5,1.], 'up':[0,-1,0]},
            {'id': 'left', 'split': 'integration', 'timestamp_s': .1, 'eye': [.2,.5,-.1], 'target': [.5,.5,1.], 'up':[0,-1,0]},
            {'id': 'novel', 'split': 'held_out', 'timestamp_s': .2, 'eye': [.75,.55,0.], 'target': [.5,.5,1.], 'up':[0,-1,0]},
        ],
    }
    (root/'configs/capture/room.yaml').write_text(yaml.safe_dump(capture))
    config = {
        'schema_version': 1, 'capture_config': '../capture/room.yaml',
        'reconstruction': {'backend':'uniform', 'volume_length_m':1.6, 'resolution':40,
                           'origin_m':[0,0,0], 'sdf_trunc_m':.1},
        'evaluation': {'depth_tolerance_m':.02}, 'resolutions':[40,48],
    }
    path = root/'configs/demos/demo.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


def _fake_capture(config_path, output_dir):
    """Actual rendered geometry exported through the public sequence contract."""
    cfg = mujoco_capture.load_capture_config(config_path)
    camera = {'width':64,'height':48,'K':[[70.,0.,31.5],[0.,70.,23.5],[0.,0.,1.]],'depth_trunc_m':3.}
    source = SyntheticScene.from_config({'objects':[{'type':'box', 'size_m':[.4,.4,.4],
        'translation_m':[.3,.3,.8], 'color_mode':'coordinate'}]})
    writer = SequenceWriter(output_dir, camera, cfg['scene_name'], 'simulator_ground_truth',
                            {'producer':'synthetic_capture_test', 'world_up':[0,-1,0]})
    for view in cfg['views']:
        rgb, depth = source.render(np.array(camera['K']), np.array(view['T_wc']), 64, 48)
        depth[(depth<=0)|~np.isfinite(depth)|(depth>=3)]=0
        rgb[depth==0]=0
        writer.add_frame(view['id'],view['split'],rgb,depth,view['T_wc'],view['timestamp_s'])
    return writer.finish()


def test_demo_captures_once_and_replays_after_relocation(tmp_path, monkeypatch):
    path, config = _setup(tmp_path)
    captured = []
    def capture_once(config_path, output_dir):
        captured.append((Path(config_path), Path(output_dir)))
        return _fake_capture(config_path, output_dir)
    monkeypatch.setattr(mujoco_capture, 'capture_mujoco', capture_once)
    messages = []
    out = tmp_path/'output'
    report = run_sim_demo(path, out, progress=messages.append)
    assert len(captured) == report['capture_count'] == 1
    assert captured[0][0] == (path.parent/'../capture/room.yaml').resolve()
    assert captured[0][1] == out/'capture'
    assert report['integration_frames'] == 2 and report['held_out_frames'] == 1
    assert report['resolutions'] == [40,48] and report['base_resolution'] == 40
    assert report['paired_inputs_verified']
    assert report['reference_kind'] == 'simulator_ground_truth'
    summary = json.loads((out/report['paths']['summary_json']).read_text())
    assert len(summary['runs']) == 2
    assert len({row['integration_depth_sha256'] for row in summary['runs']}) == 1
    assert len({row['held_out_depth_sha256'] for row in summary['runs']}) == 1
    for name, value in report['paths'].items():
        paths = value if isinstance(value, list) else [value]
        for relative in paths:
            assert not Path(relative).is_absolute()
            assert (out/relative).exists(), (name,relative)
    assert json.loads((out/'demo.json').read_text()) == report
    assert (out/'README.md').is_file()
    assert report['selected_held_out_metrics']['coverage'] > .5
    assert len([message for message in messages if 'Capturing' in message]) == 1
    assert yaml.safe_load((out/'reconstruction.yaml').read_text())['source']['manifest'] == 'capture/manifest.json'
    assert yaml.safe_load((out/'experiment.yaml').read_text())['base_config'] == 'reconstruction.yaml'

    moved = tmp_path/'relocated'
    shutil.copytree(out, moved)
    # Break original input path: relocated portable configs must resolve locally.
    (out/'capture/manifest.json').rename(out/'capture/manifest_original.json')
    replay_cfg = load_run_config(moved/'reconstruction.yaml')
    assert Path(replay_cfg['source']['manifest']) == moved/'capture/manifest.json'
    replay = run_reconstruction(moved/'reconstruction.yaml', tmp_path/'replayed')
    for key in ('coverage','mae_m','rmse_m'):
        assert replay['summary']['held_out'][key] == report['selected_held_out_metrics'][key]
    assert len(captured) == 1  # Replay uses saved observations, not a second capture.


@pytest.mark.parametrize('change', [
    {'resolutions':[40,40]}, {'resolutions':[True,48]}, {'resolutions':[48]},
    {'evaluation':{'depth_tolerance_m':float('nan')}},
    {'evaluation':{'depth_tolerance_m':0}},
    {'capture_config':'missing.yaml'},
    {'reconstruction':{'backend':'uniform','volume_length_m':-1,'resolution':40,'sdf_trunc_m':.1}},
    {'reconstruction':{'backend':'uniform','volume_length_m':1.6,'resolution':True,'sdf_trunc_m':.1}},
])
def test_invalid_demo_fails_before_capture_or_output_creation(tmp_path, monkeypatch, change):
    path, config = _setup(tmp_path)
    config.update(change)
    path.write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs):
        pytest.fail('invalid configuration reached capture')
    monkeypatch.setattr(mujoco_capture,'capture_mujoco',forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_sim_demo(path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_demo_does_not_overwrite_existing_output(tmp_path, monkeypatch):
    path, _ = _setup(tmp_path)
    out=tmp_path/'output';out.mkdir()
    (out/'keep.txt').write_text('existing data')
    def forbidden(*args, **kwargs):
        pytest.fail('existing output reached capture')
    monkeypatch.setattr(mujoco_capture,'capture_mujoco',forbidden)
    with pytest.raises(FileExistsError):
        run_sim_demo(path,out)
    assert (out/'keep.txt').read_text() == 'existing data'
    assert list(out.iterdir()) == [out/'keep.txt']
