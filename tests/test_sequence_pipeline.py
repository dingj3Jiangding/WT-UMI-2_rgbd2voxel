"""A saved observation source must preserve fusion, reference and pose contracts."""

from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.sequence import SequenceWriter, RGBDSequence
from wtumi_recon.synthetic import SyntheticScene
from wtumi_recon.tsdf import Open3DTSDFMap
from wtumi_recon.viewer import export_viewer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("reference_kind", ["simulator_ground_truth", "observed_rgbd"])
def test_captured_sequence_replays_core_without_scene_renderer_or_heldout_leak(tmp_path, monkeypatch, reference_kind):
    raw = yaml.safe_load((ROOT/'configs/scenes/multiview_cube.yaml').read_text())
    raw['camera'].update(width=100, height=80, fx=90., fy=90., cx=49.5, cy=39.5)
    raw['reconstruction']['resolution'] = 80
    cfg_path = tmp_path/'synthetic.yaml'
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg=load_run_config(cfg_path)
    scene=SyntheticScene.from_config(cfg['scene'])
    writer=SequenceWriter(tmp_path/'capture', cfg['camera'], 'saved_scene', reference_kind,
                          metadata={'producer':'test_raycast','world_up':[0,0,1]})
    for i,view in enumerate(cfg['views']):
        rgb, depth=scene.render(np.asarray(cfg['camera']['K']),np.asarray(view['T_wc']),100,80)
        clean=depth.copy()
        if view['split']=='integration':
            depth=depth.copy(); depth[::4,::4]=0
        writer.add_frame(view['id'],view['split'],rgb,depth,view['T_wc'],i,
                         reference_rgb=rgb,reference_depth_m=clean)
    manifest=writer.finish()
    sequence_cfg={'schema_version':3,'source':{'type':'rgbd_sequence','manifest':'capture/manifest.json'},
                  'reconstruction':raw['reconstruction'],'evaluation':raw['evaluation']}
    run_cfg=tmp_path/'sequence.yaml'
    run_cfg.write_text(yaml.safe_dump(sequence_cfg))
    calls=[]
    real_integrate=Open3DTSDFMap.integrate
    def track(self,rgbd,intrinsic,pose):
        calls.append(np.asarray(rgbd.depth).copy())
        return real_integrate(self,rgbd,intrinsic,pose)
    monkeypatch.setattr(Open3DTSDFMap,'integrate',track)
    real_read=RGBDSequence.read
    real_reference=RGBDSequence.read_reference
    n_fused=sum(v['split']=='integration' for v in cfg['views'])
    def check_read(self,view):
        assert view['split']=='integration'
        return real_read(self,view)
    def check_reference(self,view):
        if view['split']=='held_out':assert len(calls)==n_fused
        return real_reference(self,view)
    monkeypatch.setattr(RGBDSequence,'read',check_read)
    monkeypatch.setattr(RGBDSequence,'read_reference',check_reference)
    def forbidden(*args,**kwargs):raise AssertionError('sequence reconstruction must not render source geometry')
    monkeypatch.setattr(SyntheticScene,'from_config',forbidden)
    out=tmp_path/'result'
    report=run_reconstruction(run_cfg,out)
    assert len(calls)==n_fused
    for actual, view in zip(calls,[v for v in cfg['views'] if v['split']=='integration']):
        np.testing.assert_array_equal(actual,np.load(out/'inputs'/view['id']/'depth_m.npy'))
        clean=np.load(out/'references'/view['id']/'depth_m.npy')
        assert np.count_nonzero(clean)>np.count_nonzero(actual)
    assert report['evaluation_reference']==reference_kind
    assert report['data_source']['type']=='rgbd_sequence'
    assert report['summary']['held_out']['coverage']>.7
    assert report['summary']['held_out']['rmse_m']<.1
    replay=load_run_config(out/'config_resolved.yaml')
    original=load_run_config(run_cfg)
    for key in ('scene','camera','views','source','world_up'):
        assert replay[key]==original[key]
    saved=json.loads((out/'manifest.json').read_text())
    assert saved['world_up']==[0,0,1]
    assert all('timestamp_s' in f for f in saved['frames'])
    _, viewer=export_viewer(out)
    assert viewer['world_up']==[0,0,1]
    if reference_kind == 'observed_rgbd':
        from wtumi_recon.experiments import run_experiment
        experiment = {'schema_version':1, 'base_config':'sequence.yaml', 'resolutions':[80],
                      'seeds':[7], 'profiles':[{'name':'clean','depth_noise_std_m':0.,'dropout_probability':0.}]}
        experiment_path = tmp_path/'experiment.yaml'
        experiment_path.write_text(yaml.safe_dump(experiment))
        calls.clear()
        comparison = run_experiment(experiment_path, tmp_path/'experiment')
        assert comparison['reference_kind'] == 'observed_rgbd'
        assert comparison['held_out_references_identical'] is True
        assert comparison['clean_held_out_references_verified'] is False
    # An override that silently changes the dataset's calibration is not accepted.
    sequence_cfg['camera']={**original['camera'],'fx':12.}
    run_cfg.write_text(yaml.safe_dump(sequence_cfg))
    with pytest.raises(ValueError,match='cannot be overridden'):
        load_run_config(run_cfg)

