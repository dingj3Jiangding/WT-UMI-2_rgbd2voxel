"""Controlled head-scan configuration, observation isolation and portable replay."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.mujoco_capture import intrinsics_from_fovy
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.render import look_at_pose
from wtumi_recon.rig_capture import build_rig_mjcf, head_camera_pose, load_rig_capture_config, rig_transform
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.synthetic import SyntheticScene
from wtumi_recon.view_demo import CASES, load_view_demo_config, run_view_demo

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path):
    captures = {}
    for case, name in zip(CASES, ('fixed', 'scan')):
        raw = yaml.safe_load((ROOT/f'configs/capture/stairs_view_{name}.yaml').read_text())
        raw['camera'].update(width=64, height=48)
        raw['trajectory'] = raw['trajectory'][:2]
        raw['held_out'] = raw['held_out'][:1]
        raw['held_out'][0]['timestamp_s'] = 2.
        path = tmp_path/f'{case}.yaml'
        path.write_text(yaml.safe_dump(raw))
        captures[case] = path
    cfg = yaml.safe_load((ROOT/'configs/demos/stairs_views.yaml').read_text())
    cfg.update(fixed_capture_config=captures['fixed_head'].name,
               scan_capture_config=captures['head_scan'].name)
    cfg['reconstruction']['resolution'] = 48
    cfg['occupancy'].update(resolution=24, pixel_stride=1)
    cfg['evaluation']['sample_spacing_m'] = .12
    cfg['playback'] = {'fps': 3, 'scale': 1, 'mp4': False}
    path = tmp_path/'demo.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path, captures


def _capture(config_path, output_dir, progress=None, preview=True):
    """Analytic stair rendering with independent camera rotation and masked inputs."""
    assert preview is False
    cfg = load_rig_capture_config(config_path)
    stairs, cam = cfg['stairs'], cfg['camera']
    origin = np.array(stairs['origin_m'])
    width, tread, rise = (stairs[key] for key in ('width_m', 'tread_m', 'rise_m'))
    scene = SyntheticScene.from_config({'objects': [
        {'type': 'box', 'size_m': [width, tread, (i+1)*rise],
         'translation_m': (origin+[-width/2, i*tread, 0]).tolist(),
         'color_rgb': [70+i*15, 140, 200]} for i in range(stairs['steps'])]})
    camera = {'width': cam['width'], 'height': cam['height'],
              'K': intrinsics_from_fovy(cam['width'], cam['height'], cam['fovy_degrees']).tolist(),
              'depth_trunc_m': cam['depth_trunc_m']}
    records, rig_records = [], []
    for split, frames in (('integration', cfg['trajectory']), ('held_out', cfg['held_out'])):
        for frame in frames:
            if split == 'integration':
                rig = rig_transform(frame['position_m'], frame['yaw_degrees'])
                local = head_camera_pose(cam, frame)
                pose = rig@local
                box = rig.copy()
                box[:3, 3] = (rig@np.r_[cfg['box']['position_in_rig_m'], 1.])[:3]
                rig_records.append({'id': frame['id'], 'timestamp_s': frame['timestamp_s'],
                                    'T_wrig': rig.tolist(), 'T_wbox': box.tolist(),
                                    'T_wc': pose.tolist(), 'T_rig_camera': local.tolist()})
            else:
                pose = look_at_pose(frame['eye'], frame['target'], frame['up'])
            rgb, depth = scene.render(np.array(camera['K']), pose, cam['width'], cam['height'])
            depth[(depth <= 0) | ~np.isfinite(depth) | (depth >= cam['depth_trunc_m'])] = 0
            rgb[depth == 0] = 0
            assert (depth > 0).any()
            records.append((frame['id'], split, pose, frame['timestamp_s'], rgb, depth))
    out = Path(output_dir)
    out.mkdir(parents=True)
    for condition in ('unoccluded', 'occluded'):
        writer = SequenceWriter(out/condition, camera, cfg['scene_name'], 'simulator_ground_truth',
                                {'world_up': [0, 0, 1], 'producer': 'analytic_test_capture'})
        for name, split, pose, time, rgb, depth in records:
            raw_rgb, raw_depth = rgb.copy(), depth.copy()
            mask = np.zeros(depth.shape, bool)
            if split == 'integration' and condition == 'occluded':
                mask[:, 28:36] = True
                mask &= depth > 0
                assert mask.any()
                raw_rgb[mask], raw_depth[mask] = [220, 90, 40], .3
            observed_rgb, observed_depth = raw_rgb.copy(), raw_depth.copy()
            observed_rgb[mask], observed_depth[mask] = 0, 0
            # Explicit clean references contain the masked-away geometry.
            writer.add_frame(name, split, observed_rgb, observed_depth, pose, time,
                             reference_rgb=rgb, reference_depth_m=depth)
            if split == 'integration':
                folder = out/condition/'frames'/name
                Image.fromarray(raw_rgb).save(folder/'raw_rgb.png')
                np.save(folder/'raw_depth_m.npy', raw_depth, allow_pickle=False)
                for kind in ('excluded', 'box', 'robot'):
                    selected = np.zeros_like(mask) if kind == 'robot' else mask
                    Image.fromarray(selected.astype(np.uint8)*255).save(folder/f'{kind}_mask.png')
        writer.finish()
    masks = {}
    for name, split, _, _, _, depth in records:
        if split == 'held_out':
            folder = out/'evaluation'/name
            folder.mkdir(parents=True)
            Image.fromarray((depth > 0).astype(np.uint8)*255).save(folder/'stairs_mask.png')
            masks[name] = f'evaluation/{name}/stairs_mask.png'
    xml = build_rig_mjcf(cfg)
    (out/'model_source.xml').write_text(xml)
    (out/'capture_source.yaml').write_text(Path(config_path).read_text())
    result = {'format': 'wtumi-rig-capture-v1', 'scene_name': cfg['scene_name'],
              'stairs': stairs, 'roi': cfg['roi'], 'rig': {key: cfg[key] for key in ('camera', 'torso', 'box')},
              'frames': rig_records,
              'metadata': {'producer': 'analytic_test_capture', 'world_up': [0, 0, 1]},
              'paths': {'unoccluded_manifest': 'unoccluded/manifest.json',
                        'occluded_manifest': 'occluded/manifest.json',
                        'held_out_masks': masks, 'model': 'model_source.xml'}}
    (out/'capture.json').write_text(json.dumps(result))
    return result


@pytest.mark.parametrize('difference', ['body_path', 'calibration', 'held_out', 'same_orientations'])
def test_mismatched_conditions_fail_before_capture(tmp_path, monkeypatch, difference):
    from wtumi_recon import rig_capture
    config, captures = _setup(tmp_path)
    path = captures['head_scan']
    cfg = yaml.safe_load(path.read_text())
    if difference == 'body_path':
        cfg['trajectory'][1]['position_m'][0] += .01
    elif difference == 'calibration':
        cfg['camera']['fovy_degrees'] += 1
    elif difference == 'held_out':
        cfg['held_out'][0]['eye'][0] += .1
    else:
        for frame in cfg['trajectory']:
            frame['head_yaw_degrees'] = frame['head_pitch_degrees'] = 0
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('invalid config reached capture'))
    with pytest.raises(ValueError):
        run_view_demo(config, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_fixed_condition_rejects_nonzero_head_offset(tmp_path):
    config, captures = _setup(tmp_path)
    path = captures['fixed_head']
    cfg = yaml.safe_load(path.read_text())
    cfg['trajectory'][0]['head_pitch_degrees'] = 1
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='fixed_head'):
        load_view_demo_config(config)


def test_existing_output_is_preserved_before_capture(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture
    config, _ = _setup(tmp_path)
    out = tmp_path/'output'
    out.mkdir()
    (out/'keep.txt').write_text('existing data')
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('existing output reached capture'))
    with pytest.raises(FileExistsError):
        run_view_demo(config, out)
    assert list(out.iterdir()) == [out/'keep.txt']
    assert (out/'keep.txt').read_text() == 'existing data'


def _signature(depth, pose):
    return hashlib.sha256(np.asarray(depth, dtype='<f4').tobytes()+
                          np.asarray(pose, dtype='<f8').tobytes()).hexdigest()


def test_each_actual_masked_observation_fused_once_and_heldouts_excluded(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture
    from wtumi_recon.tsdf import Open3DTSDFMap
    from wtumi_recon.occupancy import EvidenceOccupancyGrid
    config, _ = _setup(tmp_path)
    captures, tsdf_calls, occupancy_calls = [], [], []
    original_tsdf, original_occupancy = Open3DTSDFMap.integrate, EvidenceOccupancyGrid.integrate
    def capture_once(*args, **kwargs):
        captures.append(Path(args[1]).name)
        return _capture(*args, **kwargs)
    def tsdf(self, rgbd, intrinsic, pose):
        tsdf_calls.append(_signature(np.asarray(rgbd.depth), pose))
        return original_tsdf(self, rgbd, intrinsic, pose)
    def occupancy(self, depth, k, pose, depth_trunc_m, pixel_stride=1):
        occupancy_calls.append(_signature(depth, pose))
        return original_occupancy(self, depth, k, pose, depth_trunc_m, pixel_stride)
    monkeypatch.setattr(rig_capture, 'capture_rig', capture_once)
    monkeypatch.setattr(Open3DTSDFMap, 'integrate', tsdf)
    monkeypatch.setattr(EvidenceOccupancyGrid, 'integrate', occupancy)
    out = tmp_path/'output'
    result = run_view_demo(config, out)
    assert captures == list(CASES)
    assert result['verification']['equal_body_and_payload_poses']
    expected, hidden_reference_inputs = [], []
    for case in CASES:
        sequence = RGBDSequence(out/'captures'/case/'occluded/manifest.json')
        for view in sequence.views:
            if view['split'] == 'integration':
                expected.append(_signature(sequence.read(view)[1], view['T_wc']))
                hidden_reference_inputs.append(_signature(sequence.read_reference(view)[1], view['T_wc']))
        final = out/'runs'/case
        report = json.loads((final/'metrics.json').read_text())
        assert report['integrated_frame_ids'] == ['approach', 'view_001']
        assert report['newly_integrated_frame_ids'] == ['view_001']
        assert report['held_out_frame_ids'] == ['evaluation_front_left']
        assert (final/'viewer.html').is_file()
        assert (final/'tsdf_volume.npz').is_file()
        prefix = out/'history'/case/'frame_001'
        assert not (prefix/'tsdf_volume.npz').exists()
        assert not (prefix/'inputs/approach').exists()
    assert Counter(tsdf_calls) == Counter(occupancy_calls) == Counter(expected)
    assert len(tsdf_calls) == 4
    assert set(tsdf_calls).isdisjoint(hidden_reference_inputs)
    assert (out/'index.html').is_file()
    assert (out/'visibility_evaluation.json').is_file()


def test_final_replay_survives_result_relocation(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture
    monkeypatch.setattr(rig_capture, 'capture_rig', _capture)
    config, _ = _setup(tmp_path)
    out = tmp_path/'output'
    result = run_view_demo(config, out)
    moved = tmp_path/'relocated'
    shutil.move(out, moved)
    for case in CASES:
        replay_config = moved/result['paths']['replay_configs'][case]
        resolved = load_run_config(replay_config)
        assert Path(resolved['source']['manifest']).is_relative_to(moved)
        replay = run_reconstruction(replay_config, tmp_path/f'replayed_{case}')
        original = json.loads((moved/'runs'/case/'metrics.json').read_text())
        assert replay['summary'] == original['summary']
        assert replay['integrated_frame_ids'] == original['integrated_frame_ids']


def test_captured_payload_mismatch_rejected_before_fusion(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture, view_demo
    config, _ = _setup(tmp_path)
    def invalid_payload(*args, **kwargs):
        result = _capture(*args, **kwargs)
        if Path(args[1]).name == 'head_scan':
            result['frames'][0]['T_wbox'][0][3] += .2
        return result
    monkeypatch.setattr(rig_capture, 'capture_rig', invalid_payload)
    monkeypatch.setattr(view_demo, 'run_reconstruction', lambda *a, **k: pytest.fail('bad payload reached fusion'))
    with pytest.raises(RuntimeError, match='payload'):
        run_view_demo(config, tmp_path/'output')


def test_truncated_pose_metadata_rejected_before_fusion(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture, view_demo
    config, _ = _setup(tmp_path)
    def truncated(*args, **kwargs):
        result = _capture(*args, **kwargs)
        if Path(args[1]).name == 'head_scan':
            result['frames'] = result['frames'][:1]
        return result
    monkeypatch.setattr(rig_capture, 'capture_rig', truncated)
    monkeypatch.setattr(view_demo, 'run_reconstruction', lambda *a, **k: pytest.fail('truncated metadata reached fusion'))
    with pytest.raises(RuntimeError):
        run_view_demo(config, tmp_path/'output')
