"""End-to-end observation history, source preservation and portable replay."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.memory_demo import run_memory_demo
from wtumi_recon.mujoco_capture import intrinsics_from_fovy
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.render import look_at_pose
from wtumi_recon.rig_capture import build_rig_mjcf, load_rig_capture_config, rig_transform
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.synthetic import SyntheticScene


ROOT = Path(__file__).resolve().parents[1]
CASES = ('always_occluded', 'observe_then_occluded')


def _setup(tmp_path):
    capture = yaml.safe_load((ROOT/'configs/capture/stairs_rig.yaml').read_text())
    capture['camera'].update(width=64, height=48)
    capture['trajectory'] = capture['trajectory'][:2]
    capture['held_out'] = capture['held_out'][:1]
    capture['held_out'][0]['timestamp_s'] = 2.
    capture_path = tmp_path/'capture.yaml'
    capture_path.write_text(yaml.safe_dump(capture))
    cfg = yaml.safe_load((ROOT/'configs/demos/stairs_occlusion.yaml').read_text())
    cfg.update(capture_config='capture.yaml', preobserve_frames=1)
    cfg['reconstruction']['resolution'] = 48
    cfg['occupancy'].update(resolution=24, pixel_stride=1)
    cfg['evaluation']['sample_spacing_m'] = .12
    path = tmp_path/'memory.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def _capture(config_path, output_dir, progress=None):
    """Render actual stair geometry at configured poses without a GL backend."""
    cfg = load_rig_capture_config(config_path)
    stairs = cfg['stairs']
    origin = np.array(stairs['origin_m'])
    width, tread, rise = (stairs[k] for k in ('width_m', 'tread_m', 'rise_m'))
    source = SyntheticScene.from_config({'objects': [
        {'type': 'box', 'size_m': [width, tread, (i+1)*rise],
         'translation_m': (origin+[-width/2, i*tread, 0]).tolist(),
         'color_rgb': [70+i*15, 140, 200]}
        for i in range(stairs['steps'])
    ]})
    camera_cfg = cfg['camera']
    camera = {'width': camera_cfg['width'], 'height': camera_cfg['height'],
              'K': intrinsics_from_fovy(camera_cfg['width'], camera_cfg['height'],
                                        camera_cfg['fovy_degrees']).tolist(),
              'depth_trunc_m': camera_cfg['depth_trunc_m']}
    local = look_at_pose(camera_cfg['position_in_rig_m'], camera_cfg['target_in_rig_m'], [0, 0, 1])
    records = []
    for split, items in (('integration', cfg['trajectory']), ('held_out', cfg['held_out'])):
        for frame in items:
            pose = (rig_transform(frame['position_m'], frame['yaw_degrees'])@local
                    if split == 'integration' else look_at_pose(frame['eye'], frame['target'], frame['up']))
            rgb, depth = source.render(np.array(camera['K']), pose, camera['width'], camera['height'])
            depth[(depth <= 0) | ~np.isfinite(depth) | (depth >= camera['depth_trunc_m'])] = 0
            rgb[depth == 0] = 0
            assert np.any(depth > 0)
            records.append((frame['id'], split, pose, frame['timestamp_s'], rgb, depth))
    out = Path(output_dir)
    out.mkdir(parents=True)
    for condition in ('unoccluded', 'occluded'):
        writer = SequenceWriter(out/condition, camera, cfg['scene_name'], 'simulator_ground_truth',
                                {'world_up': [0, 0, 1], 'producer': 'analytic_test_capture'})
        for name, split, pose, timestamp, rgb, depth in records:
            raw_rgb, raw_depth = rgb.copy(), depth.copy()
            mask = np.zeros(depth.shape, dtype=bool)
            if condition == 'occluded' and split == 'integration':
                mask[:, 28:36] = True
                mask &= depth > 0
                assert mask.any()
                raw_rgb[mask], raw_depth[mask] = [220, 90, 40], .3
            observed_rgb, observed_depth = raw_rgb.copy(), raw_depth.copy()
            observed_rgb[mask], observed_depth[mask] = 0, 0
            # Clean integration references intentionally differ from masked input.
            writer.add_frame(name, split, observed_rgb, observed_depth, pose, timestamp,
                             reference_rgb=rgb, reference_depth_m=depth)
            if split == 'integration':
                directory = out/condition/'frames'/name
                Image.fromarray(raw_rgb).save(directory/'raw_rgb.png')
                np.save(directory/'raw_depth_m.npy', raw_depth, allow_pickle=False)
                Image.fromarray(mask.astype(np.uint8)*255).save(directory/'excluded_mask.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(directory/'box_mask.png')
                Image.fromarray(np.zeros(depth.shape, np.uint8)).save(directory/'robot_mask.png')
        writer.finish()
    held_masks = {}
    for name, split, _, _, _, depth in records:
        if split != 'held_out':
            continue
        directory = out/'evaluation'/name
        directory.mkdir(parents=True)
        mask_path = directory/'stairs_mask.png'
        Image.fromarray((depth > 0).astype(np.uint8)*255).save(mask_path)
        held_masks[name] = str(mask_path.relative_to(out))
    Image.fromarray(records[0][-2]).save(out/'capture_preview.gif')
    (out/'capture_source.yaml').write_text(Path(config_path).read_text())
    xml = build_rig_mjcf(cfg)
    (out/'model_source.xml').write_text(xml)
    result = {'format': 'wtumi-rig-capture-v1', 'scene_name': cfg['scene_name'],
              'stairs': stairs, 'roi': cfg['roi'],
              'rig': {k: cfg[k] for k in ('camera', 'torso', 'box')},
              'integration_frame_ids': [r[0] for r in records if r[1]=='integration'],
              'held_out_frame_ids': [r[0] for r in records if r[1]=='held_out'],
              'metadata': {'world_up': [0, 0, 1], 'producer': 'analytic_test_capture',
                           'generated_mjcf_sha256': hashlib.sha256(xml.encode()).hexdigest()},
              'paths': {'unoccluded_manifest': 'unoccluded/manifest.json',
                        'occluded_manifest': 'occluded/manifest.json',
                        'held_out_masks': held_masks, 'preview_gif': 'capture_preview.gif',
                        'model': 'model_source.xml'}}
    (out/'capture.json').write_text(json.dumps(result))
    return result


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('preobserve_frames', [True, 0, 2, 3, 1.5])
def test_invalid_preobservation_budget_does_not_create_output(tmp_path, monkeypatch, preobserve_frames):
    from wtumi_recon import rig_capture

    config, cfg = _setup(tmp_path)
    cfg['preobserve_frames'] = preobserve_frames
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('invalid config reached capture'))
    with pytest.raises(ValueError):
        run_memory_demo(config, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_nonempty_output_is_preserved(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture

    config, _ = _setup(tmp_path)
    out = tmp_path/'output'
    out.mkdir()
    (out/'keep.txt').write_text('existing data')
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('nonempty output reached capture'))
    with pytest.raises(FileExistsError):
        run_memory_demo(config, out)
    assert _hashes(out) == {'keep.txt': hashlib.sha256(b'existing data').hexdigest()}


def test_default_capture_runs_once_for_equal_frame_budget(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture
    from wtumi_recon.tsdf import Open3DTSDFMap

    config, _ = _setup(tmp_path)
    calls = []
    integrated_depths = []
    integrate = Open3DTSDFMap.integrate

    def record_depth(self, rgbd, intrinsic, pose):
        integrated_depths.append(np.asarray(rgbd.depth).copy())
        return integrate(self, rgbd, intrinsic, pose)

    def capture_once(config_path, output_dir, progress=None):
        calls.append(Path(output_dir))
        return _capture(config_path, output_dir, progress)

    monkeypatch.setattr(rig_capture, 'capture_rig', capture_once)
    monkeypatch.setattr(Open3DTSDFMap, 'integrate', record_depth)
    out = tmp_path/'output'
    report = run_memory_demo(config, out)
    assert calls == [out/'capture']
    assert report['paired_inputs_verified']
    assert {row['case_id']: row['integration_frames'] for row in report['cases']} == dict.fromkeys(CASES, 2)
    heldout = []
    for case in CASES:
        final = out/'runs'/case
        metrics = json.loads((final/'metrics.json').read_text())
        assert metrics['integrated_frame_ids'] == metrics['occupancy']['integrated_frame_ids'] == ['approach', 'near']
        assert metrics['held_out_frame_ids'] == ['evaluation_front_left']
        prefix = out/'history'/case/'frame_001'
        prefix_metrics = json.loads((prefix/'metrics.json').read_text())
        assert prefix_metrics['integrated_frame_ids'] == prefix_metrics['occupancy']['integrated_frame_ids'] == ['approach']
        assert not (prefix/'inputs/near').exists()
        manifest = json.loads((final/'manifest.json').read_text())
        heldout_record = next(row for row in manifest['frames'] if row['id']=='evaluation_front_left')
        assert heldout_record['input_role'] == 'evaluation_reference_only'
        assert (final/'viewer.html').is_file()
        heldout.append(np.load(final/'references/evaluation_front_left/depth_m.npy'))
    np.testing.assert_array_equal(*heldout)
    blocked = RGBDSequence(out/'capture/occluded/manifest.json')
    clear = RGBDSequence(out/'capture/unoccluded/manifest.json')
    for name in ('approach', 'near'):
        expected = {'always_occluded': blocked.read(name)[1],
                    'observe_then_occluded': (clear if name=='approach' else blocked).read(name)[1]}
        for case in CASES:
            np.testing.assert_array_equal(np.load(out/'runs'/case/'inputs'/name/'depth_m.npy'), expected[case])
    mask = np.asarray(Image.open(blocked.root/'frames/approach/box_mask.png')) > 0
    assert np.all(np.load(out/'runs/always_occluded/inputs/approach/depth_m.npy')[mask] == 0)
    assert np.all(np.load(out/'runs/observe_then_occluded/inputs/approach/depth_m.npy')[mask] > 0)
    # Two cases, each with a one-frame and a two-frame reconstruction. All six
    # calls must use the scheduled observations, never explicit clean references.
    expected_depths = [blocked.read('approach')[1], blocked.read('approach')[1], blocked.read('near')[1],
                       clear.read('approach')[1], clear.read('approach')[1], blocked.read('near')[1]]
    digest = lambda depth: hashlib.sha256(depth.astype('<f4').tobytes()).hexdigest()
    assert Counter(map(digest, integrated_depths)) == Counter(map(digest, expected_depths))


def test_saved_capture_is_unchanged_and_replay_survives_relocation(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture

    config, _ = _setup(tmp_path)
    source = tmp_path/'saved_capture'
    _capture(config.parent/'capture.yaml', source)
    before = _hashes(source)
    monkeypatch.setattr(rig_capture, 'capture_rig', lambda *a, **k: pytest.fail('saved capture launched simulator'))
    out = tmp_path/'output'
    report = run_memory_demo(config, out, capture_dir=source)
    assert _hashes(source) == before
    relocated = tmp_path/'relocated'
    shutil.copytree(out, relocated)
    source.rename(tmp_path/'original_capture_hidden')
    out.rename(tmp_path/'original_result_hidden')
    for case in CASES:
        replay_path = relocated/report['paths']['replay_configs'][case]
        cfg = load_run_config(replay_path)
        assert Path(cfg['source']['manifest']).is_relative_to(relocated)
        replay = run_reconstruction(replay_path, tmp_path/f'replay_{case}')
        previous = json.loads((relocated/'runs'/case/'metrics.json').read_text())
        assert replay['summary'] == previous['summary']
        assert replay['integrated_frame_ids'] == ['approach', 'near']


def test_output_inside_saved_capture_is_rejected_without_mutation(tmp_path):
    config, _ = _setup(tmp_path)
    source = tmp_path/'saved_capture'
    _capture(config.parent/'capture.yaml', source)
    before = _hashes(source)
    with pytest.raises(ValueError):
        run_memory_demo(config, source/'nested_result', capture_dir=source)
    assert _hashes(source) == before


@pytest.mark.parametrize('incompatibility', ['stair_geometry', 'camera_calibration', 'paired_pose_schedule'])
def test_saved_capture_mismatch_is_rejected_before_output(tmp_path, incompatibility):
    config, _ = _setup(tmp_path)
    source = tmp_path/'saved_capture'
    _capture(config.parent/'capture.yaml', source)
    if incompatibility == 'stair_geometry':
        path = source/'capture.json'
        capture = json.loads(path.read_text())
        capture['stairs']['width_m'] += .2
        path.write_text(json.dumps(capture))
    else:
        for condition in ('unoccluded', 'occluded'):
            path = source/condition/'manifest.json'
            manifest = json.loads(path.read_text())
            if incompatibility == 'camera_calibration':
                manifest['camera']['K'][0][0] *= 1.1
            else:
                manifest['frames'][0]['T_wc'][0][3] += .05
            path.write_text(json.dumps(manifest))
    before = _hashes(source)
    with pytest.raises(ValueError):
        run_memory_demo(config, tmp_path/'output', capture_dir=source)
    assert not (tmp_path/'output').exists()
    assert _hashes(source) == before
