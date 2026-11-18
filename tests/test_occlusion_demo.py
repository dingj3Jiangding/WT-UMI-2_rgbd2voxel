"""Real three-condition mapping and replay from a portable paired fake capture."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.occlusion_demo import run_occlusion_demo, verify_paired_capture
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.render import look_at_pose
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.synthetic import SyntheticScene


ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path):
    capture = yaml.safe_load((ROOT/'configs/capture/stairs_rig.yaml').read_text())
    capture['trajectory'] = capture['trajectory'][:2]
    for view, name in zip(capture['trajectory'], ('front', 'side')):
        view['id'] = name
    capture['held_out'] = capture['held_out'][:1]
    capture['held_out'][0].update(id='novel', timestamp_s=2.)
    capture_path = tmp_path/'capture.yaml'
    capture_path.write_text(yaml.safe_dump(capture))
    config = yaml.safe_load((ROOT/'configs/demos/stairs_occlusion.yaml').read_text())
    config['capture_config'] = 'capture.yaml'
    config['reconstruction']['resolution'] = 48
    config['occupancy'].update(resolution=24, pixel_stride=1)
    config['evaluation']['sample_spacing_m'] = .12
    path = tmp_path/'demo.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


def _paired_capture(config_path, output_dir, progress=None):
    """Raycast actual stairs; simulate box pixels before applying the mask."""
    from wtumi_recon.rig_capture import load_rig_capture_config

    cfg = load_rig_capture_config(config_path)
    stairs, roi = cfg['stairs'], cfg['roi']
    origin = np.asarray(stairs['origin_m'])
    width, tread, rise = (stairs[key] for key in ('width_m', 'tread_m', 'rise_m'))
    objects = [{
        'type': 'box', 'size_m': [width, tread, (index+1)*rise],
        'translation_m': (origin + [-width/2, index*tread, 0]).tolist(),
        'color_rgb': [70+index*15, 140, 200],
    } for index in range(stairs['steps'])]
    source = SyntheticScene.from_config({'objects': objects})
    camera = {'width': 64, 'height': 48, 'K': [[55., 0., 31.5], [0., 55., 23.5], [0., 0., 1.]],
              'depth_trunc_m': 5.}
    target = origin + [0, stairs['steps']*tread/2, stairs['steps']*rise/2]
    viewpoints = [
        ('front', 'integration', origin + [-.35, -1., 1.2]),
        ('side', 'integration', origin + [.4, -.9, 1.2]),
        ('novel', 'held_out', origin + [.1, -1.15, 1.35]),
    ]
    pairs = []
    for index, (name, split, eye) in enumerate(viewpoints):
        pose = look_at_pose(eye, target, [0, 0, 1])
        rgb, depth = source.render(np.array(camera['K']), pose, 64, 48)
        depth[(depth <= 0) | ~np.isfinite(depth) | (depth >= 5)] = 0
        rgb[depth == 0] = 0
        assert (depth > 0).any()
        pairs.append((name, split, pose, index, rgb, depth))
    out = Path(output_dir)
    out.mkdir(parents=True)
    for condition in ('unoccluded', 'occluded'):
        writer = SequenceWriter(out/condition, camera, cfg['scene_name'], 'simulator_ground_truth',
                                {'world_up': [0, 0, 1]})
        for name, split, pose, timestamp, rgb, depth in pairs:
            raw_rgb, raw_depth = rgb.copy(), depth.copy()
            mask = np.zeros(depth.shape, dtype=bool)
            if condition == 'occluded' and split == 'integration':
                mask[:, 28:36] = True
                mask &= depth > 0
                assert mask.any()
                raw_depth[mask], raw_rgb[mask] = .3, [220, 90, 40]
            observed_rgb, observed_depth = raw_rgb.copy(), raw_depth.copy()
            observed_rgb[mask], observed_depth[mask] = 0, 0
            writer.add_frame(name, split, observed_rgb, observed_depth, pose, float(timestamp),
                             reference_rgb=rgb, reference_depth_m=depth)
            if split == 'integration':
                directory = out/condition/'frames'/name
                np.save(directory/'raw_depth_m.npy', raw_depth, allow_pickle=False)
                Image.fromarray(raw_rgb).save(directory/'raw_rgb.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(directory/'excluded_mask.png')
        writer.finish()
    (out/'held_out').mkdir()
    Image.fromarray((pairs[-1][-1] > 0).astype(np.uint8)*255).save(out/'held_out/novel_mask.png')
    Image.fromarray(pairs[0][-2]).save(out/'capture_preview.gif')
    result = {'stairs': stairs, 'roi': roi,
              'paths': {'unoccluded_manifest': 'unoccluded/manifest.json',
                        'occluded_manifest': 'occluded/manifest.json',
                        'held_out_masks': {'novel': 'held_out/novel_mask.png'},
                        'preview_gif': 'capture_preview.gif'}}
    (out/'capture.json').write_text(json.dumps(result))
    return result


def test_three_conditions_reuse_masked_frames_and_replay_after_relocation(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture

    config, _ = _setup(tmp_path)
    captures = []

    def capture_once(config_path, output_dir, progress=None):
        captures.append(Path(output_dir))
        return _paired_capture(config_path, output_dir, progress)

    monkeypatch.setattr(rig_capture, 'capture_rig', capture_once)
    out = tmp_path/'output'
    report = run_occlusion_demo(config, out)
    assert captures == [out/'capture']
    assert report['paired_inputs_verified']
    expected = {'unoccluded_multi': ['front', 'side'], 'occluded_single': ['front'],
                'occluded_multi': ['front', 'side']}
    assert {row['case_id']: row['integration_frames'] for row in report['cases']} == {
        key: len(value) for key, value in expected.items()}
    heldout = []
    for case, names in expected.items():
        run = out/'runs'/case
        metrics = json.loads((run/'metrics.json').read_text())
        assert metrics['integrated_frame_ids'] == metrics['occupancy']['integrated_frame_ids'] == names
        assert metrics['held_out_frame_ids'] == ['novel']
        stair_metrics = json.loads((run/'stair_metrics.json').read_text())
        assert stair_metrics['held_out_depth']['pooled']['reference_valid_pixels'] > 0
        assert stair_metrics['surface_completeness']['sample_count'] > 0
        assert stair_metrics['occupancy_roi']['roi_in_volume_cell_count'] > 0
        heldout.append(np.load(run/'references/novel/depth_m.npy'))
    for depth in heldout[1:]:
        np.testing.assert_array_equal(depth, heldout[0])

    full = RGBDSequence(out/'capture/occluded/manifest.json')
    prefix = RGBDSequence(out/'capture/occluded/manifest_prefix_001.json')
    assert [v['id'] for v in prefix.views if v['split'] == 'integration'] == ['front']
    original_record = next(frame for frame in full.manifest['frames'] if frame['id'] == 'front')
    assert prefix.manifest['frames'][0] == original_record
    _, first = prefix.read('front')
    mask = np.asarray(Image.open(prefix.root/'frames/front/excluded_mask.png')) > 0
    assert mask.any()
    np.testing.assert_array_equal(first[mask], 0)
    raw = np.load(prefix.root/'frames/front/raw_depth_m.npy')
    assert (raw[mask] > 0).all()
    for case in ('occluded_single', 'occluded_multi'):
        np.testing.assert_array_equal(np.load(out/'runs'/case/'inputs/front/depth_m.npy'), first)
    assert (out/report['paths']['viewer']).is_file()
    assert (out/report['paths']['occupancy_animation']).is_file()
    assert (out/report['paths']['condition_comparison']).is_file()

    relocated = tmp_path/'relocated'
    shutil.copytree(out, relocated)
    (out/'capture').rename(out/'capture_unavailable')
    replay_path = relocated/report['paths']['replay_configs']['occluded_single']
    resolved = load_run_config(replay_path)
    assert Path(resolved['source']['manifest']) == relocated/'capture/occluded/manifest_prefix_001.json'
    replay = run_reconstruction(replay_path, tmp_path/'replayed')
    assert replay['integrated_frame_ids'] == ['front']
    original = json.loads((out/'runs/occluded_single/metrics.json').read_text())
    assert replay['summary'] == original['summary']
    assert captures == [out/'capture']


def test_pair_verification_checks_evaluation_references_and_exclusion(tmp_path):
    config, _ = _setup(tmp_path)
    capture = _paired_capture(config.parent/'capture.yaml', tmp_path/'capture')
    clear = tmp_path/'capture'/capture['paths']['unoccluded_manifest']
    blocked = tmp_path/'capture'/capture['paths']['occluded_manifest']
    verify_paired_capture(clear, blocked)
    sequence = RGBDSequence(blocked)
    heldout = next(frame for frame in sequence.manifest['frames'] if frame['split'] == 'held_out')
    path = blocked.parent/heldout['reference_depth_m']
    original = np.load(path)
    changed = original.copy()
    changed[changed > 0] += .2
    np.save(path, changed)
    with pytest.raises(ValueError, match='held-out references'):
        verify_paired_capture(clear, blocked)
    np.save(path, original)
    first = next(frame for frame in sequence.manifest['frames'] if frame['split'] == 'integration')
    path = blocked.parent/first['depth_m']
    depth = np.load(path)
    mask = np.asarray(Image.open(blocked.parent/'frames'/first['id']/'excluded_mask.png')) > 0
    depth[mask] = .3
    np.save(path, depth)
    with pytest.raises(ValueError, match='excluded self/box pixels'):
        verify_paired_capture(clear, blocked)


@pytest.mark.parametrize('change', [
    {'reconstruction': {'backend': 'uniform', 'resolution': True, 'volume_length_m': 3.5,
                        'origin_m': [-1.75, -.5, -.15], 'sdf_trunc_m': .08}},
    {'occupancy': {'enabled': False}},
    {'occupancy': {'enabled': True, 'resolution': 2}},
    {'evaluation': {'depth_tolerance_m': float('nan'), 'sample_spacing_m': .1}},
])
def test_invalid_demo_does_not_capture_or_create_output(tmp_path, monkeypatch, change):
    from wtumi_recon import rig_capture

    path, config = _setup(tmp_path)
    config.update(copy.deepcopy(change))
    path.write_text(yaml.safe_dump(config))

    def forbidden(*args, **kwargs):
        pytest.fail('invalid configuration reached capture')

    monkeypatch.setattr(rig_capture, 'capture_rig', forbidden)
    with pytest.raises(ValueError):
        run_occlusion_demo(path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_nonempty_output_is_preserved_before_capture(tmp_path, monkeypatch):
    from wtumi_recon import rig_capture

    path, _ = _setup(tmp_path)
    out = tmp_path/'output'
    out.mkdir()
    (out/'keep.txt').write_text('existing data')

    def forbidden(*args, **kwargs):
        pytest.fail('nonempty output reached capture')

    monkeypatch.setattr(rig_capture, 'capture_rig', forbidden)
    with pytest.raises(FileExistsError):
        run_occlusion_demo(path, out)
    assert (out/'keep.txt').read_text() == 'existing data'
    assert [path.name for path in out.iterdir()] == ['keep.txt']
