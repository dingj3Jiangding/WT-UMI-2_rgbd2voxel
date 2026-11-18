"""True capture resampling and portable 30 fps suite orchestration."""

import copy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from wtumi_recon.scene_bundle import load_scene_config
from wtumi_recon.scene_presets import resample_scene_config
from wtumi_recon.scene_suite import SCENES, run_scene_suite, suite_preset_plan


@pytest.mark.parametrize('name', SCENES)
def test_pace30_samples_fresh_poses_and_retains_calibration_map_and_scene(name):
    original = load_scene_config(name)
    before = copy.deepcopy(original)
    result = resample_scene_config(original)
    assert original == before
    assert len(result['trajectory']) == 121
    assert len({frame['id'] for frame in result['trajectory']}) == 121
    times = np.array([frame['timestamp_s'] for frame in result['trajectory']])
    np.testing.assert_allclose(times, np.arange(121) / 30, atol=1e-12)
    assert result['playback']['fps'] == 30
    assert (result['camera']['width'], result['camera']['height']) == (640, 480)
    assert result['occupancy']['pixel_stride'] == 8
    for key in ('reconstruction', 'robot', 'box', 'torso', 'environment', 'held_out'):
        assert result[key] == before[key]
    for key in ('position_in_rig_m', 'target_in_rig_m', 'fovy_degrees', 'depth_trunc_m'):
        assert result['camera'][key] == before['camera'][key]
    assert min(frame['timestamp_s'] for frame in result['held_out']) > times[-1]
    assert not {frame['id'] for frame in result['held_out']} & {frame['id'] for frame in result['trajectory']}
    for source in original['trajectory']:
        frame = result['trajectory'][int(round(source['timestamp_s'] * 30))]
        np.testing.assert_allclose(frame['position_m'], source['position_m'], atol=2e-9)
        angle_difference = (frame['yaw_degrees'] - source['yaw_degrees'] + 180) % 360 - 180
        assert abs(angle_difference) < 2e-7


def test_body_yaw_crosses_wrap_by_short_arc_while_position_and_head_angles_interpolate():
    config = load_scene_config('g1_stairs')
    config['trajectory'] = [
        {'id': 'first', 'timestamp_s': 0., 'position_m': [0., 0., 0.],
         'yaw_degrees': 179., 'head_yaw_degrees': -10., 'head_pitch_degrees': 4.},
        {'id': 'last', 'timestamp_s': 1., 'position_m': [2., 4., 0.],
         'yaw_degrees': -179., 'head_yaw_degrees': 10., 'head_pitch_degrees': -4.},
    ]
    frames = resample_scene_config(config, fps=2)['trajectory']
    assert [frame['yaw_degrees'] for frame in frames] == pytest.approx([179., 180., 181.])
    assert frames[1]['position_m'] == [1., 2., 0.]
    assert frames[1]['head_yaw_degrees'] == 0
    assert frames[1]['head_pitch_degrees'] == 0


@pytest.mark.parametrize('options,match', [
    ({'fps': True}, 'fps'), ({'fps': 0}, 'fps'), ({'fps': float('nan')}, 'fps'),
    ({'fps': 120}, '240 views'), ({'fps': 30.1}, 'integer number'),
    ({'width': True}, 'width'), ({'height': 4096}, 'height'),
])
def test_invalid_capture_schedule_is_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        resample_scene_config(load_scene_config('room'), **options)


@pytest.mark.parametrize('case', ['one_pose', 'duplicate_time', 'evaluation_overlap', 'id_collision'])
def test_invalid_source_schedule_is_rejected(case):
    config = load_scene_config('room')
    if case == 'one_pose':
        config['trajectory'] = config['trajectory'][:1]
    elif case == 'duplicate_time':
        config['trajectory'][1]['timestamp_s'] = 0
    elif case == 'evaluation_overlap':
        config['held_out'][0]['timestamp_s'] = 4
    else:
        config['held_out'][0]['id'] = 'capture_0000'
    with pytest.raises(ValueError):
        resample_scene_config(config)


def test_preset_plan_needs_neither_graphics_nor_encoder_and_preserves_sources(monkeypatch):
    from wtumi_recon import video

    monkeypatch.setattr(video, 'require_video_encoder', lambda: pytest.fail('plan must not check encoder'))
    sources = {Path(load_scene_config(name)['_path']): Path(load_scene_config(name)['_path']).read_bytes()
               for name in SCENES}
    plan = suite_preset_plan('pace30')
    assert plan['requires_mp4_encoder'] is True
    assert all(row['integration_frames'] == 121 and row['playback_fps'] == 30 for row in plan['scenes'])
    assert all(row['box_enabled'] and row['robot_pose'] == 'carry' for row in plan['scenes'])
    assert all(row['video_duration_s'] == pytest.approx(121 / 30) for row in plan['scenes'])
    assert [row['voxel_size_mm'] for row in plan['scenes']] == pytest.approx([40.625, 40.625, 21.875])
    assert all(path.read_bytes() == data for path, data in sources.items())
    default = suite_preset_plan()
    assert all(row['integration_frames'] == 13 and row['playback_fps'] == 3 for row in default['scenes'])
    with pytest.raises(ValueError, match='Unknown'):
        suite_preset_plan('missing')


def test_pace30_requires_encoder_before_creating_output_or_running_capture(tmp_path, monkeypatch):
    from wtumi_recon import scene_pipeline, video

    def no_encoder():
        raise ImportError('missing MP4 encoder')

    monkeypatch.setattr(video, 'require_video_encoder', no_encoder)
    monkeypatch.setattr(scene_pipeline, 'run_scene', lambda *_args, **_kwargs: pytest.fail('must preflight'))
    destination = tmp_path / 'suite'
    with pytest.raises(ImportError, match='MP4'):
        run_scene_suite(destination, preset='pace30')
    assert not destination.exists()


def test_pace30_materializes_all_portable_inputs_before_capture_and_keeps_originals(tmp_path, monkeypatch):
    from wtumi_recon import scene_pipeline, scene_suite, video

    destination = tmp_path / 'suite'
    sources = {name: Path(load_scene_config(name)['_path']).read_bytes() for name in SCENES}
    events = []
    monkeypatch.setattr(video, 'require_video_encoder', lambda: events.append('encoder'))

    def run(source, output, progress=None):
        assert all((destination / 'inputs' / name / 'scene.yaml').is_file() for name in SCENES)
        config = load_scene_config(source)
        assert len(config['trajectory']) == 121
        assert config['playback']['fps'] == 30
        assert (source / config['robot']['asset_dir'] / 'g1.xml').is_file()
        assert output == destination / source.name
        assert 'inputs' in source.parts
        events.append(source.name)
        return {'paths': {'camera_voxel_mp4': 'camera_voxel.mp4'}}

    monkeypatch.setattr(scene_pipeline, 'run_scene', run)
    monkeypatch.setattr(scene_suite, 'write_scene_suite_report', lambda root: {'paths': {'gallery': 'index.html'}})
    report = run_scene_suite(destination, preset='pace30')
    assert events == ['encoder', *SCENES]
    assert report['preset'] == 'pace30'
    assert report['paths']['preset'] == 'preset.json'
    assert json.loads((destination / 'preset.json').read_text())['preset'] == 'pace30'
    assert all(Path(load_scene_config(name)['_path']).read_bytes() == data for name, data in sources.items())
    moved = tmp_path / 'moved'
    destination.rename(moved)
    for name in SCENES:
        assert len(load_scene_config(moved / 'inputs' / name)['trajectory']) == 121


def test_pace30_rejects_missing_video_after_capture(tmp_path, monkeypatch):
    from wtumi_recon import scene_pipeline, video

    monkeypatch.setattr(video, 'require_video_encoder', lambda: None)
    monkeypatch.setattr(scene_pipeline, 'run_scene', lambda *_args, **_kwargs: {'paths': {}})
    with pytest.raises(RuntimeError, match='requires a completed MP4'):
        run_scene_suite(tmp_path / 'suite', preset='pace30')


def test_cli_dry_run_creates_no_directory_and_never_runs_suite(tmp_path, monkeypatch, capsys):
    from wtumi_recon import scene_suite

    monkeypatch.setattr(scene_suite, 'run_scene_suite', lambda *_args, **_kwargs: pytest.fail('dry run must not execute'))
    script = Path(__file__).resolve().parents[1] / 'scripts/run_scene_suite.py'
    output = tmp_path / 'planned_output'
    monkeypatch.setattr(sys, 'argv', [str(script), '--preset', 'pace30', '--dry-run', '--output', str(output)])
    runpy.run_path(str(script), run_name='__main__')
    plan = json.loads(capsys.readouterr().out)
    assert plan['preset'] == 'pace30'
    assert plan['output'] == str(output.resolve())
    assert len(plan['scenes']) == 3
    assert not output.exists()
