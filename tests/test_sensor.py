"""Noise reproducibility, held-out isolation and independent metric checks."""

import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.evaluation import evaluate_rgbd, summarize_views
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.sensor import corrupt_depth, normalize_sensor_config
from wtumi_recon.synthetic import SyntheticScene
from wtumi_recon.tsdf import Open3DTSDFMap

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('raw', [
    {'depth_noise_std_m': -1}, {'depth_noise_std_m': float('nan')},
    {'depth_noise_std_m': True}, {'dropout_probability': 1.1},
    {'seed': -1}, {'seed': 1.5}, {'seed': True}, {'depth_noise_std_mm': 5}, [],
])
def test_invalid_sensor_configuration_is_rejected(raw):
    with pytest.raises(ValueError):
        normalize_sensor_config(raw)


def test_clean_depth_is_an_independent_copy_and_invalid_stays_invalid():
    source = np.array([[0, 1, 2], [np.nan, np.inf, 5]], dtype=np.float32)
    original = source.copy()
    result = corrupt_depth(source, {}, 'front', 5)
    np.testing.assert_equal(source, original)
    np.testing.assert_array_equal(result, [[0, 1, 2], [0, 0, 0]])
    assert result.dtype == np.float32 and not np.shares_memory(result, source)
    np.testing.assert_array_equal(corrupt_depth(source, {'dropout_probability': 1}, 'front', 5), 0)


def test_noise_statistics_reproducibility_and_independent_dropout():
    source = np.ones((300, 400), dtype=np.float32)
    source[:, :10] = 0
    cfg = {'depth_noise_std_m': .005, 'seed': 7}
    noisy = corrupt_depth(source, cfg, 'front', 5)
    np.testing.assert_array_equal(noisy, corrupt_depth(source, cfg, 'front', 5))
    assert not np.array_equal(noisy, corrupt_depth(source, cfg, 'left', 5))
    delta = noisy[source > 0] - 1
    assert abs(delta.mean()) < .0001
    assert delta.std() == pytest.approx(.005, rel=.02)
    mixed = corrupt_depth(source, {**cfg, 'dropout_probability': .1}, 'front', 5)
    observed = mixed > 0
    np.testing.assert_array_equal(mixed[observed], noisy[observed])
    np.testing.assert_array_equal(mixed[:, :10], 0)
    missing_fraction = ((source > 0) & ~observed).sum() / (source > 0).sum()
    assert missing_fraction == pytest.approx(.1, abs=.004)
    np.testing.assert_array_equal(source[:, 10:], 1)


def test_actual_fusion_inputs_match_saved_noisy_data_and_references_stay_clean(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / 'configs/scenes/multiview_cube.yaml').read_text())
    cfg['camera'].update(width=100, height=80, fx=90, fy=90, cx=49.5, cy=39.5)
    cfg['sensor'] = {'depth_noise_std_m': .005, 'dropout_probability': .1, 'seed': 7}
    actual_inputs = []
    integrate = Open3DTSDFMap.integrate

    def track(self, rgbd, intrinsic, pose):
        actual_inputs.append((np.asarray(rgbd.depth).copy(), pose.copy()))
        return integrate(self, rgbd, intrinsic, pose)

    monkeypatch.setattr(Open3DTSDFMap, 'integrate', track)
    manifests = []
    for resolution in (80, 96):
        cfg['reconstruction']['resolution'] = resolution
        config = tmp_path / f'r{resolution}.yaml'
        config.write_text(yaml.safe_dump(cfg))
        out = tmp_path / f'r{resolution}'
        actual_inputs.clear()
        report = run_reconstruction(config, out)
        resolved = load_run_config(config)
        scene = SyntheticScene.from_config(resolved['scene'])
        fused = [v for v in resolved['views'] if v['split'] == 'integration']
        assert len(actual_inputs) == len(fused)
        for (actual, pose), view in zip(actual_inputs, fused):
            saved = np.load(out / 'inputs' / view['id'] / 'depth_m.npy')
            np.testing.assert_array_equal(actual, saved)
            np.testing.assert_allclose(pose, view['T_wc'])
            reference = np.load(out / 'references' / view['id'] / 'depth_m.npy')
            assert (saved > 0).sum() < (reference > 0).sum()
        for view in resolved['views']:
            expected_rgb, expected = scene.render(np.array(resolved['camera']['K']), np.array(view['T_wc']), 100, 80)
            expected[(expected <= 0) | (expected >= resolved['camera']['depth_trunc_m']) | ~np.isfinite(expected)] = 0
            reference = np.load(out / 'references' / view['id'] / 'depth_m.npy')
            np.testing.assert_array_equal(reference, expected)
            if view['split'] == 'held_out':
                np.testing.assert_array_equal(np.load(out / 'inputs' / view['id'] / 'depth_m.npy'), reference)
                assert not any(np.allclose(pose, view['T_wc']) for _, pose in actual_inputs)
            row = next(r for r in report['views'] if r['id'] == view['id'])
            predicted = np.load(out / 'renders' / view['id'] / 'depth_m.npy')
            rgb = np.asarray(o3d.io.read_image(str(out / 'references' / view['id'] / 'rgb.png')))
            pred_rgb = np.asarray(o3d.io.read_image(str(out / 'renders' / view['id'] / 'rgb.png')))
            independently_scored = evaluate_rgbd(rgb, reference, pred_rgb, predicted)
            for key in ('reference_valid_pixels', 'accurate_reference_pixels', 'mae_m', 'coverage'):
                assert row[key] == independently_scored[key]
        assert report['evaluation_reference'] == 'clean_synthetic_ground_truth'
        manifests.append(out)
    # Changing the reconstruction grid must not change the synthetic sensor data.
    for view in resolved['views']:
        for folder in ('inputs', 'references'):
            a, b = [np.load(out / folder / view['id'] / 'depth_m.npy') for out in manifests]
            np.testing.assert_array_equal(a, b)


def test_inlier_pooling_uses_all_reference_pixels_and_requires_same_tolerance():
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    first = evaluate_rgbd(rgb, [[1, 1, 1]], rgb, [[1, 0, 0]], .02)
    second = evaluate_rgbd(rgb, [[1, 0, 0]], rgb, [[1.1, 1, 0]], .02)
    total = summarize_views([first, second])
    assert total['coverage'] == .5
    assert total['accurate_reference_pixels'] == 1
    assert total['accurate_reference_fraction'] == .25
    assert total['silhouette_iou'] == pytest.approx(2 / 5)
    second['depth_tolerance_m'] = .1
    with pytest.raises(ValueError, match='different tolerances'):
        summarize_views([first, second])


def test_bad_sensor_fails_before_creating_output(tmp_path):
    cfg = yaml.safe_load((ROOT / 'configs/scenes/multiview_cube.yaml').read_text())
    cfg['sensor'] = {'dropout_probability': 2}
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        run_reconstruction(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
