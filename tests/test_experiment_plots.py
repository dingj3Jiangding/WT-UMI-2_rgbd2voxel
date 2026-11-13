"""Physical-axis, missing-data and headless plot export checks."""

import json

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.experiment_plots import prepare_plot_series, save_experiment_plot


def _summary():
    return {"scene": "tiny_room", "runs": [
        {"profile":"noisy", "seed":7, "voxel_size_m":.03, "mae_m":.004, "rmse_m":.006, "coverage":.7},
        {"profile":"noisy", "seed":7, "voxel_size_m":.01, "mae_m":.001, "rmse_m":.002, "coverage":.9},
        {"profile":"noisy", "seed":19, "voxel_size_m":.01, "mae_m":.003, "rmse_m":.004, "coverage":.8},
        {"profile":"noisy", "seed":7, "voxel_size_m":.02, "mae_m":None, "rmse_m":None, "coverage":0},
        {"profile":"clean", "seed":7, "voxel_size_m":.01, "mae_m":0, "rmse_m":0, "coverage":1},
    ]}


def test_series_sort_physical_size_and_keep_null_gaps():
    series = prepare_plot_series(_summary())
    noisy = series[0]
    assert noisy['profile'] == 'noisy'
    np.testing.assert_allclose(noisy['voxel_size_mm'], [10,20,30])
    np.testing.assert_allclose(noisy['metrics']['mae_m']['mean'], [2,np.nan,4], equal_nan=True)
    np.testing.assert_allclose(noisy['metrics']['rmse_m']['mean'], [3,np.nan,6], equal_nan=True)
    np.testing.assert_allclose(noisy['metrics']['coverage']['mean'], [85,0,70])
    assert noisy['metrics']['mae_m']['valid_runs'] == [2,0,1]
    assert noisy['metrics']['mae_m']['individual_x_mm'] == [10,10,30]
    assert noisy['metrics']['mae_m']['individual_y'] == [1,3,4]
    assert noisy['seeds'] == [7,19]
    assert noisy['runs_per_size'] == [2,1,1]
    assert series[1]['metrics']['mae_m']['mean'][0] == 0  # True zero remains valid.


@pytest.mark.parametrize('key,value', [('voxel_size_m',0),('mae_m',float('nan')),('coverage',1.1),('seed',True)])
def test_malformed_values_are_rejected(key,value):
    data=_summary()
    data['runs'][0][key]=value
    with pytest.raises(ValueError):
        prepare_plot_series(data)


def test_duplicate_seed_at_one_size_is_not_double_counted():
    data=_summary()
    data['runs'].append(data['runs'][0].copy())
    with pytest.raises(ValueError,match='duplicate'):
        prepare_plot_series(data)


def test_headless_export_preserves_input_and_creates_both_formats(tmp_path):
    source=tmp_path/'summary.json'
    source.write_text(json.dumps(_summary()))
    original=source.read_bytes()
    png,svg=save_experiment_plot(source,tmp_path/'plots'/'quality.svg')
    assert png.name=='quality.png' and svg.name=='quality.svg'
    assert source.read_bytes()==original
    with Image.open(png) as image:
        assert image.width > 1000 and image.height > 500
        assert np.asarray(image).std() > 1
    content=svg.read_text()
    assert '<svg' in content
    assert 'Voxel edge length (mm)' in content
    assert 'Missing values stay gaps' in content
    default_png,default_svg=save_experiment_plot(source)
    assert default_png == tmp_path/'error_vs_voxel_size.png'
    assert default_svg == tmp_path/'error_vs_voxel_size.svg'


def test_invalid_suffix_creates_no_plot_directory(tmp_path):
    source=tmp_path/'summary.json'
    source.write_text(json.dumps(_summary()))
    with pytest.raises(ValueError,match='must end'):
        save_experiment_plot(source,tmp_path/'bad'/'plot.json')
    assert not (tmp_path/'bad').exists()
