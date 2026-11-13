"""M4 integration: ray evidence, occlusion, missing depth and held-out isolation."""

from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from wtumi_recon.config import load_run_config
from wtumi_recon.occupancy import EvidenceOccupancyGrid, load_occupancy
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.tsdf import Open3DTSDFMap

ROOT = Path(__file__).resolve().parents[1]


def test_occlusion_reveal_and_missing_depth_use_only_fusion_observations(tmp_path, monkeypatch):
    occupancy_inputs, tsdf_inputs = [], []
    original_occ = EvidenceOccupancyGrid.integrate
    original_tsdf = Open3DTSDFMap.integrate

    def track_occ(self, depth, k, pose, *args, **kwargs):
        occupancy_inputs.append((depth.copy(), pose.copy()))
        return original_occ(self, depth, k, pose, *args, **kwargs)

    def track_tsdf(self, rgbd, intrinsic, pose):
        tsdf_inputs.append(np.asarray(rgbd.depth).copy())
        return original_tsdf(self, rgbd, intrinsic, pose)

    monkeypatch.setattr(EvidenceOccupancyGrid, 'integrate', track_occ)
    monkeypatch.setattr(Open3DTSDFMap, 'integrate', track_tsdf)
    results, volumes = {}, {}
    for case in ('front', 'reveal', 'missing'):
        path = ROOT / 'configs/scenes' / f'm4_occlusion_{case}.yaml'
        cfg = load_run_config(path)
        occupancy_inputs.clear()
        tsdf_inputs.clear()
        out = tmp_path / case
        report = run_reconstruction(path, out)
        occ = report['occupancy']
        fused = [v for v in cfg['views'] if v['split'] == 'integration']
        held = [v for v in cfg['views'] if v['split'] == 'held_out']
        assert len(occupancy_inputs) == len(tsdf_inputs) == len(fused)
        for (depth, pose), tsdf_depth, view in zip(occupancy_inputs, tsdf_inputs, fused):
            np.testing.assert_array_equal(depth, tsdf_depth)
            np.testing.assert_array_equal(depth, np.load(out / 'inputs' / view['id'] / 'depth_m.npy'))
            np.testing.assert_allclose(pose, view['T_wc'])
            assert not any(np.allclose(pose, v['T_wc']) for v in held)
        assert occ['integrated_frame_ids'] == [v['id'] for v in fused]
        assert occ['unknown_cells'] + occ['free_cells'] + occ['occupied_cells'] == occ['total_cells']
        assert occ['total_cells'] == 80**3
        assert occ['voxel_size_m'] == .025
        assert (out / 'occupancy_slices.png').is_file()
        assert json.loads((out / 'occupancy.json').read_text()) == occ
        results[case] = {p['id']:p['state'] for p in occ['probes']}
        volumes[case] = load_occupancy(out / 'occupancy.npz')

    assert results['front']['in_front'] == 'free'
    assert results['front']['occluder_surface'] == 'occupied'
    assert results['front']['behind_occluder_space'] == 'unknown'
    assert results['front']['hidden_cube_surface'] == 'unknown'
    assert results['reveal']['behind_occluder_space'] == 'free'
    assert results['reveal']['hidden_cube_surface'] == 'occupied'
    for case in results:
        assert results[case]['hidden_cube_interior'] == 'unknown'
        assert results[case]['outside_bounds'] == 'outside'
    assert np.count_nonzero(volumes['reveal']['state'] == 0) < np.count_nonzero(volumes['front']['state'] == 0)
    # Dropout removes a subset of rays, so it can never create new evidence.
    for name in ('hit_count', 'free_count'):
        assert np.all(volumes['missing'][name] <= volumes['front'][name])
    assert np.count_nonzero(volumes['missing']['state'] == 0) > np.count_nonzero(volumes['front']['state'] == 0)


@pytest.mark.parametrize('occupancy', [
    {'enabled': 'true'}, {'enabled': True, 'resolution': True},
    {'enabled': True, 'resolution': 0}, {'enabled': True, 'pixel_stride': 0},
    {'enabled': True, 'probes': [{'id':'bad', 'point_m':[0, 1]}]},
    {'enabled': True, 'probes': [{'id':'bad', 'point_m':[0, float('nan'), 1]}]},
    {'enabled': True, 'fill_unknown': True},
])
def test_invalid_occupancy_is_rejected_before_output_creation(tmp_path, occupancy):
    cfg = yaml.safe_load((ROOT/'configs/scenes/static_door.yaml').read_text())
    cfg['occupancy'] = occupancy
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        run_reconstruction(path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_occupancy_is_opt_in_and_requires_bounded_uniform_volume(tmp_path):
    cfg = load_run_config(ROOT/'configs/scenes/static_door.yaml')
    assert cfg['occupancy']['enabled'] is False
    raw = yaml.safe_load((ROOT/'configs/scenes/static_door.yaml').read_text())
    raw['reconstruction'] = {'backend':'scalable','voxel_size_m':.02,'sdf_trunc_m':.06}
    raw['occupancy'] = {'enabled':True}
    path=tmp_path/'bad_backend.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='uniform'):
        load_run_config(path)
