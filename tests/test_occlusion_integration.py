"""Masked portable observations must reach both maps without reference leakage."""

import numpy as np
import yaml

from wtumi_recon.occupancy import EvidenceOccupancyGrid, load_occupancy
from wtumi_recon.pipeline import run_reconstruction
from wtumi_recon.render import look_at_pose
from wtumi_recon.sequence import RGBDSequence, SequenceWriter
from wtumi_recon.synthetic import SyntheticScene
from wtumi_recon.tsdf import Open3DTSDFMap


def test_masked_sequence_fuses_same_observations_into_tsdf_and_occupancy(tmp_path, monkeypatch):
    width, height = 64, 48
    camera = {
        'width': width, 'height': height,
        'K': [[70., 0., 31.5], [0., 70., 23.5], [0., 0., 1.]],
        'depth_trunc_m': 3.,
    }
    source = SyntheticScene.from_config({'objects': [{
        'type': 'box', 'size_m': [.4, .4, .4],
        'translation_m': [.3, .3, .8], 'color_mode': 'coordinate',
    }]})
    views = [
        ('front', 'integration', [.5, .5, -.3]),
        ('left', 'integration', [.2, .5, -.1]),
        ('novel', 'held_out', [.75, .55, 0.]),
    ]
    writer = SequenceWriter(tmp_path/'capture', camera, 'masked_cube',
                            'simulator_ground_truth', {'world_up': [0, -1, 0]})
    expected, references, poses, masks = {}, {}, {}, {}
    for index, (name, split, eye) in enumerate(views):
        pose = look_at_pose(eye, [.5, .5, 1.], [0, -1, 0])
        rgb, clean_depth = source.render(np.asarray(camera['K']), pose, width, height)
        clean_depth[(clean_depth <= 0) | ~np.isfinite(clean_depth) | (clean_depth >= 3)] = 0
        rgb[clean_depth == 0] = 0
        observation, observation_rgb = clean_depth.copy(), rgb.copy()
        if split == 'integration':
            # A contiguous occlusion stripe covers real returns, not background.
            mask = np.zeros_like(observation, dtype=bool)
            if index == 0:
                mask[:, width//2-4:width//2+4] = True
            else:
                mask[height//2-3:height//2+3, :] = True
            mask &= clean_depth > 0
            assert mask.any()
            observation[mask], observation_rgb[mask] = 0, 0
            assert (observation > 0).any()
            expected[name], poses[name], masks[name] = observation.copy(), pose.copy(), mask
        references[name] = clean_depth.copy()
        writer.add_frame(name, split, observation_rgb, observation, pose, float(index),
                         reference_rgb=rgb, reference_depth_m=clean_depth)
    writer.finish()
    config = {
        'schema_version': 3,
        'source': {'type': 'rgbd_sequence', 'manifest': 'capture/manifest.json'},
        'reconstruction': {'backend': 'uniform', 'volume_length_m': 1.6,
                           'resolution': 40, 'origin_m': [0., 0., 0.], 'sdf_trunc_m': .1},
        'occupancy': {'enabled': True, 'resolution': 20, 'pixel_stride': 1},
        'evaluation': {'depth_tolerance_m': .02},
    }
    config_path = tmp_path/'reconstruction.yaml'
    config_path.write_text(yaml.safe_dump(config))

    tsdf_calls, occupancy_calls, observation_reads, heldout_reads = [], [], [], []
    real_tsdf_integrate = Open3DTSDFMap.integrate
    real_occupancy_integrate = EvidenceOccupancyGrid.integrate
    real_read, real_reference = RGBDSequence.read, RGBDSequence.read_reference

    def track_tsdf(self, rgbd, intrinsic, pose):
        tsdf_calls.append((np.asarray(rgbd.depth).copy(), np.asarray(pose).copy()))
        return real_tsdf_integrate(self, rgbd, intrinsic, pose)

    def track_occupancy(self, depth, k, pose, depth_trunc_m, pixel_stride=1):
        occupancy_calls.append((np.asarray(depth).copy(), np.asarray(pose).copy()))
        return real_occupancy_integrate(self, depth, k, pose, depth_trunc_m, pixel_stride)

    def check_observation_read(self, view):
        assert view['split'] == 'integration', 'held-out observations reached fusion loading'
        observation_reads.append(view['id'])
        return real_read(self, view)

    def check_reference_read(self, view):
        if view['split'] == 'held_out':
            assert len(tsdf_calls) == len(occupancy_calls) == len(expected)
            heldout_reads.append(view['id'])
        return real_reference(self, view)

    monkeypatch.setattr(Open3DTSDFMap, 'integrate', track_tsdf)
    monkeypatch.setattr(EvidenceOccupancyGrid, 'integrate', track_occupancy)
    monkeypatch.setattr(RGBDSequence, 'read', check_observation_read)
    monkeypatch.setattr(RGBDSequence, 'read_reference', check_reference_read)
    out = tmp_path/'reconstruction'
    report = run_reconstruction(config_path, out)

    assert observation_reads == list(expected)
    assert heldout_reads == ['novel']
    assert len(tsdf_calls) == len(occupancy_calls) == len(expected)
    assert report['integrated_frame_ids'] == report['occupancy']['integrated_frame_ids'] == list(expected)
    assert report['held_out_frame_ids'] == ['novel']
    assert report['mesh_triangles'] > 0
    for name, tsdf_call, occupancy_call in zip(expected, tsdf_calls, occupancy_calls):
        tsdf_depth, tsdf_pose = tsdf_call
        occupancy_depth, occupancy_pose = occupancy_call
        np.testing.assert_array_equal(tsdf_depth, expected[name])
        np.testing.assert_array_equal(occupancy_depth, expected[name])
        np.testing.assert_array_equal(tsdf_pose, poses[name])
        np.testing.assert_array_equal(occupancy_pose, poses[name])
        np.testing.assert_array_equal(tsdf_depth[masks[name]], 0)
        np.testing.assert_array_equal(tsdf_depth[~masks[name]], references[name][~masks[name]])
        assert not np.array_equal(tsdf_depth, references[name])
        assert not np.array_equal(tsdf_depth, references['novel'])
        np.testing.assert_array_equal(np.load(out/'inputs'/name/'depth_m.npy'), expected[name])
        np.testing.assert_array_equal(np.load(out/'references'/name/'depth_m.npy'), references[name])

    # The exported evidence must include only the two masked observation frames.
    occupancy = load_occupancy(out/'occupancy.npz')
    assert occupancy['frames'] == len(expected)
    assert occupancy['valid_rays'] == sum(np.count_nonzero(depth) for depth in expected.values())
    np.testing.assert_array_equal(np.load(out/'references/novel/depth_m.npy'), references['novel'])
