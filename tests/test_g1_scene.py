"""A real robot visual model stays portable within quick-scene workflows."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from wtumi_recon.scene_bundle import build_scene, create_scene, load_scene_config


def test_g1_template_uses_provisional_head_mount_and_no_head_scan():
    cfg = load_scene_config('g1_stairs')
    assert cfg['robot']['model'] == 'unitree_g1'
    assert cfg['torso']['enabled'] is False
    assert 1.32 < cfg['camera']['position_in_rig_m'][2] < 1.45
    assert all(row['head_yaw_degrees'] == row['head_pitch_degrees'] == 0 for row in cfg['trajectory'])


def test_created_and_built_g1_scene_moves_without_source_assets(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    source = tmp_path / 'custom'
    create_scene('g1_stairs', source)
    cfg = load_scene_config(source)
    assert cfg['robot']['asset_dir'] == 'robot_assets/unitree_g1'
    assert (source / cfg['robot']['asset_dir'] / 'LICENSE').is_file()
    build_scene(source, tmp_path / 'built')
    shutil.rmtree(source)
    moved = tmp_path / 'relocated'
    shutil.move(tmp_path / 'built', moved)
    load_scene_config(moved)
    model = mujoco.MjModel.from_xml_path(str(moved / 'model.xml'))
    assert model.njnt == 0
    assert model.nmocap == 1
    assert np.count_nonzero(model.geom_group == 1) >= 29
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'torso') == -1
    assert 'include' not in (moved / 'model.xml').read_text()
    assert not np.any(model.geom_contype)
    assert not np.any(model.geom_conaffinity)
    assert (moved / 'robot_assets/unitree_g1/g1.xml').is_file()


@pytest.mark.parametrize('edit,match', [
    (lambda c: c['torso'].update(enabled=True), 'torso.enabled'),
    (lambda c: c['robot'].update(asset_dir='../outside'), 'asset_dir'),
    (lambda c: c['robot'].update(model='different_robot'), 'model'),
])
def test_robot_options_fail_before_build_output(tmp_path, edit, match):
    cfg = {k: v for k, v in load_scene_config('g1_stairs').items() if not k.startswith('_')}
    edit(cfg)
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        build_scene(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
