"""Acceptance tests for controlled, replayable reconstruction experiments."""

from pathlib import Path
import csv
import json

import numpy as np
import pytest
import yaml

from wtumi_recon.experiments import METRICS, aggregate_runs, load_experiment_config, run_experiment


def _scene():
    return {
        "schema_version": 2,
        "scene": {"name": "small_cube", "objects": [{"type": "box", "size_m": [0.4, 0.4, 0.4],
                  "translation_m": [0.3, 0.3, 0.8], "color_mode": "coordinate", "color_rgb": [180, 70, 40]}]},
        "camera": {"width": 64, "height": 48, "fx": 70.0, "fy": 70.0,
                   "cx": 31.5, "cy": 23.5, "depth_trunc_m": 3.0},
        "reconstruction": {"backend": "uniform", "volume_length_m": 1.6, "resolution": 32,
                           "origin_m": [0, 0, 0], "sdf_trunc_m": 0.1},
        "views": [{"id": "front", "split": "integration", "eye": [0.5, 0.5, -0.3], "target": [0.5, 0.5, 1]},
                  {"id": "novel", "split": "held_out", "eye": [0.7, 0.45, -0.1], "target": [0.5, 0.5, 1]}],
        "evaluation": {"depth_tolerance_m": 0.02},
    }


def _setup(tmp_path):
    root = tmp_path / "configuration"
    (root / "scenes").mkdir(parents=True)
    (root / "experiments").mkdir()
    base = root / "scenes" / "base.yaml"
    base.write_text(yaml.safe_dump(_scene()))
    data = {"schema_version": 1, "base_config": "../scenes/base.yaml", "resolutions": [32, 40],
            "seeds": [7, 19], "profiles": [
                {"name": "clean", "depth_noise_std_m": 0, "dropout_probability": 0},
                {"name": "noisy", "depth_noise_std_m": 0.002, "dropout_probability": 0.05}]}
    path = root / "experiments" / "test.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, base, data


def test_paths_resolve_from_experiment_and_base_not_cwd(tmp_path, monkeypatch):
    path, base_path, _ = _setup(tmp_path)
    data = _scene()
    data["scene"]["objects"] = [{"type": "mesh", "path": "assets/box.ply"}]
    base_path.write_text(yaml.safe_dump(data))
    monkeypatch.chdir(tmp_path.parent)
    experiment, base = load_experiment_config(path)
    assert Path(experiment["base_config"]) == base_path
    assert Path(base["scene"]["objects"][0]["path"]) == base_path.parent / "assets/box.ply"


@pytest.mark.parametrize("field,value", [
    ("resolutions", [32, 32]), ("resolutions", [True]), ("resolutions", [0]),
    ("seeds", [7, 7]), ("seeds", [-1]), ("profiles", [{"name": "../escape"}]),
    ("profiles", [{"name": "bad", "depth_noise_std_m": float("nan")}]),
    ("profiles", [{"name": "bad", "dropout_probability": 1.1}]),
])
def test_invalid_experiment_is_rejected_without_creating_output(tmp_path, field, value):
    path, _, data = _setup(tmp_path)
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    out = tmp_path / "output"
    with pytest.raises(ValueError):
        run_experiment(path, out)
    assert not out.exists()


def test_existing_results_are_not_overwritten(tmp_path):
    path, _, _ = _setup(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    original = out / "keep.txt"
    original.write_text("existing experiment")
    with pytest.raises(FileExistsError):
        run_experiment(path, out)
    assert original.read_text() == "existing experiment"
    assert list(out.iterdir()) == [original]


def test_aggregate_counts_missing_values_and_reports_descriptive_seed_stats():
    rows = []
    for seed, value in [(7, 0.5), (19, None), (31, 1.0)]:
        rows.append({"profile": "noisy", "resolution": 32, "voxel_size_m": 0.05, "seed": seed,
                     **dict.fromkeys(METRICS, None), "coverage": value})
    groups = aggregate_runs(rows)
    assert len(groups) == 1
    group = groups[0]
    assert group["runs"] == 3 and group["seeds"] == [7, 19, 31]
    assert group["metrics"]["coverage"] == {"valid_runs": 2, "mean": 0.75, "std_population": 0.25}
    assert group["metrics"]["mae_m"] == {"valid_runs": 0, "mean": None, "std_population": None}
    json.dumps(groups, allow_nan=False)


def test_small_sweep_pairs_inputs_and_preserves_clean_reference(tmp_path):
    path, base_path, data = _setup(tmp_path)
    original_config = base_path.read_bytes()
    messages = []
    out = tmp_path / "results"
    report = run_experiment(path, out, progress=messages.append)
    # Two clean cases, four noisy cases; clean seeds are not repeated.
    assert len(report["runs"]) == len(messages) == 6
    clean = [r for r in report["runs"] if r["profile"] == "clean"]
    assert len(clean) == 2 and {r["seed"] for r in clean} == {7}
    noisy = [r for r in report["runs"] if r["profile"] == "noisy"]
    for seed in (7, 19):
        same_seed = [r for r in noisy if r["seed"] == seed]
        assert len({r["integration_depth_sha256"] for r in same_seed}) == 1
    assert len({r["integration_depth_sha256"] for r in noisy}) == 2
    assert len({r["held_out_depth_sha256"] for r in report["runs"]}) == 1
    assert report["paired_inputs_verified"] and report["clean_held_out_references_verified"]
    assert base_path.read_bytes() == original_config
    for row in report["runs"]:
        replay = yaml.safe_load((out / "configs" / f"{row['case_id']}.yaml").read_text())
        assert replay["reconstruction"]["resolution"] == row["resolution"]
        assert replay["reconstruction"]["sdf_trunc_m"] == 0.1
        assert row["grid_cells"] == row["resolution"]**3
        assert row["voxel_size_m"] == pytest.approx(1.6 / row["resolution"])
        assert (out / row["comparison"]).is_file()
        assert replay["sensor"]["seed"] == row["seed"]
        run = out / row["run_dir"]
        actual = np.load(run / "inputs/front/depth_m.npy")
        reference = np.load(run / "inputs/novel/depth_m.npy")
        assert actual.dtype == reference.dtype == np.float32
    saved = json.loads((out / "summary.json").read_text())
    assert saved["runs"] == report["runs"]
    with (out / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    assert (out / "overview.png").is_file()
    assert "Single-seed" in (out / "report.md").read_text()
