"""Portable three-scene reporting and orchestration without rendering."""

import csv
import json
from pathlib import Path
import re
import runpy
import sys

import pytest

from wtumi_recon.scene_suite import SCENES, run_scene_suite, write_scene_suite_report


def _completed_run(root, name, index=0, *, video=False):
    directory = root / name
    directory.mkdir(parents=True)
    paths = {
        "gallery": "index.html",
        "camera_voxel_gif": "comparison/camera_voxel.gif",
        "camera_voxel_png": "comparison/camera_voxel.png",
        "camera_voxel_metadata": "comparison/camera_voxel.json",
    }
    if video:
        paths["camera_voxel_mp4"] = "comparison/camera_voxel.mp4"
    for value in paths.values():
        artifact = directory / value
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"fixture artifact")
    run = {
        "format": "wtumi-scene-run-v1", "paths": paths,
        "integration_frames": 13 + index, "held_out_frames": 2,
        "effective_voxel_size_m": .02 + index * .005,
        "held_out": {"coverage": .5 + index * .1,
                     "mae_m": .001 + index * .002,
                     "rmse_m": .003 + index * .004},
    }
    (directory / "scene_run.json").write_text(json.dumps(run), encoding="utf-8")
    return run


@pytest.fixture
def completed_suite(tmp_path):
    root = tmp_path / "suite"
    for index, name in enumerate(SCENES):
        _completed_run(root, name, index, video=index == 0)
    return root


def _change_path(root, name, key, value):
    path = root / name / "scene_run.json"
    run = json.loads(path.read_text())
    run["paths"][key] = value
    path.write_text(json.dumps(run))


def test_suite_reports_all_three_scenes_with_explicit_metric_units_and_relative_links(completed_suite):
    root = completed_suite
    report = write_scene_suite_report(root)
    assert tuple(row["scene"] for row in report["scenes"]) == ("room", "doorway", "g1_stairs")
    assert report == json.loads((root / "suite.json").read_text())
    with (root / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for index, (row, saved) in enumerate(zip(report["scenes"], rows)):
        assert row["integration_frames"] == 13 + index
        assert row["held_out_views"] == 2
        for key, expected in (
            ("voxel_size_mm", 20 + index * 5),
            ("depth_coverage_percent", 50 + index * 10),
            ("mean_absolute_depth_error_mm", 1 + index * 2),
            ("root_mean_square_depth_error_mm", 3 + index * 4),
        ):
            assert row[key] == pytest.approx(expected)
            assert float(saved[key]) == pytest.approx(expected)
    page = (root / "index.html").read_text()
    assert "Left: recorded head-camera RGB" in page
    assert "cumulative voxel reconstruction after the same observation" in page
    assert "not whole-scene completeness" in page
    assert '<video controls' in page and '<img src="doorway/' in page
    for value in report["paths"].values():
        assert not Path(value).is_absolute()
        assert (root / value).is_file()
    for value in re.findall(r'(?:href|src|poster)="([^"]+)"', page):
        assert not Path(value).is_absolute()
        assert (root / value).is_file(), value
    assert str(root) not in page
    moved = root.parent / "moved_suite"
    root.rename(moved)
    write_scene_suite_report(moved)
    assert (moved / "index.html").read_text() == page


@pytest.mark.parametrize("key", ["gallery", "camera_voxel_gif", "camera_voxel_png",
                                "camera_voxel_metadata", "camera_voxel_mp4"])
def test_suite_rejects_missing_artifacts_before_writing_reports(completed_suite, key):
    _change_path(completed_suite, "doorway", key, "missing/artifact")
    with pytest.raises(FileNotFoundError):
        write_scene_suite_report(completed_suite)
    assert not (completed_suite / "index.html").exists()
    assert not (completed_suite / "summary.csv").exists()
    assert not (completed_suite / "suite.json").exists()


@pytest.mark.parametrize("key", ["gallery", "camera_voxel_gif", "camera_voxel_png",
                                "camera_voxel_metadata", "camera_voxel_mp4"])
def test_suite_rejects_escaping_artifact_paths(completed_suite, key):
    outside = completed_suite / "outside.txt"
    outside.write_text("outside the per-scene directory")
    _change_path(completed_suite, "room", key, "../outside.txt")
    with pytest.raises(ValueError, match="escapes"):
        write_scene_suite_report(completed_suite)
    assert not (completed_suite / "index.html").exists()


@pytest.mark.parametrize("absolute", [False, True])
def test_suite_rejects_absolute_paths_and_symlink_escape(completed_suite, absolute):
    outside = completed_suite.parent / "outside.png"
    outside.write_text("outside suite")
    if absolute:
        value = str(outside)
    else:
        value = "linked.png"
        (completed_suite / "g1_stairs" / value).symlink_to(outside)
    _change_path(completed_suite, "g1_stairs", "camera_voxel_png", value)
    with pytest.raises(ValueError, match="escapes"):
        write_scene_suite_report(completed_suite)


def test_missing_scene_and_unsupported_report_preserve_existing_summaries(completed_suite):
    summary = completed_suite / "summary.csv"
    summary.write_text("keep previous report")
    scene = completed_suite / "g1_stairs/scene_run.json"
    original = json.loads(scene.read_text())
    scene.unlink()
    with pytest.raises(FileNotFoundError):
        write_scene_suite_report(completed_suite)
    assert summary.read_text() == "keep previous report"
    scene.write_text(json.dumps({**original, "format": "unknown-version"}))
    with pytest.raises(ValueError, match="Unsupported"):
        write_scene_suite_report(completed_suite)
    assert summary.read_text() == "keep previous report"


def test_suite_preflights_all_scenes_then_runs_and_aggregates_each_once(tmp_path, monkeypatch):
    from wtumi_recon import scene_bundle, scene_pipeline

    events, messages = [], []
    monkeypatch.setattr(scene_bundle, "load_scene_config", lambda name: events.append(("validate", name)))

    def fake_run(name, destination, progress=None):
        events.append(("run", name))
        assert destination == tmp_path / "suite" / name
        assert progress is messages.append or progress == messages.append
        return _completed_run(destination.parent, name, SCENES.index(name))

    monkeypatch.setattr(scene_pipeline, "run_scene", fake_run)
    report = run_scene_suite(tmp_path / "suite", progress=messages.append)
    assert events == [("validate", name) for name in SCENES] + [("run", name) for name in SCENES]
    assert len(messages) == 3
    assert [row["scene"] for row in report["scenes"]] == list(SCENES)


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_suite_output_is_preserved_before_configuration_or_rendering(tmp_path, monkeypatch, existing_file):
    from wtumi_recon import scene_bundle, scene_pipeline

    destination = tmp_path / "suite"
    if existing_file:
        destination.write_text("keep file")
        marker = destination
    else:
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep file")
    monkeypatch.setattr(scene_bundle, "load_scene_config", lambda *_: pytest.fail("must preserve output before validation"))
    monkeypatch.setattr(scene_pipeline, "run_scene", lambda *_args, **_kwargs: pytest.fail("must not render"))
    with pytest.raises(FileExistsError, match="new or empty"):
        run_scene_suite(destination)
    assert marker.read_text() == "keep file"


def test_invalid_template_preflight_does_not_start_any_scene(tmp_path, monkeypatch):
    from wtumi_recon import scene_bundle, scene_pipeline

    def validate(name):
        if name == "doorway":
            raise ValueError("invalid doorway")

    monkeypatch.setattr(scene_bundle, "load_scene_config", validate)
    monkeypatch.setattr(scene_pipeline, "run_scene", lambda *_args, **_kwargs: pytest.fail("must preflight all scenes"))
    with pytest.raises(ValueError, match="invalid doorway"):
        run_scene_suite(tmp_path / "suite")
    assert not (tmp_path / "suite").exists()


def test_suite_cli_passes_explicit_output_and_prints_report(tmp_path, monkeypatch, capsys):
    from wtumi_recon import scene_suite

    output = tmp_path / "requested suite"
    seen = []

    def fake_suite(destination, progress=None):
        seen.append(destination)
        return {"paths": {"gallery": "index.html", "summary": "summary.csv"}}

    monkeypatch.setattr(scene_suite, "run_scene_suite", fake_suite)
    script = Path(__file__).resolve().parents[1] / "scripts/run_scene_suite.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    assert seen == [output]
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(output.resolve())
    assert report["paths"]["gallery"] == "index.html"


def _saved_run(root, name, config):
    """Write lightweight real history/pose metadata; do not render or encode."""
    import numpy as np
    from wtumi_recon.scene_bundle import _dump_yaml

    run = _completed_run(root, name, video=True)
    directory = root / name
    scene = directory / "scene"
    scene.mkdir()
    (scene / "scene.yaml").write_text(_dump_yaml({key: value for key, value in config.items() if not key.startswith("_")}))
    (scene / "model.xml").write_text("fixture model")
    capture = directory / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text("{}")
    observations, history = [], []
    ids = [frame["id"] for frame in config["trajectory"]]
    for index, frame in enumerate(config["trajectory"]):
        frame_dir = capture / "frames" / frame["id"]
        frame_dir.mkdir(parents=True)
        (frame_dir / "raw_rgb.png").write_bytes(b"fixture raw RGB")
        relative = "reconstruction" if index == len(ids) - 1 else f"prefix/{index:03d}"
        prefix = directory / relative
        prefix.mkdir(parents=True)
        (prefix / "metrics.json").write_text(json.dumps({"integrated_frame_ids": ids[:index + 1]}))
        (prefix / "tsdf_volume.npz").write_bytes(b"fixture saved TSDF")
        observations.append({"id": frame["id"], "timestamp_s": frame["timestamp_s"],
                             "frame_directory": f"frames/{frame['id']}", "T_wc": np.eye(4).tolist()})
        history.append({"id": frame["id"], "timestamp_s": frame["timestamp_s"],
                        "frames": index + 1, "run_dir": relative})
    (capture / "capture.json").write_text(json.dumps({"frames": observations, "integration_frame_ids": ids,
                                                      "held_out_frame_ids": [f["id"] for f in config["held_out"]]}))
    (directory / "history.json").write_text(json.dumps(history))
    run["paths"].update(scene_config="scene/scene.yaml", model="scene/model.xml",
                        capture_manifest="capture/manifest.json", capture_report="capture/capture.json",
                        history="history.json", final_result="reconstruction",
                        inputs_animation="inputs.gif", reconstruction_animation="reconstruction.gif")
    for key in ("inputs_animation", "reconstruction_animation"):
        (directory / run["paths"][key]).write_bytes(b"fixture animation")
    run["integration_frames"] = len(ids)
    (directory / "scene_run.json").write_text(json.dumps(run))
    return run


@pytest.fixture
def replacement_suite(tmp_path, monkeypatch):
    from wtumi_recon import scene_bundle, scene_pipeline, scene_presets, video

    root = tmp_path / "existing"
    configs = {}
    for name in SCENES:
        config = scene_bundle.load_scene_config(name)
        config = scene_presets.preset_config(config, "default")
        config["trajectory"] = [config["trajectory"][0], config["trajectory"][-1]]
        if name in ("room", "doorway"):
            config["box"]["enabled"] = False
        configs[name] = config
        _saved_run(root, name, config)
        bundle = root / "inputs" / name
        bundle.mkdir(parents=True)
        (bundle / "scene.yaml").write_text(scene_bundle._dump_yaml(config))
    write_scene_suite_report(root)
    (root / "preset.json").write_text(json.dumps(scene_presets.describe_preset(configs, "default")))
    calls = []

    def fake_create(name, directory):
        directory.mkdir(parents=True)
        portable = scene_presets.preset_config(scene_bundle.load_scene_config(name), "default")
        (directory / "scene.yaml").write_text(scene_bundle._dump_yaml(portable))

    def fake_run(bundle, directory, progress=None):
        calls.append(bundle.name)
        return _saved_run(directory.parent, directory.name, scene_bundle.load_scene_config(bundle))

    monkeypatch.setattr(scene_bundle, "create_scene", fake_create)
    monkeypatch.setattr(scene_pipeline, "run_scene", fake_run)
    monkeypatch.setattr(video, "require_video_encoder", lambda: {"codec": "fixture"})
    return root, calls


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_selected_replacement_rebuilds_current_templates_and_preserves_stairs(replacement_suite):
    root, calls = replacement_suite
    stairs = _snapshot(root / "g1_stairs")
    stairs_input = _snapshot(root / "inputs/g1_stairs")
    for name in ("room", "doorway"):
        (root / name / "comparison/camera_voxel.mp4").unlink()
        (root / name / "inputs.gif").unlink()
    report = run_scene_suite(root, preset="pace30", replace_scenes=["doorway", "room"])
    assert calls == ["room", "doorway"]
    assert _snapshot(root / "g1_stairs") == stairs
    assert _snapshot(root / "inputs/g1_stairs") == stairs_input
    assert not (root / ".scene-replacement").exists()
    plan = json.loads((root / "preset.json").read_text())
    assert report["preset"] == plan["preset"] == "mixed"
    assert report["last_replaced_scenes"] == ["room", "doorway"]
    for row, setting in zip(report["scenes"], plan["scenes"]):
        assert row["box_enabled"] is True
        if row["scene"] in ("room", "doorway"):
            assert row["integration_frames"] == setting["integration_frames"] == 121
            assert row["robot_pose"] == "carry"
            assert row["camera_width"] == 640 and row["playback_fps"] == 30
            assert setting["source_waypoints"] == 13
            assert setting["scene_preset"] == "pace30"
        else:
            assert row["integration_frames"] == 2
            assert setting["scene_preset"] == "default"
    assert json.loads((root / "suite.json").read_text()) == report
    assert "only the stairs" not in (root / "index.html").read_text()


def test_staged_reconstruction_failure_preserves_every_existing_file(replacement_suite, monkeypatch):
    from wtumi_recon import scene_pipeline

    root, calls = replacement_suite
    before = _snapshot(root)
    run = scene_pipeline.run_scene

    def fail_second(bundle, directory, progress=None):
        if directory.name == "doorway":
            raise RuntimeError("injected reconstruction failure")
        return run(bundle, directory, progress)

    monkeypatch.setattr(scene_pipeline, "run_scene", fail_second)
    with pytest.raises(RuntimeError, match="injected reconstruction"):
        run_scene_suite(root, preset="pace30", replace_scenes=["room", "doorway"])
    assert calls == ["room"]
    assert _snapshot(root) == before
    assert not (root / ".scene-replacement").exists()


@pytest.mark.parametrize("point", ["after_scene_rename", "after_report_publish"])
def test_commit_failure_rolls_back_even_after_filesystem_change(replacement_suite, monkeypatch, point):
    from wtumi_recon import scene_suite

    root, _ = replacement_suite
    before = _snapshot(root)
    rename = Path.rename
    replace = scene_suite.os.replace
    injected = False

    def fail_rename(source, target):
        nonlocal injected
        result = rename(source, target)
        if not injected and source == root / ".scene-replacement/new/room":
            injected = True
            raise KeyboardInterrupt("injected after rename")
        return result

    def fail_publish(source, target):
        nonlocal injected
        result = replace(source, target)
        if not injected and Path(source) == root / ".scene-replacement/new/reports/preset.json":
            injected = True
            raise KeyboardInterrupt("injected after report publication")
        return result

    monkeypatch.setattr(Path, "rename", fail_rename if point == "after_scene_rename" else rename)
    monkeypatch.setattr(scene_suite.os, "replace", fail_publish if point == "after_report_publish" else replace)
    with pytest.raises(KeyboardInterrupt, match="injected"):
        run_scene_suite(root, preset="pace30", replace_scenes=["room", "doorway"])
    assert injected
    assert _snapshot(root) == before
    assert not (root / ".scene-replacement").exists()


def test_replacement_dry_run_uses_current_selected_and_saved_untouched_configs(replacement_suite, monkeypatch):
    from wtumi_recon import scene_pipeline, video
    from wtumi_recon.scene_suite import scene_replacement_plan

    root, calls = replacement_suite
    before = _snapshot(root)
    monkeypatch.setattr(video, "require_video_encoder", lambda: pytest.fail("dry run must not require encoder"))
    monkeypatch.setattr(scene_pipeline, "run_scene", lambda *_a, **_k: pytest.fail("dry run must not capture"))
    plan = scene_replacement_plan(root, ["room", "doorway"], "pace30")
    assert plan["status"] == "planned"
    assert plan["replace_scenes"] == ["room", "doorway"]
    assert [row["integration_frames"] for row in plan["scenes"]] == [121, 121, 2]
    assert [row["box_enabled"] for row in plan["scenes"]] == [True, True, True]
    assert _snapshot(root) == before and calls == []


@pytest.mark.parametrize("problem", ["pending_transaction", "unknown_format", "missing_prefix", "symlink", "untouched_missing_media"])
def test_replacement_preflight_rejects_unsafe_or_incomplete_suite(replacement_suite, problem):
    root, calls = replacement_suite
    if problem == "pending_transaction":
        (root / ".scene-replacement").mkdir()
        (root / ".scene-replacement/journal.json").write_text("interrupted")
    elif problem == "unknown_format":
        (root / "suite.json").write_text(json.dumps({"format": "unknown"}))
    elif problem == "missing_prefix":
        (root / "room/reconstruction/tsdf_volume.npz").unlink()
    elif problem == "symlink":
        (root / "room/linked").symlink_to(root / "g1_stairs")
    else:
        (root / "g1_stairs/inputs.gif").unlink()
    before = _snapshot(root)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        run_scene_suite(root, preset="pace30", replace_scenes=["room", "doorway"])
    assert _snapshot(root) == before and calls == []


@pytest.mark.parametrize("names", [[], ["room", "room"], ["unrecognized"]])
def test_replacement_api_rejects_invalid_scene_names(tmp_path, names):
    with pytest.raises(ValueError):
        run_scene_suite(tmp_path / "not_created", replace_scenes=names)
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("arguments", [["--replace-scenes", "room"],
                                       ["--output", "existing", "--replace-scenes", "room", "room"]])
def test_replacement_cli_requires_explicit_output_and_unique_names(monkeypatch, arguments):
    script = Path(__file__).resolve().parents[1] / "scripts/run_scene_suite.py"
    monkeypatch.setattr(sys, "argv", [str(script), *arguments])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_replacement_cli_passes_selected_scenes_and_preset(tmp_path, monkeypatch, capsys):
    from wtumi_recon import scene_suite

    script = Path(__file__).resolve().parents[1] / "scripts/run_scene_suite.py"
    seen = []
    monkeypatch.setattr(scene_suite, "run_scene_suite", lambda output, **options: seen.append((output, options)) or {"paths": {}})
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path), "--preset", "pace30", "--replace-scenes", "room", "doorway"])
    runpy.run_path(str(script), run_name="__main__")
    assert seen[0][0] == tmp_path
    assert seen[0][1]["preset"] == "pace30"
    assert seen[0][1]["replace_scenes"] == ["room", "doorway"]
    capsys.readouterr()
