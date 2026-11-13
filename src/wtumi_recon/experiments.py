"""Controlled resolution/noise sweeps using the existing reconstruction pipeline."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .config import load_run_config
from .pipeline import run_reconstruction, write_json
from .sensor import normalize_sensor_config


METRICS = (
    "coverage", "mae_m", "rmse_m", "missing_pixels", "extra_pixels",
    "silhouette_iou", "accurate_reference_fraction", "mesh_triangles",
    "total_seconds", "integration_seconds",
)


def load_experiment_config(path):
    """Resolve paths against the experiment YAML; reject ambiguous sweeps."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "base_config", "resolutions", "seeds", "profiles"}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("experiment requires schema_version: 1")
    if set(raw) - allowed:
        raise ValueError(f"unsupported experiment fields: {set(raw) - allowed}")
    if not isinstance(raw.get("base_config"), str) or not raw["base_config"]:
        raise ValueError("experiment.base_config must be a scene YAML path")
    base_path = (path.parent / Path(raw["base_config"]).expanduser()).resolve()
    base = load_run_config(base_path)
    if base["reconstruction"].get("backend", "uniform") != "uniform":
        raise ValueError("resolution experiments require the uniform backend")
    resolutions = raw.get("resolutions")
    if not isinstance(resolutions, list) or not resolutions or any(type(n) is not int or n < 2 for n in resolutions):
        raise ValueError("resolutions must be a non-empty list of integers >= 2")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not contain duplicates")
    seeds = raw.get("seeds", [7])
    if not isinstance(seeds, list) or not seeds or any(type(n) is not int or n < 0 for n in seeds):
        raise ValueError("seeds must be a non-empty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")
    names, normalized = set(), []
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) - {"name", "depth_noise_std_m", "dropout_probability"}:
            raise ValueError("profile fields are name, depth_noise_std_m, dropout_probability")
        name = profile.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("profile names must be unique safe alphanumeric/underscore/hyphen IDs")
        names.add(name)
        sensor = normalize_sensor_config({k: v for k, v in profile.items() if k != "name"})
        normalized.append({"name": name, "depth_noise_std_m": sensor["depth_noise_std_m"],
                           "dropout_probability": sensor["dropout_probability"]})
    for seed in seeds:
        normalize_sensor_config({"seed": seed})
    return {"schema_version": 1, "base_config": str(base_path), "resolutions": resolutions,
            "seeds": seeds, "profiles": normalized, "_path": str(path)}, base


def aggregate_runs(rows):
    """Descriptive per-seed means/std; do not confuse them with pooled pixels."""
    groups = {}
    for row in rows:
        groups.setdefault((row["profile"], row["resolution"]), []).append(row)
    output = []
    for (profile, resolution), members in groups.items():
        item = {"profile": profile, "resolution": resolution,
                "voxel_size_m": members[0]["voxel_size_m"], "runs": len(members),
                "seeds": [row["seed"] for row in members], "metrics": {}}
        for key in METRICS:
            values = [row[key] for row in members if row[key] is not None]
            item["metrics"][key] = {"valid_runs": len(values),
                                     "mean": float(np.mean(values)) if values else None,
                                     "std_population": float(np.std(values, ddof=0)) if values else None}
        output.append(item)
    return output


def _frame_digest(run_dir, views, split):
    digest = hashlib.sha256()
    for view in sorted((v for v in views if v["split"] == split), key=lambda v: v["id"]):
        digest.update(view["id"].encode("utf-8") + b"\0")
        digest.update((run_dir / "inputs" / view["id"] / "depth_m.npy").read_bytes())
    return digest.hexdigest()


def _overview(out, rows, view_id):
    """One reconstructed RGB and one depth-error panel per case, first held-out view."""
    thumbs = []
    for row in rows:
        run = out / row["run_dir"] / "renders" / view_id
        rgb = Image.open(run / "rgb.png").convert("RGB")
        depth = np.load(run / "depth_m.npy")
        errors = np.load(run / "depth_abs_error_m.npy")
        missing = np.asarray(Image.open(run / "missing.png")) > 0
        extra = np.asarray(Image.open(run / "extra.png")) > 0
        panel = np.zeros(depth.shape + (3,), dtype=np.uint8)
        valid = np.isfinite(errors)
        panel[valid, 0] = np.rint(np.clip(errors[valid] / row["depth_tolerance_m"], 0, 1) * 255).astype(np.uint8)
        panel[missing], panel[extra] = [255, 0, 255], [0, 255, 255]
        thumbs.append((row, rgb, Image.fromarray(panel)))
    # Case columns stay a manageable width; extra seeds continue on later rows.
    columns = min(3, len(thumbs))
    cell_w, image_h = 320, 240
    cell_h = 2 * image_h + 68
    canvas = Image.new("RGB", (columns * cell_w, ((len(thumbs)+columns-1)//columns)*cell_h+60), "#151b24")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"Held-out: {view_id} | Top: RGB; bottom: depth error", fill="white")
    draw.text((12, 26), "Magenta: missing; cyan: extra; red: >= tolerance. Not a 3D accuracy map.", fill="white")
    for i, (row, rgb, error) in enumerate(thumbs):
        x, y = (i % columns) * cell_w, 60 + (i // columns) * cell_h
        draw.text((x+8, y), f"{row['profile']} | seed {row['seed']}", fill="white")
        draw.text((x+8, y+17), f"N={row['resolution']} | voxel={row['voxel_size_m']*1000:.3f} mm", fill="white")
        for j, panel in enumerate((rgb, error)):
            panel.thumbnail((cell_w-16, image_h), Image.Resampling.NEAREST)
            canvas.paste(panel, (x+8, y+42+j*(image_h+5)))
    canvas.save(out / "overview.png")


def _fmt(value, scale=1, places=3):
    return "—" if value is None else f"{value*scale:.{places}f}"


def _markdown(report):
    lines = ["# Reconstruction quality comparison", "", "Keep the scene, cameras, volume bounds and TSDF truncation distance fixed while varying grid resolution and input depth perturbations.",
             "Additional noise and random dropout affect integration frames only. Evaluation uses references supplied by the data source. The table aggregates all held-out views.",
             f"Reference source: {report.get('reference_kind', 'unspecified')}. observed_rgbd denotes observed references, not error-free ground truth.",
             "MAE/RMSE use pixels with valid depth in both images. Coverage and inlier coverage use all valid reference pixels as the denominator.", "",
             "Grid cell count is resolution³, not measured memory usage. Total time includes synthesis, fusion, rendering, evaluation and file export; it is not a pure algorithm frame-rate benchmark.",
             "Noise consists of independent Gaussian Z-depth perturbations, and dropout removes pixels independently. These are not a calibrated model of a physical camera.",
             "The clean profile runs once per resolution; multiple seeds do not provide independent samples for deterministic input.",
             "Single-seed results support preliminary comparisons only, not claims of noise robustness. Multi-seed means and population standard deviations are descriptive statistics, not confidence intervals.",
             "Scene-wide metrics do not establish local door-handle accuracy, and smaller voxels do not guarantee lower error.", "",
             "[RGB and depth-error overview](overview.png)", "",
             "| Profile | Seed | N | Voxel mm | Coverage % | MAE mm | RMSE mm | Inlier coverage % | Missing / extra | Total time s |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["runs"]:
        link = f"[{r['profile']}]({r['comparison']})"
        lines.append(f"| {link} | {r['seed']} | {r['resolution']} | {_fmt(r['voxel_size_m'],1000)} | {_fmt(r['coverage'],100,2)} | {_fmt(r['mae_m'],1000)} | {_fmt(r['rmse_m'],1000)} | {_fmt(r['accurate_reference_fraction'],100,2)} | {r['missing_pixels']} / {r['extra_pixels']} | {_fmt(r['total_seconds'])} |")
    lines += ["", "Per-run configs are in `configs/`, and complete reconstruction outputs are in `runs/`. `summary.csv` stores individual results; `summary.json` also includes grouped statistics and input consistency checks.",
              "The overview shows only the first held-out view. Per-view metrics and complete comparison images are saved in each run directory.", ""]
    return "\n".join(lines)


def run_experiment(config_path, output_dir, progress=None):
    """Run a sequential sweep, preserving every case as a replayable scene YAML."""
    started = perf_counter()
    experiment, base = load_experiment_config(config_path)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"experiment output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir()
    (out / "experiment_source.yaml").write_text(Path(experiment["_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "base_source.yaml").write_text(Path(base["_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    base_saved = {k: v for k, v in base.items() if k not in {"_path", "effective_voxel_size_m"}}
    (out / "base_resolved.yaml").write_text(yaml.safe_dump(base_saved, sort_keys=False), encoding="utf-8")
    resolved = {k: v for k, v in experiment.items() if k != "_path"}
    (out / "experiment_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    cases = []
    for profile in experiment["profiles"]:
        clean = profile["depth_noise_std_m"] == 0 and profile["dropout_probability"] == 0
        for seed in experiment["seeds"][:1] if clean else experiment["seeds"]:
            for resolution in experiment["resolutions"]:
                cases.append((profile, seed, resolution))
    rows, input_digests, reference_digest = [], {}, None
    held_id = next(v["id"] for v in base["views"] if v["split"] == "held_out")
    for index, (profile, seed, resolution) in enumerate(cases):
        case_id = f"{profile['name']}_n{resolution}_s{seed}"
        if progress is not None:
            progress(f"[{index+1}/{len(cases)}] {case_id}")
        cfg = copy.deepcopy(base_saved)
        cfg["reconstruction"]["resolution"] = resolution
        cfg["sensor"] = {k: v for k, v in profile.items() if k != "name"} | {"seed": seed}
        cfg_path = out / "configs" / f"{case_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        run_dir = out / "runs" / case_id
        result = run_reconstruction(cfg_path, run_dir)
        summary = result["summary"]["held_out"]
        inputs_hash = _frame_digest(run_dir, base["views"], "integration")
        refs_hash = _frame_digest(run_dir, base["views"], "held_out")
        key = (profile["name"], seed)
        if key in input_digests and input_digests[key] != inputs_hash:
            raise RuntimeError("integration inputs changed across resolutions for the same profile and seed")
        if reference_digest is not None and reference_digest != refs_hash:
            raise RuntimeError("held-out clean references changed across experiment cases")
        input_digests[key], reference_digest = inputs_hash, refs_hash
        row = {"case_id": case_id, "profile": profile["name"], "seed": seed, "resolution": resolution,
               "voxel_size_m": result["effective_voxel_size_m"], "grid_cells": resolution**3,
               "depth_noise_std_m": profile["depth_noise_std_m"], "dropout_probability": profile["dropout_probability"],
               "depth_tolerance_m": cfg["evaluation"]["depth_tolerance_m"],
               **{k: summary[k] for k in METRICS if k in summary},
               "mesh_triangles": result["mesh_triangles"], "total_seconds": result["total_seconds"],
               "integration_seconds": result["integration_seconds"],
               "run_dir": f"runs/{case_id}", "comparison": f"runs/{case_id}/renders/{held_id}/comparison.png",
               "integration_depth_sha256": inputs_hash, "held_out_depth_sha256": refs_hash}
        rows.append(row)
        # Keep finished case summaries if a later reconstruction fails.
        write_json(out / "completed_runs.json", rows)
    report = {"schema_version": 1, "scene": base["scene"].get("name", "scene"),
              "experiment": resolved, "runs": rows, "groups": aggregate_runs(rows),
              "reference_kind": result["evaluation_reference"],
              "paired_inputs_verified": True, "held_out_references_identical": True,
              "clean_held_out_references_verified": result["evaluation_reference"] in {"clean_synthetic_ground_truth", "simulator_ground_truth"},
              "statistics": "Per-seed descriptive means and population std (ddof=0), not confidence intervals; clean cases use the first seed only.",
              "total_experiment_seconds": perf_counter() - started}
    _overview(out, rows, held_id)
    write_json(out / "summary.json", report)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report
