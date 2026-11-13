"""Headless descriptive plots of held-out error against physical voxel size."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path

import numpy as np


PLOT_METRICS = (
    ("mae_m", 1000.0, "Depth MAE (mm)"),
    ("rmse_m", 1000.0, "Depth RMSE (mm)"),
    ("coverage", 100.0, "Coverage (%)"),
)


def prepare_plot_series(summary):
    """Use per-run data to retain nulls, individual seeds and sorted metric axes.

    Mean lines use available runs at each size; an all-null size stays NaN and
    breaks the line. These are descriptive seed means, not pooled pixels or CIs.
    """
    rows = summary.get("runs") if isinstance(summary, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("experiment summary must contain a nonempty runs list")
    profiles = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each experiment run must be a mapping")
        profile, seed, size = row.get("profile"), row.get("seed"), row.get("voxel_size_m")
        if not isinstance(profile, str) or not profile:
            raise ValueError("run profile must be a nonempty string")
        if type(seed) is not int or seed < 0:
            raise ValueError("run seed must be a nonnegative integer")
        if isinstance(size, bool) or not isinstance(size, Real) or not np.isfinite(size) or size <= 0:
            raise ValueError("voxel_size_m must be finite and positive")
        key = (profile, float(size), seed)
        if key in seen:
            raise ValueError("duplicate profile, voxel size and seed in experiment runs")
        seen.add(key)
        cleaned = {"seed": seed, "voxel_size_m": float(size)}
        for metric, _, _ in PLOT_METRICS:
            if metric not in row:
                raise ValueError(f"run is missing {metric}")
            value = row[metric]
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not np.isfinite(value) or value < 0
                                      or (metric == "coverage" and value > 1)):
                raise ValueError(f"invalid run metric: {metric}")
            cleaned[metric] = float(value) if value is not None else None
        profiles.setdefault(profile, []).append(cleaned)
    output = []
    for profile, members in profiles.items():
        sizes = sorted({r["voxel_size_m"] for r in members})
        metrics = {}
        for metric, scale, _ in PLOT_METRICS:
            means, counts, individual_x, individual_y = [], [], [], []
            for size in sizes:
                values = [r[metric] for r in members if r["voxel_size_m"] == size and r[metric] is not None]
                means.append(float(np.mean(values))*scale if values else np.nan)
                counts.append(len(values))
                individual_x.extend([size*1000]*len(values))
                individual_y.extend([v*scale for v in values])
            metrics[metric] = {"mean": np.asarray(means), "valid_runs": counts,
                               "individual_x_mm": individual_x, "individual_y": individual_y}
        output.append({"profile": profile, "voxel_size_mm": np.asarray(sizes)*1000,
                       "seeds": sorted({r["seed"] for r in members}),
                       "runs_per_size": [sum(r["voxel_size_m"] == size for r in members) for size in sizes],
                       "metrics": metrics})
    return output


def save_experiment_plot(summary_path, output_path=None):
    """Write a PNG and same-stem SVG; does not rerun experiments or modify inputs.

    output_path may end in .png or .svg; both formats are always exported.
    Returns (png_path, svg_path).
    """
    summary_path = Path(summary_path).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    series = prepare_plot_series(summary)
    output = Path(output_path).expanduser().resolve() if output_path else summary_path.parent / "error_vs_voxel_size.png"
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("plot output must end in .png or .svg")
    png_path, svg_path = output.with_suffix(".png"), output.with_suffix(".svg")
    # Agg has no window/display/GPU dependency. It is selected before pyplot.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), sharex=True)
        palette = plt.get_cmap("tab10")
        for i, item in enumerate(series):
            color = palette(i % 10)
            low, high = min(item["runs_per_size"]), max(item["runs_per_size"])
            count_text = str(low) if low == high else f"{low}-{high}"
            label = f"{item['profile']} (runs/size: {count_text}; seeds: {','.join(map(str,item['seeds']))})"
            for ax, (metric, _, ylabel) in zip(axes, PLOT_METRICS):
                data = item["metrics"][metric]
                ax.scatter(data["individual_x_mm"], data["individual_y"], color=color, alpha=.28, s=28, zorder=2)
                ax.plot(item["voxel_size_mm"], data["mean"], color=color, linewidth=1.8,
                        marker="o", markersize=4, label=label, zorder=3)
        for ax, (_, _, ylabel) in zip(axes, PLOT_METRICS):
            ax.set_xlabel("Voxel edge length (mm)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
        axes[2].set_ylim(0, 102)
        title = fig.suptitle(f"Held-out reconstruction: {summary.get('scene', 'scene')}", y=.97, fontsize=14)
        title.set_parse_math(False)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91),
                            ncol=min(3, len(series)), frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_parse_math(False)
        fig.text(.5, .055, "MAE/RMSE: valid-depth overlap only. Coverage: overlap / all valid reference depth pixels.",
                 ha="center", fontsize=9)
        fig.text(.5, .018, "Faint dots: individual runs. Lines: available-run means. Missing values stay gaps; no confidence intervals.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.065, right=.985, top=.72, bottom=.2, wspace=.32)
        try:
            fig.savefig(png_path, dpi=180)
            fig.savefig(svg_path)
        finally:
            plt.close(fig)
    return png_path, svg_path
