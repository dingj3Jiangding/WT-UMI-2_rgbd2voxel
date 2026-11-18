"""Command line entry point: python -m wtumi_recon run --config ..."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .config import load_run_config
from .pipeline import run_reconstruction


def main():
    parser = argparse.ArgumentParser(description="Static multi-view RGB-D reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--output", type=Path, help="New/empty output directory; defaults to a timestamped outputs folder")
    view = sub.add_parser("view", help="Export an offline interactive 3D viewer from saved results")
    view.add_argument("--result", type=Path, required=True, help="Result directory containing mesh.ply")
    view.add_argument("--output", type=Path, help="HTML path; defaults to <result>/viewer.html")
    view.add_argument("--band-voxels", type=float, default=1.0, help="TSDF surface-band half-width in voxels")
    view.add_argument("--min-weight", type=float, default=1.0, help="Minimum observed integration weight")
    view.add_argument("--max-voxels", type=int, default=100000, help="Maximum displayed surface cells")
    experiment = sub.add_parser("experiment", help="Compare voxel resolutions and synthetic depth noise")
    experiment.add_argument("--config", type=Path, required=True, help="Experiment YAML")
    experiment.add_argument("--output", type=Path, help="New/empty directory; default: timestamped outputs/quality_...")
    capture = sub.add_parser("capture", help="Capture a static MuJoCo room as a portable RGB-D sequence")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True, help="New/empty capture directory")
    plot = sub.add_parser("plot", help="Plot held-out errors against voxel size from an existing experiment")
    plot.add_argument("--summary", type=Path, required=True)
    plot.add_argument("--output", type=Path, help="PNG or SVG path; both formats are written")
    demo = sub.add_parser("demo", help="Capture once, reconstruct at several voxel sizes, export comparisons")
    demo.add_argument("--config", type=Path, required=True)
    demo.add_argument("--output", type=Path, help="New/empty demo output directory")
    occlusion = sub.add_parser("occlusion-demo", help="Compare staircase mapping with head-camera payload occlusion")
    occlusion.add_argument("--config", type=Path, required=True)
    occlusion.add_argument("--output", type=Path, help="New/empty benchmark output directory")
    memory = sub.add_parser("memory-demo", help="Compare visual history before and after carried-box occlusion")
    memory.add_argument("--config", type=Path, required=True)
    memory.add_argument("--output", type=Path, help="New/empty benchmark output directory")
    memory.add_argument("--capture", type=Path, help="Saved paired M6 capture; omit for fresh MuJoCo capture")
    views = sub.add_parser("view-demo", help="Compare fixed and independently scanning head cameras")
    views.add_argument("--config", type=Path, required=True)
    views.add_argument("--output", type=Path, help="New/empty benchmark output directory")
    args = parser.parse_args()
    try:
        if args.command == "view-demo":
            from .view_demo import run_view_demo
            out = args.output or Path("outputs") / f"view_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            report = run_view_demo(args.config,out,progress=lambda message:print(message,flush=True))
            print(json.dumps({"output":str(out.resolve()),"paths":report["paths"],"verification":report["verification"]},indent=2))
            return
        if args.command == "memory-demo":
            from .memory_demo import run_memory_demo
            out = args.output or Path("outputs") / f"memory_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            report = run_memory_demo(args.config,out,capture_dir=args.capture,progress=lambda message:print(message,flush=True))
            print(json.dumps({"output":str(out.resolve()),"paths":report["paths"],
                              "paired_inputs_verified":report["paired_inputs_verified"]},indent=2))
            return
        if args.command == "occlusion-demo":
            from .occlusion_demo import run_occlusion_demo
            out = args.output or Path("outputs") / f"occlusion_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            report = run_occlusion_demo(args.config, out, progress=lambda message: print(message, flush=True))
            print(json.dumps({"output": str(out.resolve()), "paths": report["paths"],
                              "paired_inputs_verified": report["paired_inputs_verified"]}, indent=2))
            return
        if args.command == "capture":
            from .mujoco_capture import capture_mujoco
            manifest = capture_mujoco(args.config, args.output)
            print(json.dumps({"manifest": str(manifest)}, indent=2))
            return
        if args.command == "plot":
            from .experiment_plots import save_experiment_plot
            png, svg = save_experiment_plot(args.summary, args.output)
            print(json.dumps({"png": str(png), "svg": str(svg)}, indent=2))
            return
        if args.command == "demo":
            from .demo import run_sim_demo
            out = args.output or Path("outputs") / f"sim_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            report = run_sim_demo(args.config, out, progress=lambda message: print(message, flush=True))
            print(json.dumps({"output": str(out.resolve()), "demo": report}, indent=2))
            return
        if args.command == "experiment":
            from .experiments import run_experiment
            out = args.output or Path("outputs") / f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            report = run_experiment(args.config, out, progress=lambda message: print(message, flush=True))
            print(json.dumps({"output": str(out.resolve()), "cases": len(report["runs"]),
                              "report": str((out / "report.md").resolve()),
                              "paired_inputs_verified": report["paired_inputs_verified"]}, indent=2))
            return
        if args.command == "view":
            from .viewer import export_viewer
            path, metadata = export_viewer(args.result, args.output, band_voxels=args.band_voxels,
                                           min_weight=args.min_weight, max_voxels=args.max_voxels)
            print(json.dumps({"viewer": str(path), "surface_voxels": metadata["surface_voxels"],
                              "voxel_size_m": metadata["voxel_size_m"]}, indent=2))
            return
        if args.command == "validate":
            cfg = load_run_config(args.config)
            print(f"Valid: {len(cfg['views'])} views; voxel size {cfg['effective_voxel_size_m']:.6f} m")
            return
        out = args.output or Path("outputs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        report = run_reconstruction(args.config, out)
        print(json.dumps({"output": str(out.resolve()), "integrated_frame_ids": report["integrated_frame_ids"], "held_out_frame_ids": report["held_out_frame_ids"], "summary": report["summary"]}, indent=2, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Configuration/output error: {exc}\n")


if __name__ == "__main__":
    main()
