"""Run the M1 synthetic-cube RGB-D -> TSDF -> RGB-D round trip."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

from wtumi_recon.config import load_config
from wtumi_recon.metrics import depth_metrics
from wtumi_recon.render import (
    depth_to_preview,
    depth_to_visual_preview,
    render_mesh_rgbd,
)
from wtumi_recon.synthetic import SyntheticCubeScene
from wtumi_recon.tsdf import Open3DTSDFMap, TSDFConfig


def make_rgbd(rgb: np.ndarray, depth_m: np.ndarray) -> o3d.geometry.RGBDImage:
    """Create an Open3D RGBD image from uint8 RGB and metre depth arrays."""

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth_m, dtype=np.float32)),
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )


def run(config_path: Path, output_dir: Path) -> dict[str, float]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 128, 96
    fx = fy = 120.0
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    intrinsics = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    t_wc = np.eye(4, dtype=np.float64)
    t_wc[:3, 3] = [0.5, 0.5, 0.0]

    scene = SyntheticCubeScene()
    rgb_gt, depth_gt = scene.render(intrinsics, t_wc, width, height)
    rgbd = make_rgbd(rgb_gt, depth_gt)

    reconstruction = config["reconstruction"]
    backend = Open3DTSDFMap(
        TSDFConfig(
            backend=reconstruction.get("backend", "uniform"),
            voxel_size_m=float(reconstruction.get("voxel_size_m", 0.01)),
            sdf_trunc_m=float(reconstruction.get("sdf_trunc_m", 0.04)),
            volume_length_m=float(reconstruction.get("volume_length_m", 3.0)),
            resolution=int(reconstruction.get("resolution", 128)),
        )
    )
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    backend.integrate(rgbd, intrinsic_o3d, t_wc)
    mesh = backend.extract_mesh()
    point_cloud = backend.extract_point_cloud()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF extraction returned an empty mesh")

    rgb_pred, depth_pred = render_mesh_rgbd(mesh, intrinsics, t_wc, width, height)
    metrics = depth_metrics(depth_gt, depth_pred)
    metrics.update(
        {
            "scene": config["scene"]["name"],
            "image_width": width,
            "image_height": height,
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "point_cloud_points": len(point_cloud.points),
            "backend": reconstruction.get("backend", "uniform"),
            "effective_voxel_size_m": (
                float(reconstruction.get("volume_length_m", 3.0))
                / int(reconstruction.get("resolution", 128))
                if reconstruction.get("backend", "uniform") == "uniform"
                else float(reconstruction.get("voxel_size_m", 0.01))
            ),
        }
    )

    np.save(output_dir / "depth_ground_truth_m.npy", depth_gt)
    np.save(output_dir / "depth_reconstructed_m.npy", depth_pred)
    o3d.io.write_image(str(output_dir / "rgb_ground_truth.png"), o3d.geometry.Image(rgb_gt))
    o3d.io.write_image(
        str(output_dir / "rgb_reconstructed.png"), o3d.geometry.Image(rgb_pred)
    )
    o3d.io.write_image(
        str(output_dir / "depth_ground_truth_mm.png"),
        o3d.geometry.Image(depth_to_preview(depth_gt, 3.0)),
    )
    o3d.io.write_image(
        str(output_dir / "depth_reconstructed_mm.png"),
        o3d.geometry.Image(depth_to_preview(depth_pred, 3.0)),
    )
    o3d.io.write_image(
        str(output_dir / "depth_ground_truth_preview.png"),
        o3d.geometry.Image(depth_to_visual_preview(depth_gt, 3.0)),
    )
    o3d.io.write_image(
        str(output_dir / "depth_reconstructed_preview.png"),
        o3d.geometry.Image(depth_to_visual_preview(depth_pred, 3.0)),
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(output_dir / "pointcloud.ply"), point_cloud)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(config_path, output_dir / "config_used.yaml")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenes/synthetic_cube.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/m1_synthetic_cube"),
    )
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
