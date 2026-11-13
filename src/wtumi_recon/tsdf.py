"""Open3D TSDF backend with an explicit T_wc to T_cw boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class TSDFConfig:
    backend: str = "uniform"
    voxel_size_m: float = 0.01
    sdf_trunc_m: float = 0.04
    volume_length_m: float = 3.0
    resolution: int = 128
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def effective_voxel_size_m(self) -> float:
        return self.volume_length_m / self.resolution if self.backend == "uniform" else self.voxel_size_m

    def __post_init__(self) -> None:
        if self.backend not in {"uniform", "scalable"}:
            raise ValueError("backend must be uniform or scalable")
        for name in ("voxel_size_m", "sdf_trunc_m", "volume_length_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.resolution, int) or self.resolution < 2:
            raise ValueError("resolution must be an integer >= 2")
        if np.asarray(self.origin_m).shape != (3,) or not np.isfinite(self.origin_m).all():
            raise ValueError("origin_m must contain three finite values")


class Open3DTSDFMap:
    """Wrapper around Open3D's uniform or scalable TSDF volume.

    The public project convention is ``T_wc`` (camera to world). Open3D's
    integration API consumes the inverse world-to-camera extrinsic, so the
    conversion is kept inside this backend.
    """

    def __init__(self, config: TSDFConfig | None = None) -> None:
        config = config or TSDFConfig()
        if config.backend not in {"uniform", "scalable"}:
            raise ValueError("backend must be 'uniform' or 'scalable'")
        self.config = config
        if config.backend == "uniform":
            self._volume = o3d.pipelines.integration.UniformTSDFVolume(
                length=config.volume_length_m,
                resolution=config.resolution,
                sdf_trunc=config.sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=np.asarray(config.origin_m, dtype=np.float64),
            )
        else:
            self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=config.voxel_size_m,
                sdf_trunc=config.sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

    def integrate(
        self,
        rgbd: o3d.geometry.RGBDImage,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        t_wc: np.ndarray,
    ) -> None:
        """Integrate one RGB-D frame using a camera-to-world pose."""

        t_wc = np.asarray(t_wc, dtype=np.float64)
        if t_wc.shape != (4, 4):
            raise ValueError("t_wc must have shape (4, 4)")
        t_cw = np.linalg.inv(t_wc)
        self._volume.integrate(rgbd, intrinsics, t_cw)

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self._volume.extract_point_cloud()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self._volume.extract_triangle_mesh()

    def extract_voxel_point_cloud(self) -> o3d.geometry.PointCloud:
        return self._volume.extract_voxel_point_cloud()

    def save_volume(self, path: str | Path) -> None:
        """Export uniform TSDF, weight and color arrays in Open3D flat order.

        TSDF is normalized by truncation. Weight zero means unobserved;
        this is not a complete free/occupied map. Colors retain Open3D's scale.
        """
        if self.config.backend != "uniform":
            raise NotImplementedError("ScalableTSDFVolume has no public volume export API")
        tsdf_weight = np.asarray(self._volume.extract_volume_tsdf())
        color = np.asarray(self._volume.extract_volume_color())
        np.savez_compressed(
            path, tsdf=tsdf_weight[:, 0].astype(np.float32),
            weight=tsdf_weight[:, 1].astype(np.float32), color=color.astype(np.float32),
            origin_m=self.config.origin_m, resolution=self.config.resolution,
            voxel_size_m=self.config.effective_voxel_size_m,
            sdf_trunc_m=self.config.sdf_trunc_m,
            color_scale=255.0, tsdf_normalized=True, axis_order="xyz", flat_order="C",
        )
