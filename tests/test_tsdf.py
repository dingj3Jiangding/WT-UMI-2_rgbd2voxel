from wtumi_recon.tsdf import Open3DTSDFMap, TSDFConfig


def test_tsdf_backend_initializes():
    backend = Open3DTSDFMap(TSDFConfig(voxel_size_m=0.01, sdf_trunc_m=0.04))
    assert backend.config.voxel_size_m == 0.01
