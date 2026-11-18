"""Optional real H.264 round trip: dimensions, timing, ordering and RGB content."""

import numpy as np
from PIL import Image
import pytest

from wtumi_recon.video import require_video_encoder, save_mp4


def test_h264_roundtrip_preserves_frame_count_dimensions_and_order(tmp_path):
    ffmpeg = pytest.importorskip('imageio_ffmpeg')
    require_video_encoder()
    colors = [(225, 25, 40), (25, 215, 40), (30, 40, 225)]
    frames = (Image.new('RGB', (64, 48), color) for color in colors)
    path = tmp_path/'actual.mp4'
    info = save_mp4(path, frames, 15)
    reader = ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    metadata = next(reader)
    decoded = [np.frombuffer(raw, np.uint8).reshape(48, 64, 3) for raw in reader]
    assert metadata['size'] == (64, 48)
    assert metadata['fps'] == 15
    assert len(decoded) == info['frame_count'] == 3
    assert info['duration_s'] == pytest.approx(.2)
    for actual, expected in zip(decoded, colors):
        np.testing.assert_allclose(actual.mean(axis=(0, 1)), expected, atol=4)
