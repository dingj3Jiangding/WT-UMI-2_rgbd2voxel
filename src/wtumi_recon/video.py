"""Optional streaming H.264 export for observation and reconstruction playback."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np


def require_video_encoder():
    """Fail before an expensive capture if the optional encoder is unavailable."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ImportError("MP4 export requires the video extra: python -m pip install -e '.[video]'") from exc
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        result = subprocess.run([executable, '-hide_banner', '-encoders'], capture_output=True,
                                text=True, check=True, timeout=20)
    except subprocess.SubprocessError as exc:
        raise RuntimeError('FFmpeg encoder preflight failed; check the configured FFmpeg executable') from exc
    if not any('libx264' in line.split() for line in result.stdout.splitlines()):
        raise RuntimeError('MP4 export requires an FFmpeg build containing the libx264 encoder')
    return {'executable': executable, 'imageio_ffmpeg_version': imageio_ffmpeg.__version__,
            'ffmpeg_version': imageio_ffmpeg.get_ffmpeg_version(), 'codec': 'libx264'}


def save_mp4(path, frames, fps):
    """Write one RGB frame at a time, without retaining the animation in RAM."""
    import imageio_ffmpeg

    path = Path(path)
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError('MP4 export requires at least one frame') from exc
    size = first.size
    if any(value % 2 for value in size):
        raise ValueError('H.264 yuv420p output dimensions must be even')
    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=fps, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        quality=None, macro_block_size=1, ffmpeg_log_level='error',
        output_params=['-crf', '18', '-preset', 'medium', '-movflags', '+faststart'],
    )
    count = 0
    try:
        writer.send(None)
        frame = first
        while True:
            if frame.size != size:
                raise ValueError('all video frames must have the same dimensions')
            writer.send(np.ascontiguousarray(np.asarray(frame.convert('RGB')), dtype=np.uint8))
            count += 1
            try:
                frame = next(iterator)
            except StopIteration:
                break
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'encoder did not produce a video: {path}')
    return {'codec': 'H.264', 'pixel_format': 'yuv420p', 'width': size[0], 'height': size[1],
            'fps': fps, 'frame_count': count, 'duration_s': count/fps,
            'imageio_ffmpeg_version': imageio_ffmpeg.__version__}
