"""Fresh capture schedules for scene-suite playback presets.

Only authored camera/rig waypoints are interpolated. RGB-D images and cumulative
maps are captured and reconstructed separately at every generated timestamp.
"""

from __future__ import annotations

import copy

import numpy as np


PRESETS = ('default', 'pace30')


def validate_preset(preset):
    if preset not in PRESETS:
        raise ValueError(f'Unknown scene-suite preset: {preset}; choose default or pace30')


def resample_scene_config(config, *, fps=30, width=640, height=480):
    """Return a new configuration with a uniform, endpoint-inclusive schedule.

    Position and head angles are piecewise linear. Body yaw follows the shortest
    angular arc between neighboring source poses. Robot, door, evaluation poses
    and voxel geometry are retained; the source configuration is never changed.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError('capture fps must be finite and in (0, 120]')
    for name, value in (('width', width), ('height', height)):
        if type(value) is not int or not 8 <= value <= 2048:
            raise ValueError(f'camera {name} must be an integer in [8, 2048]')
    original = config['trajectory']
    if len(original) < 2:
        raise ValueError('capture resampling requires at least two source poses')
    timestamps = np.array([row['timestamp_s'] for row in original], dtype=float)
    if (not np.isfinite(timestamps).all() or timestamps[0] < 0
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('source pose timestamps must be finite, nonnegative and increasing')
    intervals = (timestamps[-1] - timestamps[0]) * fps
    if not np.isclose(intervals, round(intervals), atol=1e-7, rtol=0):
        raise ValueError('trajectory duration must contain an integer number of capture intervals')
    count = int(round(intervals)) + 1
    if not 2 <= count <= 240:
        raise ValueError('resampled trajectory must contain between 2 and 240 views')
    if any(float(frame['timestamp_s']) <= timestamps[-1] for frame in config['held_out']):
        raise ValueError('held-out timestamps must follow the capture trajectory')

    result = copy.deepcopy({key: value for key, value in config.items() if not key.startswith('_')})
    target_times = timestamps[0] + np.arange(count, dtype=float) / fps
    target_times[-1] = timestamps[-1]
    positions = np.array([row['position_m'] for row in original], dtype=float)
    if positions.shape != (len(original), 3) or not np.isfinite(positions).all():
        raise ValueError('source positions must be finite 3D vectors')
    interpolated = np.column_stack([np.interp(target_times, timestamps, positions[:, axis])
                                   for axis in range(3)])
    yaw = np.unwrap(np.deg2rad([row['yaw_degrees'] for row in original]))
    yaw_values = np.rad2deg(np.interp(target_times, timestamps, yaw))
    head = {key: np.interp(target_times, timestamps, [row.get(key, 0.) for row in original])
            for key in ('head_yaw_degrees', 'head_pitch_degrees')}
    if not np.isfinite(yaw_values).all() or any(not np.isfinite(values).all() for values in head.values()):
        raise ValueError('source angles must be finite')
    held_ids = {row['id'] for row in config['held_out']}
    frames = []
    for index, timestamp in enumerate(target_times):
        name = f'capture_{index:04d}'
        if name in held_ids:
            raise ValueError('generated capture IDs conflict with held-out IDs')
        frames.append({'id': name, 'timestamp_s': float(timestamp),
                       'position_m': interpolated[index].tolist(),
                       'yaw_degrees': float(yaw_values[index]),
                       **{key: float(values[index]) for key, values in head.items()}})
    result['trajectory'] = frames
    old_width = config['camera']['width']
    result['camera'].update(width=width, height=height)
    result['occupancy']['pixel_stride'] = max(1, int(round(config['occupancy']['pixel_stride'] * width / old_width)))
    result['playback']['fps'] = float(fps)
    return result


def preset_config(config, preset):
    validate_preset(preset)
    if preset == 'pace30':
        return resample_scene_config(config)
    return copy.deepcopy({key: value for key, value in config.items() if not key.startswith('_')})


def describe_preset(configs, preset):
    """Describe already validated source configurations without creating files."""
    validate_preset(preset)
    rows = []
    for name, original in configs.items():
        config = preset_config(original, preset)
        trajectory = config['trajectory']
        fps = config['playback']['fps']
        rows.append({'scene': name, 'source_waypoints': len(original['trajectory']),
                     'box_enabled': config['box']['enabled'],
                     'robot_pose': (config.get('robot') or {}).get('pose'),
                     'integration_frames': len(trajectory), 'held_out_views': len(config['held_out']),
                     'capture_start_s': trajectory[0]['timestamp_s'],
                     'capture_end_s': trajectory[-1]['timestamp_s'],
                     'playback_fps': fps, 'video_duration_s': len(trajectory) / fps,
                     'camera_width': config['camera']['width'], 'camera_height': config['camera']['height'],
                     'occupancy_pixel_stride': config['occupancy']['pixel_stride'],
                     'tsdf_resolution': config['reconstruction']['resolution'],
                     'voxel_size_mm': config['reconstruction']['volume_length_m'] / config['reconstruction']['resolution'] * 1000})
    return {'format': 'wtumi-scene-suite-preset-v1', 'preset': preset, 'scenes': rows,
            'requires_mp4_encoder': preset == 'pace30',
            'sampling': ('Piecewise-linear positions and head angles; shortest-arc body yaw. '
                         'Fresh RGB-D and map integration at every sampled pose; no image or map interpolation.'
                         if preset == 'pace30' else 'Original scene capture waypoints and playback settings.'),
            'scope': 'Playback frame rate is independent of measured rendering or reconstruction throughput.'}
