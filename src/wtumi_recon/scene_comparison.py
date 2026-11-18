"""Synchronize recorded head-camera RGB with its cumulative voxel-map evidence.

The right panel is an offscreen MuJoCo rendering of saved TSDF cells, not the
original scene. No environment model is loaded and no physics is advanced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene_viewer import _inside, _pose, load_display_voxels


def comparison_inputs(result_dir):
    """Validate one-to-one observation/prefix alignment before exporting."""
    from .scene_bundle import load_scene_config

    root = Path(result_dir).expanduser().resolve()
    config = load_scene_config(root / 'scene/scene.yaml')
    capture = json.loads((root / 'capture/capture.json').read_text())
    history = json.loads((root / 'history.json').read_text())
    observations = capture['frames']
    ids = [row['id'] for row in observations]
    if (not observations or len(ids) != len(set(ids)) or len(history) != len(observations)
            or ids != capture['integration_frame_ids']
            or set(ids) & set(capture['held_out_frame_ids'])):
        raise ValueError('comparison requires one unique saved prefix per integration observation')
    records, previous = [], -np.inf
    for index, (observation, prefix) in enumerate(zip(observations, history)):
        timestamp = float(observation['timestamp_s'])
        if (prefix['id'] != observation['id'] or prefix['frames'] != index + 1
                or not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous
                or not np.isclose(prefix['timestamp_s'], timestamp, atol=1e-8, rtol=0)):
            raise ValueError('comparison frame IDs, timestamps and observation counts must match')
        previous = timestamp
        directory = _inside(root / 'capture', observation['frame_directory'], directory=True)
        raw = _inside(directory, 'raw_rgb.png')
        run = _inside(root, prefix['run_dir'], directory=True)
        metrics = json.loads((run / 'metrics.json').read_text())
        if metrics['integrated_frame_ids'] != ids[:index + 1]:
            raise ValueError('comparison prefix contains missing, reordered or future observations')
        voxels = _inside(run, 'tsdf_volume.npz')
        transform = _pose(observation['T_wc'], 'captured camera pose')
        records.append({'id': observation['id'], 'frames': index + 1,
                        'timestamp_s': timestamp, 'raw_rgb': raw,
                        'volume': voxels, 'T_wc': transform})
    return root, config, records


def overview_camera(config):
    """Use authored bounds only for framing; their geometry is never rendered."""
    points = []
    for obj in config['environment']['objects']:
        position = np.asarray(obj['position_m'], dtype=float)
        if obj['type'] == 'box':
            # A bounding sphere also encloses boxes with nonzero Euler angles.
            if np.any(np.asarray(obj.get('euler_degrees', [0, 0, 0]))):
                half = np.full(3, np.linalg.norm(obj['size_m']) / 2)
            else:
                half = np.asarray(obj['size_m']) / 2
        elif obj['type'] == 'sphere':
            half = np.full(3, obj['radius_m'])
        elif obj['type'] == 'cylinder':
            half = np.full(3, max(obj['radius_m'], obj['height_m'] / 2))
        else:
            # Generic meshes need no source-geometry load in this exporter.
            # Use configured map bounds for framing their possible extent.
            half = np.full(3, config['reconstruction']['volume_length_m'] / 2)
            position = np.asarray(config['reconstruction']['origin_m']) + half
        points.extend([position - half, position + half])
    points.extend(np.asarray(frame['position_m']) + config['camera']['position_in_rig_m']
                  for frame in config['trajectory'])
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    span = max(float(np.max(upper - lower)), 1.)
    target = (lower + upper) / 2
    target[2] = lower[2] + .38 * (upper[2] - lower[2])
    return {'lookat': target.tolist(), 'distance_m': 1.85 * span,
            'azimuth_degrees': 125., 'elevation_degrees': -60., 'extent_m': span,
            'framing_source': 'Authored scene bounds and camera path; source geometry is not rendered'}


def make_voxel_renderer(config, width, height, max_voxels):
    """Create an empty MuJoCo world with capacity for evidence cubes only."""
    import mujoco

    camera_info = overview_camera(config)
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><statistic extent="{camera_info["extent_m"]}"/>'
        f'<visual><global offwidth="{width}" offheight="{height}"/>'
        '<quality offsamples="0"/><headlight ambient="0.8 0.8 0.8" '
        'diffuse="0.2 0.2 0.2" specular="0 0 0"/></visual><worldbody/></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = camera_info['lookat']
    camera.distance = camera_info['distance_m']
    camera.azimuth, camera.elevation = camera_info['azimuth_degrees'], camera_info['elevation_degrees']
    renderer = mujoco.Renderer(model, width=width, height=height, max_geom=max_voxels + 2)
    return renderer, model, data, camera, camera_info


def render_voxel_panel(renderer, model, data, camera, voxels, camera_pose):
    """Render actual saved cell centers/colors plus a camera reference marker."""
    import mujoco

    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    # The blank model has no geoms. Clearing also makes that invariant explicit.
    scene.ngeom = 0
    rotation = np.eye(3).reshape(9)
    size = np.full(3, voxels.size_m * .47)
    for index, (center, color) in enumerate(zip(voxels.centers, voxels.colors)):
        mujoco.mjv_initGeom(scene.geoms[index], mujoco.mjtGeom.mjGEOM_BOX,
                          size, center, rotation, np.r_[color, 1.].astype(np.float32))
    count = len(voxels.centers)
    origin = camera_pose[:3, 3]
    marker_radius = max(.025, .7 * voxels.size_m)
    mujoco.mjv_initGeom(scene.geoms[count], mujoco.mjtGeom.mjGEOM_SPHERE,
                      np.full(3, marker_radius), origin, rotation,
                      np.array([1., .8, .05, 1.], dtype=np.float32))
    mujoco.mjv_initGeom(scene.geoms[count + 1], mujoco.mjtGeom.mjGEOM_ARROW,
                      np.zeros(3), np.zeros(3), rotation,
                      np.array([1., .8, .05, 1.], dtype=np.float32))
    mujoco.mjv_connector(scene.geoms[count + 1], mujoco.mjtGeom.mjGEOM_ARROW,
                        marker_radius * .6, origin, origin + camera_pose[:3, 2] * .4)
    scene.ngeom = count + 2
    return Image.fromarray(renderer.render().copy())


def _font(size):
    # Pillow's bundled font is available on both Mac and PACE.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # The size argument was added in Pillow 10.1; 10.0 is also supported.
        return ImageFont.load_default()


def comparison_panel(raw, voxel_image, record, voxels, scene_name):
    """Label the different viewpoints and any deterministic display sampling."""
    width, height = voxel_image.size
    header, footer = 76, 66
    panel = Image.new('RGB', (width * 2, height + header + footer), '#111827')
    # Enlarge while preserving the original aspect ratio, with letterboxing.
    scale = min(width / raw.width, height / raw.height)
    fitted = raw.resize((round(raw.width * scale), round(raw.height * scale)), Image.Resampling.LANCZOS)
    panel.paste(fitted, ((width - fitted.width) // 2, header + (height - fitted.height) // 2))
    panel.paste(voxel_image, (width, header))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), f"{scene_name} | frame {record['frames']} | {record['id']} | t={record['timestamp_s']:.3f} s",
              fill='white', font=_font(19))
    draw.text((12, 38), 'Recorded head-camera RGB', fill='white', font=_font(19))
    draw.text((width + 12, 38), 'Cumulative voxel map | MuJoCo overview', fill='white', font=_font(19))
    draw.text((12, header + height + 8), f'Input: {raw.width} x {raw.height}; resized for display',
              fill='#cbd5e1', font=_font(15))
    draw.text((width + 12, header + height + 8),
              f'Displayed / selected cells: {len(voxels.centers):,} / {voxels.selected_count:,}',
              fill='#cbd5e1', font=_font(15))
    draw.text((12, header + height + 34),
              'Right: observed TSDF surface cells only; yellow arrow = recorded camera. Unknown space is not filled.',
              fill='#cbd5e1', font=_font(15))
    return panel


def export_camera_voxel_comparison(result_dir, *, panel_width=640, panel_height=480,
                                   max_voxels=50000, export_mp4=True):
    """Export exact saved observations paired with their corresponding maps.

    The MP4 and individual PNGs retain every observation. The GIF samples at
    most 60 frames for file size, with durations adjusted to preserve playback.
    No interpolation is performed and display FPS is not mapping throughput.
    """
    from .video import require_video_encoder, save_mp4

    if (type(panel_width) is not int or type(panel_height) is not int
            or min(panel_width, panel_height) < 64 or max(panel_width, panel_height) > 2048
            or panel_width % 2 or panel_height % 2):
        raise ValueError('comparison panel dimensions must be even integers in [64, 2048]')
    if type(max_voxels) is not int or not 1 <= max_voxels <= 200000:
        raise ValueError('comparison max_voxels must be an integer in [1, 200000]')
    root, config, inputs = comparison_inputs(result_dir)
    paths = {'gif': 'camera_voxel.gif', 'png': 'camera_voxel.png',
             'metadata': 'camera_voxel.json', 'frames': 'camera_voxel_frames'}
    if any((root / value).exists() for value in [*paths.values(), 'camera_voxel.mp4']):
        raise FileExistsError('camera-voxel comparison outputs already exist')
    video_status = {'status': 'disabled'}
    if export_mp4:
        try:
            encoder = require_video_encoder()
            video_status = {'status': 'ready', 'codec': encoder['codec']}
        except (ImportError, RuntimeError, OSError) as exc:
            video_status = {'status': 'unavailable', 'reason': str(exc)}
    renderer, model, data, camera, camera_info = make_voxel_renderer(
        config, panel_width, panel_height, max_voxels)
    rows, frame_paths = [], []
    try:
        (root / paths['frames']).mkdir()
        for record in inputs:
            voxels = load_display_voxels(record['volume'], max_voxels=max_voxels)
            right = render_voxel_panel(renderer, model, data, camera, voxels, record['T_wc'])
            with Image.open(record['raw_rgb']) as source:
                raw = source.convert('RGB')
                panel = comparison_panel(raw, right, record, voxels, config['scene_name'])
            relative = f"{paths['frames']}/frame_{record['frames']:03d}.png"
            panel.save(root / relative)
            frame_paths.append(root / relative)
            rows.append({'id': record['id'], 'observations': record['frames'],
                         'timestamp_s': record['timestamp_s'], 'image': relative,
                         'raw_rgb': record['raw_rgb'].relative_to(root).as_posix(),
                         'tsdf_volume': record['volume'].relative_to(root).as_posix(),
                         'T_wc': record['T_wc'].tolist(),
                         'displayed_cells': len(voxels.centers),
                         'selected_surface_cells': voxels.selected_count,
                         'observed_cells': voxels.observed_count,
                         'voxel_size_m': voxels.size_m})
        panel.save(root / paths['png'])
    finally:
        renderer.close()
    count, fps = len(rows), config['playback']['fps']
    indices = list(range(count)) if count <= 60 else [round(i * (count - 1) / 59) for i in range(60)]
    gif_frames = []
    for index in indices:
        with Image.open(frame_paths[index]) as source:
            reduced = source.convert('RGB')
            reduced.thumbnail((960, 700), Image.Resampling.LANCZOS)
            gif_frames.append(reduced)
    durations = [max(10, round(1000 * ((indices[i + 1] if i + 1 < len(indices) else count) - index) / fps))
                 for i, index in enumerate(indices)]
    gif_frames[0].save(root / paths['gif'], save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, disposal=2)
    if video_status['status'] == 'ready':
        def images():
            for path in frame_paths:
                with Image.open(path) as source:
                    yield source.convert('RGB')
        video_status = {'status': 'created', **save_mp4(root / 'camera_voxel.mp4', images(), fps)}
        paths['mp4'] = 'camera_voxel.mp4'
    report = {
        'format': 'wtumi-camera-voxel-comparison-v1', 'scene_name': config['scene_name'],
        'paths': paths, 'frame_count': count, 'playback_fps': fps,
        'gif_frame_indices': indices, 'gif_durations_ms': durations,
        'panel_width': panel_width, 'panel_height': panel_height,
        'overview_camera': camera_info, 'display_max_voxels': max_voxels,
        'sampling': 'Deterministic evenly spaced selected flat indices when the display cap is exceeded; saved map unchanged',
        'surface_filter': 'finite TSDF, integration weight > 0, abs(normalized TSDF) * truncation <= voxel size',
        'left_panel': 'Recorded raw RGB from the robot head camera; robot/payload may be visible',
        'right_panel': 'MuJoCo rendering of actual saved cumulative voxel cells; original scene geometry is not loaded',
        'camera_marker': 'Yellow sphere and arrow denote the recorded T_wc origin and optical axis; they are reference annotations, not reconstructed cells',
        'alignment': 'Each left observation matches the right prefix through that observation; held-out images and future observations are excluded',
        'playback_scope': 'Exact saved frame sequence at configured playback FPS, without interpolation; not reconstruction throughput',
        'video': video_status, 'frames': rows,
    }
    (root / paths['metadata']).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
