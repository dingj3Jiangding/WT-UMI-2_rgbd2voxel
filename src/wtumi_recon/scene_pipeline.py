"""Single-scene RGB-D capture, cumulative reconstruction and portable playback."""

from __future__ import annotations

import html
import os
from pathlib import Path
from time import perf_counter

import yaml
from PIL import Image, ImageDraw


def _write_config(path, manifest, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        'schema_version': 3,
        'source': {'type': 'rgbd_sequence', 'manifest': os.path.relpath(manifest, path.parent)},
        'reconstruction': cfg['reconstruction'], 'occupancy': cfg['occupancy'],
        'evaluation': cfg['evaluation'],
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')


def _export_gallery(out, cfg, history, capture, comparison=None):
    """Small shareable previews; native MuJoCo remains the full 3D playback UI."""
    inputs, reconstruction = [], []
    view_id = cfg['held_out'][0]['id']
    count = len(history)
    indices = list(range(count)) if count <= 60 else [round(i * (count - 1) / 59) for i in range(60)]
    for index in indices:
        row, observation = history[index], capture['frames'][index]
        directory = out / 'capture' / observation['frame_directory']
        with Image.open(directory / 'raw_rgb.png') as raw, Image.open(directory / 'rgb.png') as masked:
            raw, masked = raw.convert('RGB'), masked.convert('RGB')
            raw.thumbnail((480, 320))
            masked.thumbnail((480, 320))
            width, height = raw.size
            panel = Image.new('RGB', (width * 2, height + 48), '#162030')
            panel.paste(raw.convert('RGB'), (0, 48))
            panel.paste(masked.convert('RGB'), (width, 48))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 6), f"Frame {row['frames']} | t={row['timestamp_s']:g} s", fill='white')
            draw.text((8, 28), 'Raw camera RGB', fill='white')
            draw.text((width + 8, 28), 'RGB used for mapping (self/payload excluded)', fill='white')
            inputs.append(panel)
        path = out / row['run_dir'] / 'renders' / view_id / 'comparison.png'
        with Image.open(path) as source:
            frame = source.convert('RGB')
            frame.thumbnail((1000, 650))
            panel = Image.new('RGB', (frame.width, frame.height + 30), 'white')
            panel.paste(frame, (0, 30))
            ImageDraw.Draw(panel).text((10, 8), f"Cumulative map after {row['frames']} observation(s)", fill='black')
            reconstruction.append(panel)
    # Keep sampled previews bounded while preserving their configured timeline.
    durations = [max(10, round(1000 * ((indices[i + 1] if i + 1 < len(indices) else count) - index)
                              / cfg['playback']['fps'])) for i, index in enumerate(indices)]
    for filename, frames in (('inputs.gif', inputs), ('reconstruction.gif', reconstruction)):
        frames[0].save(out / filename, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, disposal=2)
    title = html.escape(cfg['scene_name'])
    comparison_section = ''
    if comparison is not None:
        paths = comparison['paths']
        if paths.get('mp4'):
            media = f'<video controls loop playsinline preload="metadata" poster="{html.escape(paths["png"], quote=True)}" src="{html.escape(paths["mp4"], quote=True)}"></video>'
        else:
            media = f'<img src="{html.escape(paths["gif"], quote=True)}" alt="Camera RGB and cumulative voxel reconstruction">'
        comparison_section = ('<h2>Camera view and cumulative voxel reconstruction</h2>' + media +
            '<p>Left: recorded head-camera RGB. Right: saved observed surface voxels rendered in MuJoCo from a fixed overview. '
            'The two panels use different viewpoints and the same observation index. '
            'No original scene surfaces are shown on the right; any display sampling is labeled in the animation.</p>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#eef1f6;color:#182337}}
img,video{{max-width:100%;background:white;border-radius:8px}}a{{color:#1650ac}}</style>
<h1>{title}</h1><p>Recorded RGB-D to voxel reconstruction. Each step includes only observations captured up to that step.
This is saved playback, not live reconstruction or measured real-time performance.</p>
<p><a href="reconstruction/viewer.html">Final interactive 3D map</a> · <a href="scene_run.json">Run report</a> ·
<a href="history.json">Per-frame metrics</a> · <a href="scene/scene.yaml">Scene configuration</a></p>
<p>Preview shows {len(indices)} of {count} saved observations at reduced image size. The native viewer retains every saved step.</p>
{comparison_section}
<h2>Camera observations</h2><img src="inputs.gif" alt="Raw and masked camera observations">
<h2>Reconstruction from an evaluation-only viewpoint</h2><img src="reconstruction.gif" alt="Ground truth and cumulative reconstructed RGB-D">
<p>Held-out images are never fused. Missing measurements remain unknown; box and robot pixels are excluded.
Depth error is evaluated only where both reference and reconstructed depth exist. Coverage is not geometric accuracy.</p>
<p>For the original scene, reconstructed meshes and observed TSDF surface voxels in MuJoCo, use
<code>python scripts/scene.py view --result PATH_TO_THIS_DIRECTORY</code>.</p></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')


def run_scene(scene_ref, output_dir, progress=None):
    """Capture one scene and fuse each observation once, preserving every prefix."""
    from .scene_bundle import build_scene, load_scene_config
    from .scene_capture import capture_scene
    from .occlusion_demo import write_prefix_manifest
    from .pipeline import ReconstructionState, run_reconstruction, write_json
    from .viewer import export_viewer

    started = perf_counter()
    # Validate the source before creating any outputs.
    load_scene_config(scene_ref)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'scene run output must be new or empty: {out}')
    if progress:
        progress('Building the configured MuJoCo scene')
    built = build_scene(scene_ref, out / 'scene')
    cfg = load_scene_config(out / 'scene' / built['paths']['config'])
    capture = capture_scene(cfg, out / 'scene' / built['paths']['model'], out / 'capture', progress)
    manifest = out / 'capture' / capture['paths']['manifest']
    state, history = ReconstructionState(), []
    count = len(capture['frames'])
    for index, frame in enumerate(capture['frames'], start=1):
        if progress:
            progress(f'Reconstructing observation {index}/{count}: {frame["id"]}')
        final = index == count
        prefix = manifest if final else write_prefix_manifest(manifest, index)
        config_path = out / 'configs' / f'prefix_{index:03d}.yaml'
        _write_config(config_path, prefix, cfg)
        run_dir = 'reconstruction' if final else f'history/frame_{index:03d}'
        report = run_reconstruction(config_path, out / run_dir, prefix_state=state,
                                    compact=not final, snapshot_volume=True)
        history.append({
            'id': frame['id'], 'frames': index, 'timestamp_s': frame['timestamp_s'],
            'run_dir': run_dir, 'held_out': report['summary']['held_out'],
            'mesh_vertices': report['mesh_vertices'], 'mesh_triangles': report['mesh_triangles'],
            'newly_integrated_frame_ids': report['newly_integrated_frame_ids'],
        })
        write_json(out / 'history.json', history)
    if progress:
        progress('Exporting final 3D map and preview animations')
    # Large enclosed-room maps can exceed the standalone viewer's default cap.
    # Keep all selected surface cells and raise the limit explicitly for this workflow.
    export_viewer(out / 'reconstruction', max_voxels=200000)
    from .scene_comparison import export_camera_voxel_comparison
    if progress:
        progress('Rendering synchronized camera and cumulative voxel comparisons')
    comparison = export_camera_voxel_comparison(out)
    _export_gallery(out, cfg, history, capture, comparison)
    result = {
        'format': 'wtumi-scene-run-v1', 'scene_name': cfg['scene_name'],
        'integration_frames': count, 'held_out_frames': len(cfg['held_out']),
        'effective_voxel_size_m': report['effective_voxel_size_m'],
        'held_out': report['summary']['held_out'], 'playback': cfg['playback'],
        'paths': {'scene_config': 'scene/scene.yaml', 'model': 'scene/model.xml',
                  'capture_manifest': 'capture/manifest.json', 'capture_report': 'capture/capture.json',
                  'history': 'history.json', 'final_result': 'reconstruction',
                  'gallery': 'index.html', 'inputs_animation': 'inputs.gif',
                  'reconstruction_animation': 'reconstruction.gif',
                  **{'camera_voxel_' + key: value for key, value in comparison['paths'].items()}},
        'camera_voxel_comparison': {key: value for key, value in comparison.items() if key != 'paths'},
        'scope': 'Static environment, prescribed kinematic camera/rig path, exact simulator poses and self/payload masks. Held-out frames are evaluation-only. Prefix maps contain no future observations. No localization, locomotion, tactile fusion, or hidden-surface completion.',
        'playback_scope': 'Saved per-frame reconstruction replay; playback frame rate is not mapping throughput.',
        'total_seconds': perf_counter() - started,
    }
    write_json(out / 'scene_run.json', result)
    (out / 'README.md').write_text(
        '# Single-scene reconstruction\n\n'
        '[Preview gallery](index.html) · [Final interactive map](reconstruction/viewer.html)\n\n'
        'The camera_voxel animation pairs recorded head-camera RGB with the matching cumulative voxel map, '
        'rendered in MuJoCo from a fixed overview. Display sampling is labeled; full maps remain saved.\n\n'
        'Open this result in native MuJoCo from the project root:\n\n'
        '```bash\npython scripts/scene.py view --result PATH_TO_THIS_DIRECTORY\n```\n\n'
        'Keys: 1 original, 2 reconstructed mesh, 3 observed TSDF surface voxels, '
        '4 overlay, P play/pause, [ / ] step, C camera/free view, I RGB-D insets.\n\n'
        'Copy the whole result directory to preserve model assets and frame history. '
        'The gallery replays saved captures and prefix maps; it does not perform live fusion. '
        'The final map uses only masked camera observations, not the source mesh.\n', encoding='utf-8')
    return result
