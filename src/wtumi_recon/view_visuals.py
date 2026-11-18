"""M8 playback for fixed and independently rotating head-camera observations.

Rendering primitives are shared with M7. Each image and mesh is a measured
observation or its actual cumulative reconstruction; camera directions may
change while the shared rig trajectory and camera centers remain fixed.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .memory_visuals import (_capture_frames, _history_viewer, _map_frames,
                             _progress_plot, _run_path, _save_animation)
from .sequence import RGBDSequence


CASES = ('fixed_head', 'head_scan')
LABELS = {'fixed_head': 'Fixed head camera', 'head_scan': 'Independent head scan'}


def _source_label(case, index, count):
    return 'Box present throughout / masked fusion input'


def _validate(root, histories, sequences):
    if set(histories) != set(CASES) or set(sequences) != set(CASES):
        raise ValueError('view playback requires fixed_head and head_scan histories')
    loaded = {case: RGBDSequence(sequences[case]) for case in CASES}
    views = {case: [v for v in loaded[case].views if v['split'] == 'integration'] for case in CASES}
    held = {case: [v for v in loaded[case].views if v['split'] == 'held_out'] for case in CASES}
    baseline = CASES[0]
    n = len(views[baseline])
    if n < 1 or not held[baseline]:
        raise ValueError('view playback needs integration and evaluation frames')
    for case in CASES:
        if len(histories[case]) != n or len(views[case]) != n:
            raise ValueError('both view histories must contain every integration prefix')
        if loaded[case].camera != loaded[baseline].camera:
            raise ValueError('view playback requires matching camera calibration')
        if len(held[case]) != len(held[baseline]):
            raise ValueError('view playback requires shared held-out views')
        for actual, expected in zip(held[case], held[baseline]):
            if (actual['id'] != expected['id'] or actual['timestamp_s'] != expected['timestamp_s']
                    or not np.allclose(actual['T_wc'], expected['T_wc'], rtol=0, atol=1e-10)):
                raise ValueError('view playback requires shared held-out views')
        for index, (row, view) in enumerate(zip(histories[case], views[case])):
            other = views[baseline][index]
            if row['id'] != view['id'] or row['frames'] != index+1:
                raise ValueError('history IDs and prefix lengths must match capture order')
            if (view['id'] != other['id'] or view['timestamp_s'] != other['timestamp_s']
                    or not np.allclose(np.asarray(view['T_wc'])[:3, 3],
                                       np.asarray(other['T_wc'])[:3, 3], rtol=0, atol=1e-10)):
                raise ValueError('view playback requires shared camera centers, IDs and timestamps')
            run = _run_path(root, row)
            for path in (run/'mesh.ply', run/'occupancy.npz',
                         run/'renders'/held[baseline][0]['id']/'comparison.png',
                         loaded[case].root/'frames'/view['id']/'raw_rgb.png'):
                if not path.is_file():
                    raise FileNotFoundError(path)
    return loaded, views, held[baseline][0]['id'], n


def _gallery(root, n, playback, paths):
    def media(key, description):
        path = html.escape(paths[key], quote=True)
        if playback['mp4']:
            return f'<video controls loop playsinline preload="metadata" src="{path}" aria-label="{description}"></video>'
        return f'<img src="{path}" alt="{description}">'
    cards = ''.join(f'<article><h3>{html.escape(LABELS[case])}</h3><p>{n} masked box observations</p>'
                    f'<a class="button" href="runs/{case}/viewer.html" target="_blank" rel="noopener">Open final 3D map</a></article>' for case in CASES)
    animations = ''.join(
        f'<figure class="{wide}"><h3>{title}</h3><p>{description}</p>{media(key, description)}'
        f'<p><a href="{paths[key]}" target="_blank" rel="noopener">Open full animation</a></p></figure>'
        for key, title, description, wide in (
            ('capture_animation', 'Camera input history', 'Left: fixed head. Right: independent head scan. Raw RGB is above masked fusion RGB. The box is present in both histories throughout.', ''),
            ('occupancy_animation', 'Occupancy accumulation', 'The same center-X spatial slice at every frame. Gray: unknown. Teal: free ray evidence. Orange: observed surface hit.', ''),
            ('reconstruction_animation', 'Shared evaluation view', 'Each actual cumulative map is rendered from the same held-out camera. Magenta indicates missing predictions; evaluation images never enter fusion.', 'wide')))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>M8 · Independent head scanning</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#0c1320;color:#edf3fa;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}}main{{max-width:1200px;margin:auto;padding:36px 24px}}h1{{font-size:34px;line-height:1.2}}h2{{margin:30px 0 14px}}p{{color:#b4c4d7}}a{{color:#80e9d9}}.button{{display:inline-block;padding:10px 16px;border:1px solid #43827f;border-radius:7px;background:#1c4549;text-decoration:none}}.cards,.animations{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}article,figure{{margin:0;padding:20px;border:1px solid #2c3e58;border-radius:12px;background:#141f30}}video,img{{display:block;max-width:100%;height:auto;margin:14px auto}}.wide{{grid-column:1/-1}}nav{{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}}footer{{border-top:1px solid #2c3e58;margin-top:28px;padding-top:16px;font-size:12px}}@media(max-width:750px){{.cards,.animations{{grid-template-columns:1fr}}h1{{font-size:27px}}main{{padding:24px 16px}}}}</style></head>
<body><main><p>M8 · HEAD-CAMERA VIEWPOINTS</p><h1>What can independent head scanning reveal?</h1>
<p>Both conditions carry the same box along the same body trajectory, using {n} observations with matching camera centers, timestamps and calibration. The scanning condition changes camera orientation independently of the torso and box. Both use the same mapping algorithm and shared evaluation references.</p>
<p><a class="button" href="history_viewer.html" target="_blank" rel="noopener">Play cumulative 3D maps</a></p>
<p>White segments in the temporal viewer show each current camera optical axis. Camera position markers share the same path. Each displayed mesh is a saved cumulative reconstruction; geometry is not interpolated.</p>
<h2>Final maps</h2><div class="cards">{cards}</div>
<h2>Observations and map evidence</h2><p>Video playback: {playback['fps']:g} frames per second. This is presentation timing, not measured mapping throughput. Interactive browser playback is best effort.</p>
<div class="animations">{animations}</div><nav><a href="observation_progress.png">Coverage and unknown-region curves</a><a href="summary.csv">Numeric results</a><a href="README.md">Run notes</a><a href="animation_metadata.json">Playback metadata</a></nav>
<footer><p>Known simulator poses and masks, a static environment and prescribed head rotations. No physical pickup, walking controller, pose estimation or tactile fusion. Masked observations alone update the maps. Evaluation geometry never fills unseen regions. Occupancy evidence does not establish collision safety.</p></footer></main></body></html>'''
    (root/'index.html').write_text(page, encoding='utf-8')


def export_view_visuals(out_dir, histories, sequences, playback=None):
    """Export M8 animations and linked actual-prefix geometry without changing data."""
    if playback is not None and (not isinstance(playback, dict) or set(playback)-{'fps', 'scale', 'mp4'}):
        raise ValueError('playback supports fps, scale and mp4 only')
    playback = {'fps': 15, 'scale': 2, 'mp4': True, **(playback or {})}
    fps, scale, mp4 = playback['fps'], playback['scale'], playback['mp4']
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError('playback fps must be finite and between 0 and 60')
    if type(scale) is not int or scale not in (1, 2) or type(mp4) is not bool:
        raise ValueError('playback requires scale 1 or 2 and a boolean mp4')
    root = Path(out_dir).expanduser().resolve()
    loaded, views, evaluation_view, n = _validate(root, histories, sequences)
    if mp4:
        from .video import require_video_encoder
        require_video_encoder()
    extension = 'mp4' if mp4 else 'gif'
    paths = {'gallery': 'index.html', 'history_viewer': 'history_viewer.html',
             'history_viewer_metadata': 'history_viewer.json',
             'capture_animation': f'capture_history.{extension}',
             'reconstruction_animation': f'reconstruction_history.{extension}',
             'occupancy_animation': f'occupancy_history.{extension}',
             'animation_metadata': 'animation_metadata.json', 'observation_progress': 'observation_progress.png'}
    display = {'cases': CASES, 'labels': LABELS, 'source_label': _source_label}
    streams = {'capture': _capture_frames(loaded, views, None, scale, **display,
                         subtitle='Same body path and frame budget. Box present in both conditions throughout.'),
               'reconstruction': _map_frames(root, histories, evaluation_view, None, scale, 'reconstruction', **display),
               'occupancy': _map_frames(root, histories, evaluation_view, None, scale, 'occupancy', **display)}
    animations = {}
    for name, frames in streams.items():
        path = paths[name+'_animation']
        animations[name] = {'path': path, **_save_animation(root/path, frames, fps, mp4)}
    metadata = {'format': 'wtumi-view-animations-v1', 'case_ids': list(CASES),
                'animations': animations,
                'capture_timestamps_s': [view['timestamp_s'] for view in views[CASES[0]]],
                'semantics': 'One encoded frame per actual observation and cumulative map. Camera centers match; '
                             'head orientations may differ. Playback fps is presentation timing, not mapping throughput.'}
    (root/'animation_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    _history_viewer(root, histories, views, None, fps, **display,
                    title='Independent head scanning', show_orientation=True,
                    format_name='wtumi-view-history-viewer-v1')
    _progress_plot(root, histories, None, cases=CASES, labels=LABELS,
                   title='Independent head scanning while carrying a box',
                   footer='Same body trajectory, camera centers and frame budget. The box remains present throughout.')
    _gallery(root, n, playback, paths)
    return paths
