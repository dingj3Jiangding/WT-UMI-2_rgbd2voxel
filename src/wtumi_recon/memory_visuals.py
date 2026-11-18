"""Offline playback of measured M7 input sequences and cumulative maps.

Every temporal frame comes from a saved prefix reconstruction. These exports
do not interpolate geometry, integrate evaluation images, or simulate motion.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .occupancy import load_occupancy
from .sequence import RGBDSequence


CASES = ('always_occluded', 'observe_then_occluded')
LABELS = {'always_occluded': 'Always carrying box',
          'observe_then_occluded': 'Observe first, then carry box'}
PALETTE = np.array([[100, 116, 139], [45, 212, 191], [251, 146, 60]], dtype=np.uint8)


def _source_label(case, index, count):
    return 'No box / pre-observation' if case == 'observe_then_occluded' and index < count else 'Box present / masked input'


def _run_path(root, row):
    relative = Path(row['run_dir'])
    result = (root / relative).resolve()
    if relative.is_absolute() or not result.is_relative_to(root):
        raise ValueError('history run_dir must be relative and stay inside the output directory')
    return result


def _validate(root, histories, sequences, preobserve_frames):
    if set(histories) != set(CASES) or set(sequences) != set(CASES):
        raise ValueError('memory playback requires the two supported conditions')
    loaded = {case: RGBDSequence(sequences[case]) for case in CASES}
    views = {case: [v for v in loaded[case].views if v['split'] == 'integration'] for case in CASES}
    n = len(views[CASES[0]])
    if type(preobserve_frames) is not int or not 1 <= preobserve_frames < n:
        raise ValueError('preobserve_frames must leave both pre-observation and box phases')
    held_out = [v['id'] for v in loaded[CASES[0]].views if v['split'] == 'held_out']
    for case in CASES:
        if len(histories[case]) != n or len(views[case]) != n:
            raise ValueError('both histories must contain every integration prefix')
        if loaded[case].camera != loaded[CASES[0]].camera:
            raise ValueError('playback cameras must have matching calibration')
        if [v['id'] for v in loaded[case].views if v['split'] == 'held_out'] != held_out:
            raise ValueError('playback conditions must share evaluation view IDs')
        for index, (row, view) in enumerate(zip(histories[case], views[case])):
            other = views[CASES[0]][index]
            if row['id'] != view['id'] or row['frames'] != index + 1:
                raise ValueError('history IDs and prefix lengths must match capture order')
            if view['id'] != other['id'] or view['timestamp_s'] != other['timestamp_s'] or not np.allclose(view['T_wc'], other['T_wc'], rtol=0, atol=1e-10):
                raise ValueError('playback conditions must use the same poses and timestamps')
            run = _run_path(root, row)
            for path in (run / 'mesh.ply', run / 'occupancy.npz', run / 'renders' / held_out[0] / 'comparison.png',
                         loaded[case].root / 'frames' / view['id'] / 'raw_rgb.png'):
                if not path.is_file():
                    raise FileNotFoundError(path)
    return loaded, views, held_out[0], n


def _drawing(canvas, scale):
    draw = ImageDraw.Draw(canvas)
    if scale == 1:
        font = None
    else:
        import matplotlib
        font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), 10*scale)

    def text(position, value, fill):
        draw.text(tuple(round(v*scale) for v in position), value, fill=fill, font=font)
    return text


def _capture_frames(loaded, views, preobserve_frames, scale, *, cases=CASES,
                    labels=LABELS, source_label=_source_label, subtitle=None):
    n = len(views[cases[0]])
    for index in range(n):
        canvas = Image.new('RGB', (720*scale, 620*scale), '#101827')
        text = _drawing(canvas, scale)
        text((14, 10), f"Input frame {index+1}/{n}: {views[cases[0]][index]['id']}", fill='white')
        text((14, 29), subtitle or f'Box phase starts at frame {preobserve_frames+1} in the observe-first condition.', fill='#b9c8db')
        for column, case in enumerate(cases):
            x = 10 + column * 360
            view = views[case][index]
            text((x, 53), labels[case], fill='white')
            text((x, 71), source_label(case, index, preobserve_frames), fill='#80e9d9')
            with Image.open(loaded[case].root / 'frames' / view['id'] / 'raw_rgb.png') as image:
                raw = image.convert('RGB')
            masked, _ = loaded[case].read(view)
            for row, (label, picture) in enumerate((('Raw camera RGB', raw), ('Masked RGB used for fusion', Image.fromarray(masked)))):
                y = 95 + row * 250
                text((x, y), label, fill='#b9c8db')
                fit = min(340*scale/picture.width, 226*scale/picture.height)
                resampling = Image.Resampling.NEAREST if scale == 1 else Image.Resampling.LANCZOS
                picture = picture.resize((round(picture.width*fit), round(picture.height*fit)), resampling)
                canvas.paste(picture, (x*scale + (340*scale-picture.width)//2, (y+18)*scale))
        text((14, 596), 'Black masked pixels add no depth evidence. No physical pickup or walking is simulated.', fill='#b9c8db')
        yield canvas


def _map_frames(root, histories, evaluation_view, preobserve_frames, scale, kind, *,
                cases=CASES, labels=LABELS, source_label=_source_label):
    shared_grid = None
    n = len(histories[cases[0]])
    for index in range(n):
        if kind == 'reconstruction':
            canvas = Image.new('RGB', (1280*scale, 500*scale), '#101827')
            text = _drawing(canvas, scale)
            text((14, 10), f'Cumulative maps after {index+1}/{n} integration frames | Evaluation view: {evaluation_view}', fill='white')
        else:
            canvas = Image.new('RGB', (800*scale, 478*scale), '#101827')
            text = _drawing(canvas, scale)
            text((12, 10), f'After {index+1}/{n} frames | Center-X slice: Y right, Z up', fill='white')
            text((12, 28), 'Gray: unknown | Teal: free ray evidence | Orange: observed surface hit', fill='#b9c8db')
        for column, case in enumerate(cases):
            row = histories[case][index]
            run = _run_path(root, row)
            if kind == 'reconstruction':
                text((10+640*column, 35), labels[case], fill='white')
                text((10+640*column, 53), source_label(case, index, preobserve_frames), fill='#80e9d9')
                with Image.open(run / 'renders' / evaluation_view / 'comparison.png') as image:
                    image = image.convert('RGB')
                    image.thumbnail((630*scale, 405*scale), Image.Resampling.LANCZOS)
                    canvas.paste(image, ((5+640*column)*scale+(630*scale-image.width)//2, 76*scale))
            else:
                grid = load_occupancy(run / 'occupancy.npz')
                parameters = (grid['resolution'], grid['voxel_size_m'], grid['origin_m'].tolist())
                if shared_grid is None:
                    shared_grid = parameters
                elif parameters != shared_grid:
                    raise ValueError('all history snapshots must share occupancy grid bounds and resolution')
                mid = grid['resolution']//2
                pixels = PALETTE[grid['state'][mid, :, :].T[::-1]]
                picture = Image.fromarray(pixels).resize((380*scale, 380*scale), Image.Resampling.NEAREST)
                canvas.paste(picture, ((10+400*column)*scale, 70*scale))
                text((10+400*column, 51), labels[case], fill='white')
                text((10+400*column, 457), f"Region unknown: {100*row['roi_unknown_fraction']:.2f}%", fill='white')
        if kind == 'reconstruction':
            text((14, 483), 'Shared evaluation reference only; magenta marks missing predictions. These images never enter fusion.', fill='#b9c8db')
        yield canvas


def _save_animation(path, frames, fps, mp4):
    if mp4:
        from .video import save_mp4
        return save_mp4(path, frames, fps)
    # GIF is the small default preview; the high-resolution preset streams MP4.
    frames = list(frames)
    duration_ms = max(10, round(1000/fps/10)*10)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return {'codec': 'GIF', 'width': frames[0].width, 'height': frames[0].height,
            'fps': 1000/duration_ms, 'requested_fps': fps, 'frame_count': len(frames),
            'duration_s': len(frames)*duration_ms/1000}


def _mesh_trace(path, scene):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f'history mesh is empty or invalid: {path}')
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full(vertices.shape, .65)
    rgb = np.rint(np.clip(colors, 0, 1)*255).astype(np.uint8)
    trace = {'type': 'mesh3d', 'scene': scene,
             **{key: vertices[:, i].tolist() for i, key in enumerate(('x', 'y', 'z'))},
             **{key: triangles[:, i].tolist() for i, key in enumerate(('i', 'j', 'k'))},
             'vertexcolor': [f'rgb({r},{g},{b})' for r, g, b in rgb],
             'lighting': {'ambient': .85, 'diffuse': .55, 'specular': .03},
             'flatshading': False, 'name': 'Reconstructed surface', 'hoverinfo': 'skip', 'showlegend': False}
    return trace, vertices


def _history_viewer(root, histories, views, preobserve_frames, fps, *, cases=CASES,
                    labels=LABELS, source_label=_source_label,
                    title='Visual memory', show_orientation=False,
                    format_name='wtumi-memory-history-viewer-v1'):
    from plotly.offline import get_plotlyjs

    frames, bounds = [], []
    n = len(histories[cases[0]])
    for index in range(n):
        traces, orientation_traces = [], []
        for column, case in enumerate(cases):
            scene = 'scene' if column == 0 else 'scene2'
            trace, vertices = _mesh_trace(_run_path(root, histories[case][index]) / 'mesh.ply', scene)
            traces.append(trace)
            bounds.extend((vertices.min(axis=0), vertices.max(axis=0)))
            positions = np.array([np.asarray(view['T_wc'])[:3, 3] for view in views[case][:index+1]])
            bounds.extend((positions.min(axis=0), positions.max(axis=0)))
            traces.append({'type': 'scatter3d', 'scene': scene,
                           **{key: positions[:, i].tolist() for i, key in enumerate(('x', 'y', 'z'))},
                           'mode': 'lines+markers', 'line': {'color': '#2dd4bf', 'width': 5},
                           'marker': {'color': '#fbbf24', 'size': 4},
                           'name': 'Observed camera positions', 'showlegend': False,
                           'text': [view['id'] for view in views[case][:index+1]],
                           'hovertemplate': '%{text}<extra></extra>'})
            if show_orientation:
                pose = np.asarray(views[case][index]['T_wc'])
                segment = np.stack((pose[:3, 3], pose[:3, 3] + .35*pose[:3, 2]))
                bounds.extend(segment)
                orientation_traces.append({'type': 'scatter3d', 'scene': scene,
                    **{key: segment[:, i].tolist() for i, key in enumerate(('x', 'y', 'z'))},
                    'mode': 'lines+markers', 'line': {'color': '#f8fafc', 'width': 7},
                    'marker': {'color': '#f8fafc', 'size': 3}, 'name': 'Current optical axis',
                    'showlegend': False, 'hoverinfo': 'name'})
        traces.extend(orientation_traces)
        frames.append({'id': histories[cases[0]][index]['id'], 'integrated_frames': index+1,
                       'source_labels': [source_label(case, index, preobserve_frames) for case in cases],
                       'traces': traces})
    points = np.asarray(bounds)
    low, high = points.min(axis=0), points.max(axis=0)
    pad = np.maximum((high-low)*.07, .05)
    ranges = np.stack((low-pad, high+pad), axis=1).tolist()
    payload = {'format': format_name, 'case_ids': list(cases),
               'labels': [labels[case] for case in cases], 'preobserve_frames': preobserve_frames,
               'ranges': ranges, 'frames': frames, 'target_fps': fps,
               'semantics': 'Saved cumulative prefix meshes; no interpolation, hidden geometry or physics playback.'}
    if preobserve_frames is None:
        payload.pop('preobserve_frames')
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    page = _HISTORY_HTML.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', encoded)
    page = page.replace('Visual memory', html.escape(title))
    if show_orientation:
        page = page.replace('Yellow markers: camera positions already observed.',
                            'Yellow markers: camera positions already observed. White segments: current camera optical axes.')
    (root / 'history_viewer.html').write_text(page, encoding='utf-8')
    metadata = {key: value for key, value in payload.items() if key != 'frames'}
    metadata['frames'] = [{'id': f['id'], 'integrated_frames': f['integrated_frames'],
                           'mesh_vertices': [len(f['traces'][i]['x']) for i in (0, 2)]} for f in frames]
    (root / 'history_viewer.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')


def _progress_plot(root, histories, preobserve_frames, *, cases=CASES, labels=LABELS,
                   title='Visual memory under later box occlusion', footer=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    columns = [('stair_depth_coverage', 'Stair depth coverage (%)'),
               ('surface_coverage_within_tolerance_fraction', 'Surface samples within tolerance (%)'),
               ('roi_unknown_fraction', 'Unknown cells in fixed region (%)')]
    for case, color in zip(cases, ('#d76f36', '#2786b5')):
        rows = histories[case]
        for ax, (key, label) in zip(axes, columns):
            ax.plot([r['frames'] for r in rows], [100*r[key] for r in rows], marker='o', label=labels[case], color=color)
            ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlabel('Integrated frames')
        ax.set_ylim(0, 101)
        if len(histories[cases[0]]) <= 10:
            ax.set_xticks(range(1, len(histories[cases[0]])+1))
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if preobserve_frames is not None:
            ax.axvline(preobserve_frames+.5, color='#64748b', linestyle='--', linewidth=1)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.text(.5, .015, footer or f'Dashed line: box introduced after frame {preobserve_frames} in the observe-first condition. Same frame count and camera poses.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(root / 'observation_progress.png', dpi=160)
    plt.close(fig)


def _gallery(root, n, preobserve_frames, playback):
    cards = ''.join(f'<article><h3>{html.escape(LABELS[case])}</h3><p>{n} integrated frames</p><a class="button" href="runs/{case}/viewer.html" target="_blank" rel="noopener">Open final 3D map ↗</a></article>' for case in CASES)
    page = _GALLERY_HTML.replace('__CARDS__', cards).replace('__N__', str(n)).replace('__K__', str(preobserve_frames)).replace('__AFTER__', str(n-preobserve_frames))
    if playback['mp4']:
        import re
        page = re.sub(r'<img src="([^"]+)_history.gif" alt="([^"]+)">',
                      r'<video controls loop playsinline preload="metadata" src="\1_history.mp4" aria-label="\2"></video>', page)
        page = page.replace('_history.gif', '_history.mp4')
    page = page.replace('__FPS__', f"{playback['fps']:g}")
    (root / 'index.html').write_text(page, encoding='utf-8')


def export_memory_visuals(out_dir, histories, sequences, preobserve_frames=2, playback=None):
    """Export actual cumulative maps and observations at a declared playback rate.

    MP4 encodes frames as a stream. Playback speed does not measure capture or
    fusion performance. The interactive mesh viewer's actual rate is limited
    by browser rendering, independently of the video frame rate.
    """
    playback = {'fps': 1000/900, 'scale': 1, 'mp4': False, **(playback or {})}
    fps, scale, mp4 = playback['fps'], playback['scale'], playback['mp4']
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError('playback fps must be a finite number greater than 0 and at most 60')
    if type(scale) is not int or scale not in (1, 2) or type(mp4) is not bool:
        raise ValueError('playback scale must be 1 or 2 and mp4 must be boolean')
    if mp4:
        from .video import require_video_encoder
        require_video_encoder()
    root = Path(out_dir).expanduser().resolve()
    loaded, views, evaluation_view, n = _validate(root, histories, sequences, preobserve_frames)
    extension = 'mp4' if mp4 else 'gif'
    paths = {'gallery': 'index.html', 'history_viewer': 'history_viewer.html',
             'history_viewer_metadata': 'history_viewer.json',
             'capture_animation': f'capture_history.{extension}',
             'reconstruction_animation': f'reconstruction_history.{extension}',
             'occupancy_animation': f'occupancy_history.{extension}',
             'animation_metadata': 'animation_metadata.json', 'observation_progress': 'observation_progress.png'}
    streams = {'capture': _capture_frames(loaded, views, preobserve_frames, scale),
               'reconstruction': _map_frames(root, histories, evaluation_view, preobserve_frames, scale, 'reconstruction'),
               'occupancy': _map_frames(root, histories, evaluation_view, preobserve_frames, scale, 'occupancy')}
    animations = {}
    for name, frames in streams.items():
        path = paths[name+'_animation']
        animations[name] = {'path': path, **_save_animation(root/path, frames, fps, mp4)}
    metadata = {'format': 'wtumi-memory-animations-v1', 'animations': animations,
                'capture_timestamps_s': [view['timestamp_s'] for view in views[CASES[0]]],
                'semantics': 'One encoded frame per actual observation and cumulative map prefix. '
                             'Uniform playback fps is presentation timing, not measured reconstruction throughput. '
                             'Interactive browser playback is best effort.'}
    (root/'animation_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    _history_viewer(root, histories, views, preobserve_frames, fps)
    _progress_plot(root, histories, preobserve_frames)
    _gallery(root, n, preobserve_frames, playback)
    return paths


_HISTORY_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual memory · cumulative 3D playback</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101827;color:#e5edf8;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:20px 24px 12px;border-bottom:1px solid #2b394f}h1{font-size:24px;margin:0 0 8px}p{color:#b4c3d6;margin:7px 0;line-height:1.5}a{color:#7de7d8}.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0 8px}button{background:#1b3c43;border:1px solid #3d7c7c;color:white;border-radius:6px;padding:9px 16px;cursor:pointer}button:disabled{opacity:.5}input{width:min(420px,50vw);accent-color:#2dd4bf}#status{font-size:13px;color:#b4c3d6}#plot{width:100%;height:calc(100vh - 215px);min-height:440px}.labels{display:grid;grid-template-columns:1fr 1fr;text-align:center;gap:20px;padding:12px 16px 0}.labels strong{display:block;font-size:16px}.labels span{display:block;color:#80e9d9;font-size:13px;margin-top:5px}.help{font-size:12px}@media(max-width:700px){header{padding:14px}h1{font-size:20px}#plot{height:65vh}.labels strong{font-size:13px}}</style>
<script>__PLOTLY__</script></head><body>
<header><h1>Visual memory: cumulative 3D maps</h1><p>Each frame is the map reconstructed from observations 1 through that frame. Rotate either map to align both views.</p>
<div class="toolbar"><button id="play">Play</button><button id="pause" disabled>Pause</button><label for="frame">Integrated frames</label><input id="frame" type="range" min="1" step="1" value="1"><strong id="counter"></strong><a href="index.html">All results</a></div>
<p id="status" role="status">Loading saved meshes...</p><p class="help">Yellow markers: camera positions already observed. Shared axes and scale. This slider changes time; final-map occupancy sliders select spatial slices. Interactive playback is best effort and may run slower than the encoded video. Prescribed capture only, with no pickup or walking animation.</p></header>
<div class="labels"><div><strong id="label-a"></strong><span id="source-a"></span></div><div><strong id="label-b"></strong><span id="source-b"></span></div></div><div id="plot"></div>
<script id="memory-history-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('memory-history-data').textContent);
const plot=document.getElementById('plot'), slider=document.getElementById('frame');
const play=document.getElementById('play'), pauseButton=document.getElementById('pause');
slider.max=String(data.frames.length);
document.getElementById('label-a').textContent=data.labels[0];
document.getElementById('label-b').textContent=data.labels[1];
let current=0, playing=false, timer=null, queue=Promise.resolve(), generation=0, syncing=false;
let camera={up:{x:0,y:0,z:1},center:{x:0,y:0,z:0},eye:{x:1.4,y:-1.8,z:1.25}};
function scene(domain){const result={domain:{x:domain,y:[0,1]},aspectmode:'data',camera:camera,uirevision:'memory-camera'};['x','y','z'].forEach((axis,i)=>{result[axis+'axis']={title:axis.toUpperCase()+' (m)',range:data.ranges[i],autorange:false,backgroundcolor:'#111b2b',gridcolor:'#30415c',showbackground:true};});return result;}
function layout(){return {paper_bgcolor:'#101827',font:{color:'#e5edf8'},margin:{l:0,r:0,t:8,b:0},showlegend:false,uirevision:'memory-camera',scene:scene([0,.485]),scene2:scene([.515,1])};}
function show(index){queue=queue.then(async()=>{const frame=data.frames[index];await Plotly.react(plot,frame.traces,layout(),{displaylogo:false,responsive:true,scrollZoom:true});current=index;slider.value=String(index+1);document.getElementById('counter').textContent=(index+1)+' / '+data.frames.length;document.getElementById('status').textContent='Frame '+(index+1)+': '+frame.id+' · '+frame.integrated_frames+' observations fused';document.getElementById('source-a').textContent=frame.source_labels[0];document.getElementById('source-b').textContent=frame.source_labels[1];});return queue;}
function pause(){playing=false;generation++;clearTimeout(timer);play.disabled=false;pauseButton.disabled=true;}
async function tick(token){if(!playing||token!==generation)return;const started=performance.now();await show((current+1)%data.frames.length);if(playing&&token===generation)timer=setTimeout(()=>tick(token),Math.max(0,1000/data.target_fps-(performance.now()-started)));}
play.addEventListener('click',()=>{if(playing)return;playing=true;play.disabled=true;pauseButton.disabled=false;const token=++generation;timer=setTimeout(()=>tick(token),1000/data.target_fps);});
pauseButton.addEventListener('click',pause);
slider.addEventListener('change',()=>{pause();show(Number(slider.value)-1);});
show(0).then(()=>{plot.on('plotly_relayout',event=>{if(syncing)return;const first=event['scene.camera'],second=event['scene2.camera'];if(!first&&!second)return;camera=first||second;syncing=true;Promise.resolve(Plotly.relayout(plot,{[(first?'scene2':'scene')+'.camera']:camera})).finally(()=>{syncing=false;});});window.memoryPlaybackReady=true;});
window.addEventListener('resize',()=>Plotly.Plots.resize(plot));
</script></body></html>'''


_GALLERY_HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M7 · Visual memory under occlusion</title><style>*{box-sizing:border-box}body{margin:0;background:#0c1320;color:#edf3fa;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}main{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2;margin:10px 0}h2{margin:30px 0 14px}h3{margin:0 0 8px}p{color:#b4c4d7}a{color:#80e9d9}.eyebrow{color:#80e9d9;letter-spacing:.1em;font-size:12px}.button{display:inline-block;padding:10px 16px;border:1px solid #43827f;border-radius:7px;background:#1c4549;text-decoration:none}.cards,.animations{display:grid;grid-template-columns:1fr 1fr;gap:18px}article,figure{margin:0;padding:20px;border:1px solid #2c3e58;border-radius:12px;background:#141f30}figure img,figure video{display:block;max-width:100%;height:auto;margin:14px auto 0}.wide{grid-column:1/-1}.links{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}footer{font-size:12px;border-top:1px solid #2c3e58;margin-top:28px;padding-top:16px}@media(max-width:750px){.cards,.animations{grid-template-columns:1fr}h1{font-size:27px}main{padding:24px 16px}}</style></head><body><main>
<p class="eyebrow">M7 · VISION-TO-VOXEL MEMORY</p><h1>What remains when the box blocks the view?</h1><p>Same __N__ camera poses and frame budget. One sequence has a box throughout; the other observes without the box for __K__ frames, then uses __AFTER__ masked box frames. The histories contain different visual information, so this is a memory experiment, not an algorithm comparison.</p>
<p><a class="button" href="history_viewer.html" target="_blank" rel="noopener">Play the cumulative 3D maps ↗</a></p><p>The 3D playback has synchronized frame selection, Play/Pause and linked viewing angles. It displays actual saved prefix meshes with fixed axes; no geometry is interpolated between observations.</p>
<h2>Final maps</h2><div class="cards">__CARDS__</div><p>Final viewers provide mesh, surface voxels, point cloud and occupancy slices. Their slice sliders select space, not time.</p>
<h2>Animated observations and map evidence</h2><p>Playback target: __FPS__ frames per second. Each frame uses a real captured observation and its cumulative map; this is not a real-time processing benchmark.</p><div class="animations"><figure><h3>Camera input history</h3><p>Left: box throughout. Right: observe first. Raw RGB is above masked fusion RGB. Black masked regions add no depth evidence.</p><img src="capture_history.gif" alt="Actual raw and masked camera images across the two observation schedules"><p><a href="capture_history.gif" target="_blank" rel="noopener">Open full animation ↗</a></p></figure><figure><h3>Occupancy accumulation</h3><p>Same center-X slice at each prefix. Gray: unknown. Teal: free ray evidence. Orange: surface-hit evidence.</p><img src="occupancy_history.gif" alt="Accumulated occupancy evidence in a common Y-Z slice"><p><a href="occupancy_history.gif" target="_blank" rel="noopener">Open full animation ↗</a></p></figure><figure class="wide"><h3>Held-out view after each observation</h3><p>Both maps are rendered from the same evaluation camera. Each comparison shows the shared reference, current reconstruction and error. Magenta denotes missing predictions.</p><img src="reconstruction_history.gif" alt="Reconstructed RGB and depth accumulating across frames, compared with evaluation-only references"><p><a href="reconstruction_history.gif" target="_blank" rel="noopener">Open full animation ↗</a></p></figure></div>
<nav class="links"><a href="observation_progress.png" target="_blank" rel="noopener">Coverage and unknown-region curves ↗</a><a href="summary.csv">Numeric results</a><a href="README.md">Run notes</a></nav>
<footer><p>Known simulator poses and masks, static environment and scheduled paired observations. No physical pickup, walking, pose estimation or tactile fusion. Held-out images and analytic stair samples are evaluation-only and never enter fusion. Occupancy evidence is not a collision-safety guarantee.</p></footer></main></body></html>'''
