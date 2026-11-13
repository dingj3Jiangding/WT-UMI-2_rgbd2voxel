"""Offline interactive inspection of reconstructed geometry and TSDF samples."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import html
import json
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class SurfaceVoxels:
    indices: np.ndarray
    colors: np.ndarray
    origin: np.ndarray
    voxel_size: float
    resolution: int
    band_voxels: float
    min_weight: float
    observed_count: int
    total_count: int

    @property
    def centers(self):
        return self.origin + (self.indices + 0.5) * self.voxel_size


def load_surface_voxels(path, band_voxels=1.0, min_weight=1.0, max_voxels=100000):
    """Select an observed TSDF band, not occupied solids or all known free space.

    Open3D Uniform flat index is (x*N+y)*N+z (C order). This was verified
    against injected asymmetric voxel coordinates and per-axis RGB gradients.
    """
    if not np.isfinite(band_voxels) or band_voxels <= 0:
        raise ValueError("band_voxels must be finite and positive")
    if not np.isfinite(min_weight) or min_weight <= 0 or max_voxels < 1:
        raise ValueError("min_weight and max_voxels must be positive")
    with np.load(path, allow_pickle=False) as grid:
        n = int(grid["resolution"].item())
        size, trunc = float(grid["voxel_size_m"]), float(grid["sdf_trunc_m"])
        origin = np.asarray(grid["origin_m"], dtype=float)
        if n < 2 or not np.isfinite([size, trunc]).all() or size <= 0 or trunc <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("invalid TSDF grid metadata")
        for key, expected in (("axis_order", "xyz"), ("flat_order", "C")):
            if key in grid and str(grid[key].item()) != expected:
                raise ValueError(f"unsupported volume {key}")
        if "tsdf_normalized" in grid and not bool(grid["tsdf_normalized"].item()):
            raise ValueError("expected normalized Open3D TSDF")
        sdf, weight, color = grid["tsdf"], grid["weight"], grid["color"]
        if sdf.shape != (n**3,) or weight.shape != sdf.shape or color.shape != (n**3, 3):
            raise ValueError("TSDF/color array dimensions do not match resolution")
        observed = np.isfinite(weight) & (weight > 0)
        selected = observed & (weight >= min_weight) & np.isfinite(sdf) & (np.abs(sdf)*trunc <= band_voxels*size)
        flat = np.flatnonzero(selected)
        if len(flat) > max_voxels:
            raise ValueError(f"{len(flat)} selected voxels exceeds max_voxels={max_voxels}; reduce the band or explicitly raise the limit")
        if not len(flat):
            raise ValueError("no observed voxels satisfy the surface-band filters")
        indices = np.column_stack(np.unravel_index(flat, (n, n, n), order="C"))
        scale = float(grid["color_scale"]) if "color_scale" in grid else 255.0
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(color[flat]).all():
            raise ValueError("invalid voxel colors or scale")
        colors = np.clip(color[flat] / scale, 0, 1)
    return SurfaceVoxels(indices, colors, origin, size, n, band_voxels, min_weight,
                         int(observed.sum()), n**3)


def voxel_shell_mesh(voxels):
    """Build actual cubical faces; remove shared faces without moving samples."""
    occupied = {tuple(i) for i in voxels.indices}
    faces = [
        ((-1, 0, 0), [(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),
        ((1, 0, 0), [(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
        ((0,-1,0), [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
        ((0,1,0), [(0,1,0),(0,1,1),(1,1,1),(1,1,0)]),
        ((0,0,-1), [(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),
        ((0,0,1), [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),
    ]
    vertices, triangles, colors, vertex_ids = [], [], [], {}
    for idx, color in zip(voxels.indices, voxels.colors):
        for offset, corners in faces:
            if tuple(idx + offset) in occupied:
                continue
            ids = []
            for corner in corners:
                key = tuple(idx + corner)
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertices)
                    vertices.append(key)
                ids.append(vertex_ids[key])
            triangles.extend([[ids[0],ids[1],ids[2]], [ids[0],ids[2],ids[3]]])
            colors.extend([color, color])
    world_vertices = voxels.origin + np.asarray(vertices, dtype=float) * voxels.voxel_size
    return world_vertices, np.asarray(triangles, dtype=np.int32), np.asarray(colors)


def camera_frustum_segments(k, t_wc, width, height, depth=0.25):
    """World-space line segments, using integer pixel centers and T_wc."""
    k, pose = np.asarray(k, dtype=float), np.asarray(t_wc, dtype=float)
    if k.shape != (3,3) or pose.shape != (4,4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("invalid camera metadata")
    pixels = np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]], dtype=float)
    corners = (pixels @ np.linalg.inv(k).T * depth) @ pose[:3,:3].T + pose[:3,3]
    center = pose[:3,3]
    segments = [(center, p) for p in corners] + [(corners[i], corners[(i+1)%4]) for i in range(4)]
    return np.asarray(segments)


def _rgb_strings(colors):
    return [f"rgb({r},{g},{b})" for r,g,b in np.rint(np.clip(colors,0,1)*255).astype(np.uint8)]


def export_viewer(result_dir, output=None, *, band_voxels=1.0, min_weight=1.0, max_voxels=100000):
    """Export one self-contained HTML; no server, uploads or CDN required."""
    import plotly.graph_objects as go
    import plotly.io as pio

    root = Path(result_dir).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else root / "viewer.html"
    if destination.suffix.lower() != ".html":
        raise ValueError("viewer output must end in .html")
    metadata_path = destination.with_suffix(".json")
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text())
        except (ValueError, OSError):
            previous = None
        if not isinstance(previous, dict) or previous.get("format") != "wtumi-recon-viewer-v1":
            raise FileExistsError(f"refusing to overwrite non-viewer metadata: {metadata_path}")
    mesh_path = root / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise ValueError("reconstruction mesh is empty")
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not np.isfinite(vertices).all():
        raise ValueError("mesh contains non-finite vertices")
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full(vertices.shape,0.7)
    fig = go.Figure()
    lighting = dict(ambient=0.8, diffuse=0.65, specular=0.05, roughness=1)
    fig.add_trace(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], vertexcolor=_rgb_strings(colors),
        name="Reconstructed mesh", lighting=lighting, flatshading=False, hoverinfo="skip"))

    voxel_path = root / "tsdf_volume.npz"
    voxels = load_surface_voxels(voxel_path,band_voxels,min_weight,max_voxels) if voxel_path.is_file() else None
    if voxels:
        vv, tt, cc = voxel_shell_mesh(voxels)
        fig.add_trace(go.Mesh3d(x=vv[:,0],y=vv[:,1],z=vv[:,2],i=tt[:,0],j=tt[:,1],k=tt[:,2],
            facecolor=_rgb_strings(cc), name="Observed TSDF surface-band voxels",
            visible=False,flatshading=True,lighting=lighting,hoverinfo="skip"))
    else:
        fig.add_trace(go.Mesh3d(x=[],y=[],z=[],name="No saved uniform volume",visible=False))

    cloud_path = root / "pointcloud.ply"
    if cloud_path.is_file():
        pc = o3d.io.read_point_cloud(str(cloud_path))
        points = np.asarray(pc.points)
        pc_colors = np.asarray(pc.colors) if pc.has_colors() else np.full(points.shape, 0.7)
    else:
        points, pc_colors = vertices, colors
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("point cloud is empty or contains non-finite points")
    step = max(1, int(np.ceil(len(points)/100000)))
    fig.add_trace(go.Scatter3d(x=points[::step,0],y=points[::step,1],z=points[::step,2],mode="markers",
        marker=dict(size=2,color=_rgb_strings(pc_colors[::step])),name="Reconstructed point cloud",visible=False,hoverinfo="skip"))

    mode_trace_indices = {"0": [0], "1": [1], "2": [2]}
    occupancy_payload = None
    occupancy_metadata = None
    occupancy_path = root / "occupancy.npz"
    if occupancy_path.is_file():
        from .occupancy import load_occupancy
        occupancy = load_occupancy(occupancy_path)
        n = int(occupancy["resolution"])
        if n * n > max_voxels:
            raise ValueError(f"occupancy slice has {n*n} cells, exceeding max_voxels={max_voxels}")
        occupancy_metadata = {
            "resolution": n, "origin_m": np.asarray(occupancy["origin_m"]).tolist(),
            "voxel_size_m": float(occupancy["voxel_size_m"]),
            "counts": occupancy["summary"], "slice_cells": n * n,
            "initial_slice_axis": "y", "initial_slice_index": n // 2,
            "semantics": "Ray evidence: 0 unknown, 1 free, 2 occupied. Occupied evidence dominates conflicting free evidence; interiors are not inferred. A traversed cell is not a whole-cell collision guarantee.",
        }
        occupancy_payload = {
            **occupancy_metadata,
            "state_base64": base64.b64encode(np.asarray(occupancy["state"], dtype=np.uint8).tobytes(order="C")).decode("ascii"),
        }
        mode_trace_indices["3"] = [len(fig.data)]
        fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
            name="Occupancy evidence slice", visible=False, flatshading=True,
            lighting=dict(ambient=1, diffuse=0.2, specular=0, roughness=1), hoverinfo="skip"))

    manifest = json.loads((root/"manifest.json").read_text()) if (root/"manifest.json").exists() else None
    camera_rows, camera_indices = [], []
    if manifest:
        for frame in manifest["frames"]:
            pose = np.asarray(frame["T_wc"],dtype=float)
            segments = camera_frustum_segments(manifest["K"],pose,manifest["width"],manifest["height"])
            x,y,z = [],[],[]
            for segment in segments:
                x.extend([segment[0,0],segment[1,0],None]); y.extend([segment[0,1],segment[1,1],None]); z.extend([segment[0,2],segment[1,2],None])
            color = "#2dd4bf" if frame["split"] == "integration" else "#fbbf24"
            camera_indices.append(len(fig.data))
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="lines",line=dict(color=color,width=4),
                name=f"{frame['id']} ({frame['split']})",visible=False,hoverinfo="name"))
            camera_rows.append({"id":frame["id"],"split":frame["split"],"position":pose[:3,3].tolist()})

    world_up = np.asarray(manifest.get("world_up", [0, -1, 0]) if manifest else [0, -1, 0], dtype=float)
    if world_up.shape != (3,) or not np.isfinite(world_up).all() or np.linalg.norm(world_up) < 1e-8:
        raise ValueError("invalid manifest world_up")
    world_up = world_up / np.linalg.norm(world_up)
    eye = dict(x=1.5,y=-1.7,z=1.2) if np.allclose(world_up, [0,0,1]) else dict(x=1.25,y=-0.65,z=-1.8)
    default_camera = dict(up=dict(zip(("x","y","z"), world_up.tolist())),center=dict(x=0,y=0,z=0),eye=eye)
    axis = dict(backgroundcolor="#111b2b",gridcolor="#26354b",zerolinecolor="#415570",showbackground=True)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#101827",plot_bgcolor="#101827",
        margin=dict(l=0,r=0,t=0,b=0),showlegend=False,uirevision="keep-view",
        scene=dict(aspectmode="data",camera=default_camera,
                   xaxis=dict(title="X (m)",**axis),yaxis=dict(title="Y (m)",**axis),zaxis=dict(title="Z (m)",**axis)))
    figure = pio.to_html(fig,full_html=False,include_plotlyjs=True,div_id="scene-view",
                         config=dict(displaylogo=False,scrollZoom=True,responsive=True,locale="en"),default_height="100%",
                         post_script="window.wtumiPlotReady = true; window.dispatchEvent(new Event('wtumi-plot-ready'));")
    metadata = {
        "format": "wtumi-recon-viewer-v1",
        "world_up": world_up.tolist(),
        "source":str(root),"mesh_vertices":len(vertices),"mesh_triangles":len(triangles),
        "point_count":len(points),"displayed_point_count":len(points[::step]),
        "surface_voxels":len(voxels.indices) if voxels else None,
        "voxel_size_m":voxels.voxel_size if voxels else None,"band_voxels":band_voxels,
        "min_weight":min_weight,"observed_cells":voxels.observed_count if voxels else None,
        "total_cells":voxels.total_count if voxels else None,"cameras":camera_rows,
        "occupancy": occupancy_metadata,
        "semantics":"Observed near-zero TSDF cells only. Unobserved cells are hidden, not free. This is not a complete occupancy map.",
    }
    summary = json.loads((root/"metrics.json").read_text()) if (root/"metrics.json").exists() else {}
    name = html.escape(str(summary.get("scene",root.name)))
    payload = json.dumps({"meta":metadata,"camera_indices":camera_indices,"mode_trace_indices":mode_trace_indices,"occupancy":occupancy_payload,"default_camera":default_camera},ensure_ascii=False,allow_nan=False).replace("</","<\\/")
    disabled = "" if voxels else " disabled"
    count = f"{len(voxels.indices):,}" if voxels else "Unavailable"
    voxel_size = f"{voxels.voxel_size*1000:.3f} mm" if voxels else "No TSDF file"
    occupancy_button = '<button data-mode="3">Occupancy · State slices</button>' if occupancy_payload else ""
    occupancy_controls = ""
    if occupancy_payload:
        counts = occupancy_metadata["counts"]
        occupancy_controls = f'''<section id="occupancy-controls" hidden><div class="label">Occupancy slices</div>
<label>Slice axis <select id="slice-axis"><option value="0">X</option><option value="1" selected>Y</option><option value="2">Z</option></select></label>
<label>Slice index <input id="slice-index" type="range" min="0" max="{n-1}" value="{n//2}" step="1"></label>
<div id="slice-label" class="small"></div><div class="legend small"><span style="color:#8190a5">Gray: unknown, no observation evidence</span><span style="color:#2dd4bf">Teal: free, traversed by an observed ray</span><span style="color:#fb923c">Orange: occupied, observed ray endpoint</span></div>
<div class="stats small">Occupancy voxel size: {occupancy_metadata["voxel_size_m"]*1000:.3f} mm<br>Unknown: {counts["unknown_cells"]:,}<br>Free: {counts["free_cells"]:,}<br>Occupied: {counts["occupied_cells"]:,}<br>Cells with conflicting evidence: {counts["conflict_cells"]:,}</div>
<div class="note">One layer of cells is shown. Change the axis or drag the index slider to inspect another slice.<br>Occupied evidence takes precedence over conflicting free evidence; conflict counts are retained separately. Surface hits do not fill object interiors. Free records ray traversal and does not guarantee clearance throughout a cell.</div></section>'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} · Reconstruction viewer</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#101827;color:#e5edf8;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{padding:20px 26px;border-bottom:1px solid #2b394f}}h1{{font-size:23px;margin:0 0 8px}}header p{{color:#a8b8ce;margin:0}}main{{display:grid;grid-template-columns:285px 1fr;height:calc(100vh - 95px);min-height:570px}}aside{{padding:22px;border-right:1px solid #2b394f;overflow:auto}}.label{{font-size:12px;color:#9aacbf;letter-spacing:.06em;margin:20px 0 10px}}.modes{{display:grid;gap:8px}}button{{cursor:pointer;border:1px solid #3b4c65;background:#1b2a40;color:#e5edf8;padding:11px;border-radius:7px;font:inherit}}button.active{{background:#145e68;border-color:#2dd4bf}}button:disabled{{opacity:.4;cursor:default}}[hidden]{{display:none!important}}select{{background:#1b2a40;color:#e5edf8;padding:5px;border:1px solid #3b4c65}}input[type=range]{{width:100%}}.small{{font-size:12px;line-height:1.7;color:#a8b8ce}}.stats{{line-height:1.9}}.note{{padding:12px;background:#19283d;border-radius:8px;margin-top:18px;line-height:1.7;font-size:13px}}label{{display:block;margin:15px 0}}input{{accent-color:#2dd4bf}}#canvas{{min-width:0;height:100%;position:relative}}#canvas > div{{height:100%}}#scene-view{{height:100%;width:100%}}.legend span{{display:block;margin:5px 0}}.teal{{color:#2dd4bf}}.amber{{color:#fbbf24}}#camera-list{{padding-left:18px}}@media(max-width:760px){{main{{display:block;height:auto}}aside{{border-right:0}}#canvas{{height:65vh;min-height:450px}}}}</style></head>
<body><header><h1>3D reconstruction viewer · {name}</h1><p>Drag to rotate · Scroll to zoom · Use the toolbar to pan | Data from saved reconstruction results</p></header>
<main><aside><div class="label">Display mode</div><div class="modes"><button class="active" data-mode="0">Mesh · Reconstructed surface</button><button data-mode="1"{disabled}>Voxel · Surface cells</button><button data-mode="2">Point cloud · Surface points</button>{occupancy_button}</div>{occupancy_controls}
<label><input id="show-cameras" type="checkbox"> Show camera poses</label><div class="legend small"><span class="teal">Teal: integration cameras</span><span class="amber">Yellow: held-out evaluation cameras</span></div><ul id="camera-list" class="small"></ul>
<button id="reset-view">Reset view</button><div class="label">Result size</div><div class="stats small">Mesh vertices: {len(vertices):,}<br>Mesh triangles: {len(triangles):,}<br>Selected surface voxels: {count}<br>Voxel edge length: {voxel_size}</div>
<div class="note">Voxel mode shows observed cells near the TSDF zero crossing.<br>Filter: |TSDF| × truncation distance ≤ {band_voxels:g} × voxel edge length, weight ≥ {min_weight:g}.<br><br>Cells hidden in TSDF mode cannot be assumed free. Switch to Occupancy if a separate ray-evidence map is available.</div>
<div id="mode-note" class="small" style="margin-top:18px">Showing the fused mesh with its reconstructed vertex colors.</div></aside><section id="canvas">{figure}</section></main>
<script type="application/json" id="viewer-data">{payload}</script><script>{_VIEWER_SCRIPT}</script></body></html>'''
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(page,encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    return destination, metadata


_VIEWER_SCRIPT = r"""
function initializeViewer() {
const state = JSON.parse(document.getElementById('viewer-data').textContent);
const plot = document.getElementById('scene-view');
const occupancy = state.occupancy;
const modeTraces = state.mode_trace_indices;
let mode = 0;
const notes = [
  'Showing the fused mesh with its reconstructed vertex colors.',
  'Showing actual cubic cells near the TSDF surface. Shared internal faces are hidden; cell positions are unchanged.',
  'Showing the reconstructed surface point cloud: ' + state.meta.displayed_point_count + ' / ' + state.meta.point_count + ' points.',
  'Showing one layer of the separate ray-evidence grid. Gray: unknown; teal: free; orange: surface hit.'
];
let occupancyState = null;
if (occupancy) {
  occupancyState = Uint8Array.from(atob(occupancy.state_base64), c => c.charCodeAt(0));
}
function update() {
  const visible = plot.data.map(() => false);
  for (const i of modeTraces[String(mode)] || []) visible[i] = true;
  if (document.getElementById('show-cameras').checked) {
    for (const i of state.camera_indices) visible[i] = true;
  }
  Plotly.restyle(plot, {visible: visible});
  const controls = document.getElementById('occupancy-controls');
  if (controls) controls.hidden = mode !== 3;
  document.getElementById('mode-note').textContent = notes[mode];
}
function buildSlice(axis, slice) {
  const n = occupancy.resolution, size = occupancy.voxel_size_m, origin = occupancy.origin_m;
  const axes = [0,1,2].filter(a => a !== axis);
  const faces = [
    {axis:0, sign:-1, corners:[[0,0,0],[0,0,1],[0,1,1],[0,1,0]]},
    {axis:0, sign:1, corners:[[1,0,0],[1,1,0],[1,1,1],[1,0,1]]},
    {axis:1, sign:-1, corners:[[0,0,0],[1,0,0],[1,0,1],[0,0,1]]},
    {axis:1, sign:1, corners:[[0,1,0],[0,1,1],[1,1,1],[1,1,0]]},
    {axis:2, sign:-1, corners:[[0,0,0],[0,1,0],[1,1,0],[1,0,0]]},
    {axis:2, sign:1, corners:[[0,0,1],[1,0,1],[1,1,1],[0,1,1]]}
  ];
  const colors = ['rgb(100,116,139)','rgb(45,212,191)','rgb(251,146,60)'];
  const result = {x:[], y:[], z:[], i:[], j:[], k:[], facecolor:[]};
  const vertices = new Map();
  const sliceCounts = [0,0,0];
  for (let a=0; a<n; a++) for (let b=0; b<n; b++) {
    const index = [0,0,0]; index[axis]=slice; index[axes[0]]=a; index[axes[1]]=b;
    const code = occupancyState[(index[0]*n+index[1])*n+index[2]];
    sliceCounts[code]++;
    for (const face of faces) {
      // Within-slice neighbors share an invisible face, whatever their state.
      if (face.axis !== axis && !(face.sign < 0 ? index[face.axis] === 0 : index[face.axis] === n-1)) continue;
      const ids = face.corners.map(corner => {
        const grid = corner.map((v,k) => v+index[k]);
        const key = grid.join(',');
        if (!vertices.has(key)) {
          vertices.set(key, result.x.length);
          result.x.push(origin[0]+grid[0]*size);
          result.y.push(origin[1]+grid[1]*size);
          result.z.push(origin[2]+grid[2]*size);
        }
        return vertices.get(key);
      });
      result.i.push(ids[0],ids[0]); result.j.push(ids[1],ids[2]); result.k.push(ids[2],ids[3]);
      result.facecolor.push(colors[code],colors[code]);
    }
  }
  result.sliceCounts = sliceCounts;
  return result;
}
function renderSlice() {
  if (!occupancy) return;
  const axis = Number(document.getElementById('slice-axis').value);
  const slice = Number(document.getElementById('slice-index').value);
  const data = buildSlice(axis, slice);
  const changes = {};
  for (const key of ['x','y','z','i','j','k','facecolor']) changes[key] = [data[key]];
  const lower = occupancy.origin_m[axis]+slice*occupancy.voxel_size_m;
  document.getElementById('slice-label').textContent =
    ['X','Y','Z'][axis] + ' index ' + slice + ' / ' + (occupancy.resolution-1) +
    ', range [' + lower.toFixed(4) + ', ' + (lower+occupancy.voxel_size_m).toFixed(4) + ') m. ' +
    'Slice unknown / free / occupied = ' + data.sliceCounts.join(' / ');
  return Plotly.restyle(plot, changes, modeTraces['3']);
}
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  mode = Number(button.dataset.mode);
  document.querySelectorAll('[data-mode]').forEach(other => other.classList.toggle('active', other === button));
  update();
}));
document.getElementById('show-cameras').addEventListener('change', update);
document.getElementById('reset-view').addEventListener('click', () => Plotly.relayout(plot, {'scene.camera': state.default_camera}));
for (const camera of state.meta.cameras) {
  const item = document.createElement('li');
  item.textContent = camera.id + ' · ' + (camera.split === 'integration' ? 'integration' : 'evaluation');
  document.getElementById('camera-list').appendChild(item);
}
if (occupancy) {
  document.getElementById('slice-axis').addEventListener('change', renderSlice);
  // Commit a slider position on release; avoid rebuilding geometry on every pixel of a drag.
  document.getElementById('slice-index').addEventListener('change', renderSlice);
  renderSlice();
}
window.addEventListener('resize', () => Plotly.Plots.resize(plot));
}
if (window.wtumiPlotReady) initializeViewer();
else window.addEventListener('wtumi-plot-ready', initializeViewer, {once:true});
"""
