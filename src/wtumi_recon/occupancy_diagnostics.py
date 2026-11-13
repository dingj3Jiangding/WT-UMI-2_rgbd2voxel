"""Static orthogonal occupancy slices with explicit axes and world units."""

import numpy as np
from PIL import Image, ImageDraw


PALETTE = np.array([[71, 85, 105], [20, 184, 166], [251, 146, 60]], dtype=np.uint8)


def save_occupancy_slices(path, grid):
    n = grid.resolution
    mid = n // 2
    cell, top, left, display = 380, 110, 48, 300
    canvas = Image.new("RGB", (cell*3, 480), "#101827")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Occupancy evidence: three middle slices (one cell thick)", fill="white")
    for i, (name, color) in enumerate(zip(("Unknown", "Free ray evidence", "Occupied hit cell"), PALETTE)):
        x = 16+i*240
        draw.rectangle((x, 35, x+14, 49), fill=tuple(color.tolist()))
        draw.text((x+21, 35), name, fill="white")
    summary = grid.summary()
    draw.text((16, 60), f"Voxel {grid.voxel_size_m*1000:.3f} mm | N={n} | Unknown={summary['unknown_cells']} Free={summary['free_cells']} Occupied={summary['occupied_cells']} Conflicts={summary['conflict_cells']}", fill="white")
    planes = [("X", 0, grid.state[mid, :, :], 2, 1),
              ("Y", 1, grid.state[:, mid, :].T, 0, 2),
              ("Z", 2, grid.state[:, :, mid].T, 0, 1)]
    for i, (name, axis, pixels, horizontal, vertical) in enumerate(planes):
        x = i*cell+left
        coordinate = grid.origin_m[axis] + (mid+.5)*grid.voxel_size_m
        draw.text((x, top-25), f"{name} = {coordinate:.4f} m (index {mid})", fill="white")
        picture = Image.fromarray(PALETTE[pixels]).resize((display, display), Image.Resampling.NEAREST)
        canvas.paste(picture, (x, top))
        draw.rectangle((x-1, top-1, x+display, top+display), outline="#94a3b8")
        h_axis, v_axis = "XYZ"[horizontal], "XYZ"[vertical]
        low = grid.origin_m[horizontal]
        high = low+n*grid.voxel_size_m
        draw.text((x, top+display+8), f"{h_axis}: {low:g} -> {high:g} m (right)", fill="white")
        low = grid.origin_m[vertical]
        high = low+n*grid.voxel_size_m
        draw.text((x, top+display+25), f"{v_axis}: {low:g} -> {high:g} m (down)", fill="white")
    draw.text((16, 462), "Unknown is not free. Occupied cells dominate conflicts. Free evidence is not whole-cell collision clearance.", fill="#cbd5e1")
    canvas.save(path)
