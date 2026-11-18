"""Portable image comparisons without a GPU or an OpenGL window."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def save_comparison(path, rgb_ref, depth_ref, rgb_pred, depth_pred, title, tolerance_m):
    valid_ref, valid_pred = depth_ref > 0, depth_pred > 0
    overlap = valid_ref & valid_pred
    missing, extra = valid_ref & ~valid_pred, valid_pred & ~valid_ref
    values = np.concatenate([depth_ref[valid_ref], depth_pred[valid_pred]])
    low, high = (float(values.min()), float(values.max())) if values.size else (0, 1)
    span = max(high - low, 0.1)

    def depth_panel(depth):
        intensity = np.where(depth > 0, 55 + 190 * (1 - np.clip((depth - low) / span, 0, 1)), 0)
        return np.repeat(intensity.astype(np.uint8)[..., None], 3, axis=2)

    error = np.zeros_like(rgb_ref)
    strength = np.clip(np.abs(depth_pred - depth_ref) / tolerance_m, 0, 1)
    error[overlap, 0] = (strength[overlap] * 255).astype(np.uint8)
    error[missing], error[extra] = [255, 0, 255], [0, 255, 255]
    rgb_error = np.abs(rgb_pred.astype(float) - rgb_ref.astype(float)).astype(np.uint8)
    rgb_error[~overlap] = 0
    rgb_error[missing], rgb_error[extra] = [255, 0, 255], [0, 255, 255]
    panels = [rgb_ref, rgb_pred, rgb_error, depth_panel(depth_ref), depth_panel(depth_pred), error]
    labels = ["Reference RGB", "Reconstructed RGB", "RGB absolute error",
              "Reference depth (shared scale)", "Reconstructed depth", "Depth error / missing / extra"]
    height, width = depth_ref.shape
    scale = max(1, int(np.ceil(300 / width)))
    cell_w, cell_h = width * scale, height * scale + 28
    canvas = Image.new("RGB", (3 * cell_w, 2 * cell_h + 80), "#151b24")
    draw = ImageDraw.Draw(canvas)
    # Keep panel labels readable as camera resolution grows.
    import matplotlib
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),
                              max(11, min(22, cell_w // 30)))
    draw.text((12, 10), title, fill="white", font=font)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (i % 3) * cell_w, 32 + (i // 3) * cell_h
        draw.text((x + 8, y + 3), label, fill="white", font=font)
        canvas.paste(Image.fromarray(panel).resize((width * scale, height * scale), Image.Resampling.NEAREST), (x, y + 24))
    draw.text((12, canvas.height - 30), f"Black: zero error/background | Magenta: missing | Cyan: extra | Full red: >= {tolerance_m*1000:g} mm depth error", fill="white", font=font)
    canvas.save(path)
