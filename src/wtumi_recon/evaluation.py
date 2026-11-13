"""RGB-D evaluation with explicit visibility masks and standard JSON values."""

import numpy as np


def evaluate_rgbd(rgb_ref, depth_ref, rgb_pred, depth_pred, tolerance_m=0.02):
    """Errors on overlap; missing/extra pixels and inlier coverage separately."""
    a, b = np.asarray(depth_ref, dtype=float), np.asarray(depth_pred, dtype=float)
    ar, br = np.asarray(rgb_ref, dtype=float), np.asarray(rgb_pred, dtype=float)
    if a.ndim != 2 or b.shape != a.shape or ar.shape != a.shape + (3,) or br.shape != ar.shape:
        raise ValueError("RGB-D shapes must be aligned (H,W) and (H,W,3)")
    va, vb = np.isfinite(a) & (a > 0), np.isfinite(b) & (b > 0)
    both, union = va & vb, va | vb
    n, total = int(both.sum()), int(va.sum())
    error = b[both] - a[both]
    color_error = (br[both] - ar[both]) / 255.0
    color_mse = float(np.mean(color_error**2)) if n else None
    return {
        "status": "ok" if n else "no_overlap",
        "reference_valid_pixels": total, "prediction_valid_pixels": int(vb.sum()),
        "overlap_pixels": n, "missing_pixels": int((va & ~vb).sum()),
        "extra_pixels": int((vb & ~va).sum()),
        "coverage": n / total if total else None,
        "silhouette_iou": n / int(union.sum()) if union.any() else None,
        "mae_m": float(np.abs(error).mean()) if n else None,
        "rmse_m": float(np.sqrt(np.mean(error**2))) if n else None,
        "depth_tolerance_m": tolerance_m,
        "accurate_reference_pixels": int((np.abs(error) <= tolerance_m).sum()),
        "accurate_reference_fraction": int((np.abs(error) <= tolerance_m).sum()) / total if total else None,
        "rgb_mae_0_1": float(np.abs(color_error).mean()) if n else None,
        "rgb_psnr_db": float(-10 * np.log10(color_mse)) if color_mse is not None and color_mse > 0 else None,
        "rgb_exact_match": color_mse == 0 if n else None,
    }


def summarize_views(rows):
    """Pool pixels rather than weighting empty and dense views equally."""
    tolerances = {r["depth_tolerance_m"] for r in rows}
    if len(tolerances) > 1:
        raise ValueError("cannot pool depth inliers with different tolerances")
    total = sum(r["reference_valid_pixels"] for r in rows)
    n = sum(r["overlap_pixels"] for r in rows)
    extra = sum(r["extra_pixels"] for r in rows)
    accurate = sum(r["accurate_reference_pixels"] for r in rows)
    weighted = lambda key: sum(r[key] * r["overlap_pixels"] for r in rows if r[key] is not None) / n if n else None
    return {
        "views": len(rows), "reference_valid_pixels": total, "overlap_pixels": n,
        "coverage": n / total if total else None,
        "silhouette_iou": n / (total + extra) if total + extra else None,
        "accurate_reference_pixels": accurate,
        "accurate_reference_fraction": accurate / total if total else None,
        "depth_tolerance_m": next(iter(tolerances)) if tolerances else None,
        "mae_m": weighted("mae_m"),
        "rmse_m": float(np.sqrt(sum(r["rmse_m"]**2 * r["overlap_pixels"] for r in rows if r["rmse_m"] is not None) / n)) if n else None,
        "rgb_mae_0_1": weighted("rgb_mae_0_1"),
        "missing_pixels": sum(r["missing_pixels"] for r in rows),
        "extra_pixels": sum(r["extra_pixels"] for r in rows),
    }
