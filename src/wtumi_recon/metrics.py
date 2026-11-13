"""Metrics for RGB-D reconstruction round trips."""

from __future__ import annotations

import numpy as np


def depth_metrics(reference_m: np.ndarray, prediction_m: np.ndarray) -> dict[str, float]:
    """Compare two depth images in metres, ignoring zero-valued invalid pixels."""

    reference = np.asarray(reference_m, dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction must have the same shape")

    ref_valid = np.isfinite(reference) & (reference > 0)
    pred_valid = np.isfinite(prediction) & (prediction > 0)
    overlap = ref_valid & pred_valid
    ref_count = int(ref_valid.sum())
    overlap_count = int(overlap.sum())
    if overlap_count == 0:
        return {
            "reference_valid_pixels": float(ref_count),
            "prediction_valid_pixels": float(pred_valid.sum()),
            "overlap_pixels": 0.0,
            "coverage": 0.0,
            "mae_m": float("inf"),
            "rmse_m": float("inf"),
        }

    error = prediction[overlap] - reference[overlap]
    return {
        "reference_valid_pixels": float(ref_count),
        "prediction_valid_pixels": float(pred_valid.sum()),
        "overlap_pixels": float(overlap_count),
        "coverage": float(overlap_count / max(ref_count, 1)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
    }
