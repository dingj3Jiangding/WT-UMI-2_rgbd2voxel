# M3: Voxel resolution and depth perturbation comparison

Historical record: generated M3 output directories were removed on 2026-11-17. Configurations and tests are retained; rerun the experiment before opening the outputs linked below.

Nine static door experiments were completed: grid resolutions of 80, 160, and 240, each with clean depth, Gaussian depth noise with a 5 mm standard deviation, or the same noise plus 10% random dropout. The volume edge length was fixed at 2.5 m, TSDF truncation distance at 0.06 m, with three fusion frames and one novel evaluation view.

## Results

| Input | Voxel edge mm | Coverage % | Depth MAE mm | Depth RMSE mm | Coverage within 20 mm % |
|---|---:|---:|---:|---:|---:|
| Clean | 31.250 | 75.16 | 4.121 | 23.722 | 72.40 |
| Clean | 15.625 | 98.79 | 2.630 | 19.534 | 97.37 |
| Clean | 10.417 | 99.22 | 2.491 | 19.322 | 97.64 |
| 5 mm noise | 31.250 | 74.82 | 4.587 | 21.672 | 72.00 |
| 5 mm noise | 15.625 | 98.73 | 3.365 | 20.695 | 97.33 |
| 5 mm noise | 10.417 | 99.13 | 3.203 | 18.481 | 97.73 |
| 5 mm noise + 10% dropout | 31.250 | 72.37 | 4.666 | 22.200 | 69.69 |
| 5 mm noise + 10% dropout | 15.625 | 96.57 | 3.512 | 21.041 | 95.06 |
| 5 mm noise + 10% dropout | 10.417 | 97.53 | 3.465 | 19.549 | 96.00 |

At 31.25 mm, this configuration misses a substantial part of the floor; 15.625 mm improves coverage considerably. Reducing the voxel edge further from 15.625 mm to 10.417 mm raises clean-input coverage by only about 0.43 percentage points, while increasing the grid from 4,096,000 to 13,824,000 cells (3.375 times as many). The default remains 15.625 mm; whether a higher resolution is worthwhile depends on the details that need to be reconstructed.

At 15.625 mm, adding 5 mm depth noise increases MAE from 2.630 mm to 3.365 mm. Adding 10% random dropout as well reduces coverage to 96.57%, with an MAE of 3.512 mm. Holes in the door panel and floor are visible in the overview image.

MAE/RMSE are computed only where both reference and prediction have valid depth. Coverage and coverage within the threshold use all valid reference pixels as the denominator. RMSE sometimes decreases after noise is added; changes in the error distribution and overlap pixel set can contribute to this, so a lower RMSE alone does not establish a better reconstruction.

Only seed=7 was used. These results do not establish real-camera accuracy, robustness to noise, or local handle accuracy. Total runtime includes file export and is affected by system load; this run was not an independent performance benchmark. Cell counts are not measured memory usage.

## Files and reproduction

Results are in `outputs/m3_door_quality/`: `overview.png` summarizes RGB and depth error, `report.md` lists individual runs, and `summary.csv` and `summary.json` contain values and grouped statistics. Complete outputs for each case are stored in `runs/`; the existing `view --result ...` command can generate a 3D viewer for them.

```bash
open outputs/m3_door_quality/overview.png
./.venv/bin/python -m wtumi_recon experiment --config configs/experiments/door_quality.yaml
```

The experiment command creates a new timestamped output directory by default, preserving these results. To change the base scene, edit `base_config` in the experiment YAML. To add random seeds, edit `seeds`; clean inputs are not run repeatedly.

## Validation

- 51 tests passed, including an actual small multi-seed experiment.
- Saved fusion depths match the depths passed to TSDF pixel for pixel; clean references are stored separately.
- All resolutions share the same noisy inputs, and all experiments share clean held-out references. Depth-file hash checks passed.
- The noise-free 160³ result exactly reproduces the original M2 door coverage, MAE, RMSE, and missing/extra pixel counts.
- The static overview PNG was inspected for readable layout and labels.

At this milestone, a complete free/occupied/unknown map, real-data adapters, dynamic scenes, and an independent quantitative evaluation of the handle region remained unimplemented.
