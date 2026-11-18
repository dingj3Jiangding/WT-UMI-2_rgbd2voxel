# M7: visual memory before and after box occlusion

M7 compares two observation histories of the same static staircase. Both use the same five poses, timestamps, calibration, reconstruction settings and two held-out evaluation views. The first history has box occlusion throughout. The second has two no-box observations followed by three observations with the box. Every frame contributes to the environment map only after self/payload masking.

Both histories use the same mapping algorithm; the observe-first condition intentionally receives additional early visual information. Paired simulator captures are scheduled into two portable sequences. The transition does not simulate lifting a box, locomotion or stair climbing.

## Run

From the project root with the virtual environment active, capture a fresh paired example and run M7:

```bash
python scripts/run_memory_demo.py
```

To use previously captured M6 observations without launching MuJoCo:

```bash
python scripts/run_memory_demo.py \
  --capture outputs/occlusion_demo_20261116_184221_889717/capture
```

The source capture is checked against the selected camera/scene configuration, copied into the new result, and left unchanged. The default result is `outputs/memory_demo_<timestamp>/`. Use `--output` to select a new or empty directory. A source capture and output directory may not overlap. `--config configs/demos/stairs_memory.yaml` controls the pre-observation count, map sizes and tolerances. The pre-observation count must leave at least one later box frame; the default replay mode is bounded to 20 input poses. With `incremental_prefixes: true`, up to 100 actual input poses are supported. See [High-quality playback on PACE](m7_high_quality.md) for a 61-frame, 640 x 480 preset and MP4 export.

The package entry point is `python -m wtumi_recon memory-demo --config configs/demos/stairs_memory.yaml`. It also accepts `--capture` and `--output`.

## Inspect

Open `index.html` from the output directory in a browser. All resources are local and portable with the full result folder.

- `history_viewer.html`: both cumulative 3D maps with Play/Pause, a shared frame slider, fixed bounds and linked orbit controls. The frame selector changes the number of fused observations. Geometry is read from real prefix reconstructions, with no invented interpolation.
- `runs/always_occluded/viewer.html` and `runs/observe_then_occluded/viewer.html`: final mesh, surface voxels, point cloud, camera poses and occupancy slices. These viewers' occupancy sliders select space, not time.
- `capture_history.gif`: actual raw and masked inputs; the observe-first column switches to box observations after frame two.
- `reconstruction_history.gif`: cumulative maps rendered at one shared evaluation camera. Reported depth metrics pool both held-out cameras.
- `occupancy_history.gif`: a fixed center-X slice showing unknown, free-ray and surface-hit evidence over time.
- `observation_progress.png`: per-frame depth coverage, surface completeness and unknown-region curves.
- `summary.csv`, `observation_history.json`, `memory_retention.json` and `demo.json`: measured final/per-frame results, capture provenance and validation records.

Earlier prefixes are saved under `history/<case>/frame_001/`, etc.; final maps are under `runs/<case>/`. Each prefix contains its cumulative input list. The default mode reconstructs each prefix independently. The high-quality preset integrates each observation once per condition and exports intermediate held-out evaluations without a dense TSDF archive; the final result has the full export. Neither mode establishes real-time throughput. Saved relative replay configurations can reconstruct final or intermediate maps after moving the entire output directory.

## Metric interpretation

Stair depth coverage, overlap-only depth errors, sampled tread/riser completeness and fixed-region occupancy follow the M6 definitions. Coverage retains missing predictions in its denominator; a small error on a sparse prediction does not imply a complete map. Unknown-region fractions include hidden volumes and solid interiors.

The additional memory diagnostic uses a common cohort for both histories: analytic stair samples that agree with observed no-box depth during the prefix and are behind an observed box pixel at the current frame. A cohort sample is represented when it lies within the configured distance of the current reconstructed mesh. Before the transition there is no post-switch cohort, so the fraction is null, not zero. The cohort can change as the camera moves; sample counts and hashes are recorded for each frame.

Visibility is a nearest-pixel, camera-Z consistency approximation. The metric measures sampled geometric representation, not retention of identical mesh triangles, exact surface area or collision safety. Unobserved analytic samples do not imply that their entire finite occupancy cells are unknown.

Only selected observations enter fusion. Clean integration references from source datasets are not carried into the mixed history. Held-out references, counterfactual no-box images used in diagnostics and analytic stair samples are evaluation data. They do not fill masked pixels or hidden geometry.

## PACE and verification

Sync code using the existing source sync script. On an allocated compute node with the project environment active, saved-capture replay needs no EGL capture:

```bash
MPLBACKEND=Agg python scripts/run_memory_demo.py \
  --capture outputs/occlusion_demo_20261116_184221_889717/capture
```

For a fresh capture on an allocated GPU node:

```bash
MUJOCO_GL=egl MPLBACKEND=Agg python scripts/run_memory_demo.py
```

Focused tests cover observation selection, masks, shared cohorts, held-out isolation, cumulative prefixes, source preservation, relocated replay and temporal visualization:

```bash
python -m pytest -q tests/test_memory_sequence.py tests/test_memory_demo.py tests/test_memory_visuals.py
```

M7 uses known simulator poses and masks with a static environment. Dynamic clearing, pose drift, physical pickup and tactile fusion remain outside this milestone.
