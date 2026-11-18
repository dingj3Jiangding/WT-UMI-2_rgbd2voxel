# M7 high-quality PACE result

Source: `outputs/m7_hq_pace_20261117_021425/`. This is one static staircase scene with two observation histories, captured using MuJoCo EGL on PACE with Python 3.12.5 and Open3D 0.19.0.

Each history integrates 61 actual RGB-D frames at 640 by 480 resolution. One history always carries the box; the other has 30 clear frames followed by 31 box-occluded frames. Camera poses, timestamps and two held-out evaluation views are shared. The observation histories intentionally contain different early visual information.

| Final measure | Always carrying the box | Observe first, then carry the box |
|---|---:|---:|
| Stair depth coverage | 37.0739% | 94.8320% |
| Mean absolute depth error on overlap | 2.0220 millimetres | 3.6368 millimetres |
| Root mean square depth error on overlap | 24.3265 millimetres | 32.7234 millimetres |
| Stair surface samples within 30 millimetres of the mesh | 1015/1960 (51.7857%) | 1960/1960 (100%) |
| Unknown cells in the specified region | 73.4609% | 53.8776% |
| Final common preobserved, currently hidden samples represented within tolerance | 455/1400 (32.5%) | 1400/1400 (100%) |

Previously observed static geometry remains represented after the box blocks the view. The always-occluded history does not automatically recover unobserved geometry. A smaller overlap-only depth error in the less complete condition does not establish a better map. The 100% result describes finite sampled surfaces within a tolerance, not exact geometry, complete scene occupancy or collision clearance. Unknown volumes include hidden space and object interiors.

Local validation of the downloaded files found all referenced output assets, all 122 cumulative snapshots and valid paired captures. Each of the three MP4 files decoded to 61 distinct frames at 15 frames per second. Playback duration is approximately 4.07 seconds. Total recorded experiment time was 174.7024 seconds, including capture, both histories, evaluation and export; this is not a real-time throughput measurement. Per-case final-row timing covers only the final incremental snapshot call.

The current result uses known simulator poses and masks, a static environment and prescribed kinematic motion. It includes no physical walking, pickup, localization error or tactile fusion. It should not be treated as a controlled comparison against the earlier five-frame M7 run, since image resolution, frame count and clear-prefix schedule all changed.
