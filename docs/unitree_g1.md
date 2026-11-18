# Unitree G1 in the scene reconstruction workflow

The `g1_stairs` template replaces the blue torso proxy with the actual link meshes of Unitree G1. It includes the robot, a box, a head-mounted virtual RGB-D camera, a finite floor and a staircase.

The `room` and `doorway` templates also use G1 in a fixed carry pose with a carried box, matching the stairs setup. Their room interiors measure 6.4 x 10.0 m; the robot model retains its original scale and provisional camera mount. Robot and box pixels are excluded from environment mapping. The doorway includes a separate operable hinge. See [the quick-scene guide](quick_scenes.md) for layouts and the three-scene experiment. Older saved results retain their original pose and box settings until recaptured.

## View, customize and reconstruct

From the project root with the existing environment activated:

```bash
python scripts/scene.py view --scene g1_stairs
```

Press `C` for the camera view and `P` to replay the prescribed rig path. Close the window before issuing the next terminal command.

Create an editable copy:

```bash
python scripts/scene.py create --template g1_stairs --output scenes/my_g1
```

Edit `scenes/my_g1/scene.yaml`. The copied bundle includes its robot model assets. For example:

```yaml
robot:
  model: unitree_g1
  pose: carry
  base_height_m: 0.793
  joint_positions_degrees: {}
torso:
  enabled: false
```

The snippet illustrates the relevant fields; retain the other fields in the generated file. `pose` accepts `stand` or `carry`. Optional `joint_positions_degrees` entries override individual source-model hinge joints and are checked against their joint ranges. The blue proxy must be disabled when a robot model is selected.

Reconstruct and open the saved result:

```bash
python scripts/scene.py run --scene scenes/my_g1 --output outputs/my_g1_run
python scripts/scene.py view --result outputs/my_g1_run
```

Result-viewer controls remain `1` original, `2` mesh, `3` observed TSDF surface voxels, `4` overlay, `P` play/pause, `[` / `]` step, `C` camera/free view and `I` saved input/comparison images. The result's `scene/model.xml` can also be loaded directly into a standard MuJoCo viewer.

## Model source and adaptation

- Source: [MuJoCo Menagerie, unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_g1).
- Pinned commit: `370c43965323c0944b433b202aac5058b5f907a9`.
- Source model: `g1_29dof_rev_1_0`, with 29 hinge joints plus a floating base and 35 visual mesh assets.
- Upstream description derives from [Unitree's official G1 MJCF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml).
- License: Unitree BSD-3-Clause, retained alongside `SOURCE.json` with source URLs and checksums.

The complete original articulated XML, its companion scene, meshes and license remain under `assets/robots/unitree_g1/`. The quick-scene builder evaluates its joint pose with forward kinematics, preserves the source visual geometry, and writes a fixed visual hierarchy under `sensor_rig`. Source +X-forward is rotated to the project's +Y-forward convention. Visual and collision meshes are not rendered twice; the capture scene has no active robot joints or actuators.

This adaptation serves camera occlusion and RGB-D mapping. It is not a walking controller, a validated carrying motion, a physical grasp, or a dynamics-accurate integration of the articulated robot. The original model is available for future controller work separately.

## Camera, box and masking

The source G1 reaches about 1.324 m in the zero-leg pose at pelvis height 0.793 m. This template places the camera at `[0, 0.05, 1.36]` metres in the ground-origin rig frame. It is a provisional virtual mount above the head, not calibrated factory sensor placement. Keep the camera mounting fields configurable until hardware extrinsics are known.

The box measures 0.5 x 0.4 x 0.4 m and is centered at `[0, 0.36, 0.95]` m. The carry pose places the arms around it for visual inspection; no force or contact constraint attaches it to the hands. Independent head-scan angles are zero in this template, since no movable head-camera mechanism has been specified.

Every G1 visual link belongs to robot group 1. Segmentation excludes all robot links and the box from RGB-D fusion. Environment-only held-out references hide both groups. The map therefore contains only geometry actually observed outside the robot/payload silhouettes.

## PACE

The existing sync script includes the new template and the approximately 20 MB robot asset bundle. No new dependency is required beyond the project's existing MuJoCo extra. After syncing and entering an allocated GPU node:

```bash
MUJOCO_GL=egl MPLBACKEND=Agg python scripts/scene.py run --scene g1_stairs --output outputs/g1_stairs_pace
```

Download the entire result directory and inspect it on the Mac. It includes copied robot assets and portable scene paths.
