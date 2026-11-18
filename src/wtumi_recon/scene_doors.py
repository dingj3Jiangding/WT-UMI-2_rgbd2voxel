"""Configurable hinged door geometry for scene preview and static RGB-D capture.

Door angles are set kinematically. The helpers do not step physics, actuate a
handle, enforce contact, or infer a robot interaction with the door.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET

import numpy as np

from .rig_capture import _number, _vector


def normalize_doors(value):
    """Validate portable door descriptions without importing MuJoCo."""
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("environment.doors must be a list containing at most 32 doors")
    required = {"name", "hinge_position_m", "width_m", "height_m", "thickness_m"}
    defaults = {"closed_yaw_degrees": 0., "angle_degrees": 0.,
                "range_degrees": [0., 110.], "color_rgba": [.62, .40, .20, 1.]}
    result, names = [], set()
    for item in value:
        if not isinstance(item, dict) or set(item) - required - set(defaults) or not required <= set(item):
            raise ValueError("each door requires name, hinge_position_m, width_m, height_m and thickness_m; "
                             "optional fields are closed_yaw_degrees, angle_degrees, range_degrees and color_rgba")
        door = {**copy.deepcopy(defaults), **copy.deepcopy(item)}
        name = door["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError("door.name must be a safe alphanumeric/underscore/hyphen string")
        if name in names:
            raise ValueError(f"duplicate door name: {name}")
        names.add(name)
        door["hinge_position_m"] = _vector(door["hinge_position_m"], f"door {name}.hinge_position_m")
        for key in ("width_m", "height_m", "thickness_m"):
            door[key] = _number(door[key], f"door {name}.{key}", True)
        door["closed_yaw_degrees"] = _number(door["closed_yaw_degrees"], f"door {name}.closed_yaw_degrees")
        limits = door["range_degrees"]
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError("door.range_degrees must contain lower and upper angles")
        door["range_degrees"] = [_number(angle, "door.range_degrees") for angle in limits]
        if not door["range_degrees"][0] < door["range_degrees"][1]:
            raise ValueError("door.range_degrees lower angle must be less than upper angle")
        door["angle_degrees"] = _angle(door, door["angle_degrees"])
        rgba = door["color_rgba"]
        if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
            raise ValueError("door.color_rgba must contain four values in [0, 1]")
        door["color_rgba"] = [_number(v, "door.color_rgba") for v in rgba]
        if any(not 0 <= v <= 1 for v in door["color_rgba"]):
            raise ValueError("door.color_rgba values must be in [0, 1]")
        result.append(door)
    return result


def _angle(door, value):
    angle = _number(value, f"door {door['name']}.angle_degrees")
    lower, upper = door["range_degrees"]
    if not lower <= angle <= upper:
        raise ValueError(f"door {door['name']} angle must be within range_degrees [{lower}, {upper}]")
    return angle


def _values(values):
    return " ".join(f"{float(v):.17g}" for v in values)


def append_scene_doors(world, doors):
    """Append physical hinge coordinates and visual leaf geometry to MJCF.

    qpos is the absolute opening angle. Body yaw includes the configured angle
    and joint ref initializes qpos0 to it, so plain MjData and standard viewers
    display the selected state without a custom initialization script.
    """
    for door in doors:
        name = "door_" + door["name"]
        body = ET.SubElement(world, "body", name=name,
                             pos=_values(door["hinge_position_m"]),
                             euler=_values([0, 0, door["closed_yaw_degrees"] + door["angle_degrees"]]))
        ET.SubElement(body, "joint", name=name + "_hinge", type="hinge", axis="0 0 1",
                      limited="true", range=_values(door["range_degrees"]),
                      ref=f"{door['angle_degrees']:.17g}", damping="1")
        ET.SubElement(body, "geom", name=name + "_leaf", type="box", group="0",
                      pos=_values([door["width_m"] / 2, 0, door["height_m"] / 2]),
                      size=_values([door["width_m"] / 2, door["thickness_m"] / 2, door["height_m"] / 2]),
                      rgba=_values(door["color_rgba"]), contype="0", conaffinity="0")
        # Decorative lever handles rotate with the leaf. They have no latch
        # mechanism and do not imply a grasp or force-controlled interaction.
        handle_x, handle_z = .86 * door["width_m"], .50 * door["height_m"]
        reach = door["thickness_m"] / 2 + .045
        ET.SubElement(body, "geom", name=name + "_handle_spindle", type="capsule", group="0",
                      fromto=_values([handle_x, -reach, handle_z, handle_x, reach, handle_z]),
                      size="0.008", rgba="0.72 0.74 0.76 1", contype="0", conaffinity="0")
        for side, y in (("front", -reach), ("back", reach)):
            ET.SubElement(body, "geom", name=name + "_handle_" + side, type="box", group="0",
                          pos=_values([handle_x - .04, y, handle_z]), size="0.05 0.012 0.012",
                          rgba="0.72 0.74 0.76 1", contype="0", conaffinity="0")


def configured_door_angles(cfg):
    return {door["name"]: door["angle_degrees"] for door in cfg.get("environment", {}).get("doors", [])}


def scene_door_ids(mujoco, model, cfg):
    """Validate the generated model against the saved door configuration."""
    result = {}
    for door in cfg.get("environment", {}).get("doors", []):
        name = "door_" + door["name"]
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_hinge"))
        geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + "_leaf"))
        if min(body, joint, geom) < 0:
            raise ValueError(f"scene model is missing configured door: {door['name']}")
        qpos = int(model.jnt_qposadr[joint])
        dof = int(model.jnt_dofadr[joint])
        yaw = np.deg2rad(door["closed_yaw_degrees"] + door["angle_degrees"]) / 2
        quat = np.array([np.cos(yaw), 0., 0., np.sin(yaw)])
        checks = [model.body_parentid[body] == 0,
                  model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE,
                  model.jnt_bodyid[joint] == body, model.jnt_limited[joint],
                  model.geom_bodyid[geom] == body, model.geom_group[geom] == 0,
                  model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX,
                  np.allclose(model.body_pos[body], door["hinge_position_m"], atol=1e-10, rtol=0),
                  np.isclose(abs(np.dot(model.body_quat[body], quat)), 1, atol=1e-10, rtol=0),
                  np.allclose(model.jnt_pos[joint], 0, atol=1e-10, rtol=0),
                  np.allclose(model.jnt_axis[joint], [0, 0, 1], atol=1e-10, rtol=0),
                  np.allclose(model.jnt_range[joint], np.deg2rad(door["range_degrees"]), atol=1e-10, rtol=0),
                  np.isclose(model.qpos0[qpos], np.deg2rad(door["angle_degrees"]), atol=1e-10, rtol=0),
                  np.allclose(model.geom_pos[geom], [door["width_m"] / 2, 0, door["height_m"] / 2], atol=1e-10, rtol=0),
                  np.allclose(model.geom_size[geom], np.array([door["width_m"], door["thickness_m"], door["height_m"]]) / 2,
                              atol=1e-10, rtol=0)]
        if not all(checks):
            raise ValueError(f"scene model disagrees with configured door: {door['name']}")
        result[door["name"]] = {"body": body, "joint": joint, "geom": geom, "qpos": qpos, "dof": dof}
    actual = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
              for i in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("door_")}
    expected = {"door_" + name + suffix for name in result
                for suffix in ("_leaf", "_handle_spindle", "_handle_front", "_handle_back")}
    if actual != expected:
        raise ValueError("scene model contains door geometry without matching configuration")
    for name, ids in result.items():
        for suffix in ("_handle_spindle", "_handle_front", "_handle_back"):
            geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "door_" + name + suffix))
            if model.geom_bodyid[geom] != ids["body"] or model.geom_group[geom] != 0:
                raise ValueError(f"door handles must belong to door {name} and environment group 0")
    return result


def set_scene_doors(mujoco, model, data, cfg, angles=None):
    """Set configured angles or supplied per-door overrides; caller forwards.

    This never steps physics. Each call starts with configured angles, so an
    unmentioned door cannot retain an accidental interactive preview state.
    """
    ids = scene_door_ids(mujoco, model, cfg)
    if angles is None:
        angles = {}
    if not isinstance(angles, dict) or set(angles) - set(ids):
        raise ValueError("door angle overrides must map configured door names to angles")
    desired = {door["name"]: _angle(door, angles.get(door["name"], door["angle_degrees"]))
               for door in cfg.get("environment", {}).get("doors", [])}
    for name, angle in desired.items():
        data.qpos[ids[name]["qpos"]] = np.deg2rad(angle)
        data.qvel[ids[name]["dof"]] = 0
    return desired
