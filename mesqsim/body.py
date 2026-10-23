"""Pods, segments and the 14-joint tree of the Mesquite suit (brief §3.1 and §3.4).

Per-pod arrays in this project put the pod id on an axis of length :data:`N_PODS` (= 15, ids 0-14).
Per-joint arrays put the joint on an axis of length :data:`N_JOINTS` (= 14) in the order of
:data:`JOINTS`, which lists every parent before its children (outward from the pelvis).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Pod",
    "Joint",
    "PODS",
    "UNRENDERED_PODS",
    "N_PODS",
    "ROOT_POD",
    "POD_BY_NAME",
    "JOINTS",
    "N_JOINTS",
    "JOINT_BY_NAME",
    "JOINT_TYPES",
]


@dataclass(frozen=True)
class Pod:
    id: int
    name: str
    segment: str


@dataclass(frozen=True)
class Joint:
    name: str
    parent: int  # pod id
    child: int  # pod id
    jtype: str  # "3dof", "2dof" or "hinge"


PODS: tuple[Pod, ...] = (
    Pod(0, "Head", "head"),
    Pod(1, "Spine", "chest"),
    Pod(2, "HipsAlt", "pelvis"),
    Pod(3, "LeftArm", "left upper arm"),
    Pod(4, "LeftForeArm", "left forearm"),
    Pod(5, "LeftHand", "left hand"),
    Pod(6, "RightArm", "right upper arm"),
    Pod(7, "RightForeArm", "right forearm"),
    Pod(8, "RightHand", "right hand"),
    Pod(9, "LeftUpLeg", "left thigh"),
    Pod(10, "LeftLeg", "left shin"),
    Pod(11, "LeftFoot", "left foot"),
    Pod(12, "RightUpLeg", "right thigh"),
    Pod(13, "RightLeg", "right shin"),
    Pod(14, "RightFoot", "right foot"),
)
"""The 15 rendered pods, indexed by pod id."""

UNRENDERED_PODS: tuple[Pod, ...] = (
    Pod(15, "LeftShoulder", "left shoulder blade"),
    Pod(16, "RightShoulder", "right shoulder blade"),
)
"""Ids 15 and 16 exist in the firmware but are not rendered and are not simulated."""

N_PODS = len(PODS)
ROOT_POD = 2  # HipsAlt, the pelvis
POD_BY_NAME = {p.name: p for p in PODS}

JOINTS: tuple[Joint, ...] = (
    Joint("lower_back", 2, 1, "3dof"),
    Joint("neck", 1, 0, "3dof"),
    Joint("l_shoulder", 1, 3, "3dof"),
    Joint("r_shoulder", 1, 6, "3dof"),
    Joint("l_elbow", 3, 4, "2dof"),
    Joint("r_elbow", 6, 7, "2dof"),
    Joint("l_wrist", 4, 5, "2dof"),
    Joint("r_wrist", 7, 8, "2dof"),
    Joint("l_hip", 2, 9, "3dof"),
    Joint("r_hip", 2, 12, "3dof"),
    Joint("l_knee", 9, 10, "hinge"),
    Joint("r_knee", 12, 13, "hinge"),
    Joint("l_ankle", 10, 11, "2dof"),
    Joint("r_ankle", 13, 14, "2dof"),
)
"""The 14 joints, parent -> child, ordered so every parent pod is reached before its children."""

N_JOINTS = len(JOINTS)
JOINT_BY_NAME = {j.name: j for j in JOINTS}
JOINT_TYPES = ("3dof", "2dof", "hinge")


def _check_tree() -> None:
    reached = {ROOT_POD}
    for j in JOINTS:
        if j.parent not in reached:
            raise AssertionError(f"joint {j.name}: parent {j.parent} not reached before it")
        if j.child in reached:
            raise AssertionError(f"joint {j.name}: child {j.child} reached twice")
        reached.add(j.child)
    if reached != {p.id for p in PODS}:
        raise AssertionError("the joint tree does not span the 15 rendered pods")


_check_tree()
