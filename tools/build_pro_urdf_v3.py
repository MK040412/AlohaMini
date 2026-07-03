#!/usr/bin/env python3
"""Build aloha_mini_pro_v3.urdf: restore joint6 (wrist roll) and mount the
parallel gripper DISTALLY.

pro_v2 consumed the arm's 6th joint when the parallel gripper was grafted:
joint5 connects link4 directly to the gripper body (Fixed_Jaw), leaving 5
positioning joints per arm. v3 re-inserts the donor chain (Aloha.urdf:
link5 -> joint6[wrist roll, +-180deg about the forearm axis] -> link6) and
re-mounts Fixed_Jaw on link6 via a fixed joint, so each arm has a true
6-DOF wrist and CuRobo's full-pose grasp constraint becomes satisfiable.

Donor mesh paths (package://Aloha/meshes/*.STL) are rewritten to the local
meshes/ dir, which already contains left/right_link5|6.STL.
"""
from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DONOR = ROOT / "simulation/src/Aloha/urdf/Aloha.urdf"
BASE = ROOT / "maniskill/aloha_mini/aloha_mini_pro_v2.urdf"
OUT = ROOT / "maniskill/aloha_mini/aloha_mini_pro_v3.urdf"

# gripper mount frame on link6: identity to start; tune rpy/xyz after the
# 4-view visual check if the jaw axis is misaligned with the wrist roll axis
MOUNT_XYZ = "0 0 0"
MOUNT_RPY = "0 0 0"


def find(tree: ET.Element, tag: str, name: str) -> ET.Element:
    for el in tree.iter(tag):
        if el.get("name") == name:
            return el
    raise KeyError(f"{tag} {name!r} not found")


def rewrite_meshes(el: ET.Element) -> None:
    for mesh in el.iter("mesh"):
        fn = mesh.get("filename", "")
        mesh.set("filename", re.sub(r"^package://Aloha/meshes/", "meshes/", fn))


def main() -> None:
    donor = ET.parse(DONOR).getroot()
    robot = ET.parse(BASE).getroot()

    for side in ("left", "right"):
        jaw = f"{side}_Fixed_Jaw"
        # 1) joint5: retarget child Fixed_Jaw -> link5, restore donor pitch limit
        # joint5 is the WRIST ROLL (axis Y along the forearm — see the v2 agent
        # docstring); keep v2's intentionally widened +-180deg limit, do NOT
        # restore the donor's +-90. joint6 (donor's old jaw hinge, axis X) now
        # serves as the wrist PITCH — roll+pitch completes a 6-DOF-capable chain.
        j5 = find(robot, "joint", f"{side}_joint5")
        j5.find("child").set("link", f"{side}_link5")

        # 2) bring in donor link5, link6 (meshes re-pathed) and joint6
        link5 = copy.deepcopy(find(donor, "link", f"{side}_link5"))
        link6 = copy.deepcopy(find(donor, "link", f"{side}_link6"))
        rewrite_meshes(link5)
        rewrite_meshes(link6)
        j6 = copy.deepcopy(find(donor, "joint", f"{side}_joint6"))
        # match pro_v2 actuation conventions
        j6.find("limit").set("effort", "50")
        j6.find("limit").set("velocity", "3.14159")

        # 2b) strip link6's meshes: the donor link6 STL carries the ORIGINAL
        # jaw geometry, which the parallel gripper replaces — leaving it in
        # shows a vestigial second gripper AND its collision prong would
        # interfere with grasping. link6 stays as a bare frame link (inertial
        # only); the visible wrist is link5 + the gripper body itself.
        # (user judgment on the side-view A/B: v2's compact wrist is the correct
        # look — the donor link5 bracket reads as loose extra hardware once
        # link6's body is gone). Keep BOTH inserted links as bare frame links:
        # joint6 kinematics without any donor wrist geometry.
        for lk in (link5, link6):
            for tag in ("visual", "collision"):
                for el in list(lk.findall(tag)):
                    lk.remove(el)
        # bridge the now-bare joint5->joint6 span with a plain wrist block so
        # the gripper doesn't float in mid-air (donor O6 = link6 origin in the
        # link5 frame); visual only — no collision, no old-jaw geometry
        o6 = [float(v) for v in find(donor, "joint", f"{side}_joint6").find("origin").get("xyz").split()]
        bridge = ET.SubElement(link5, "visual")
        ET.SubElement(bridge, "origin", {"xyz": f"{o6[0]/2:.4f} {o6[1]/2:.4f} {o6[2]/2:.4f}", "rpy": "0 0 0"})
        bgeom = ET.SubElement(bridge, "geometry")
        ET.SubElement(bgeom, "box", {"size": "0.042 0.038 0.038"})

        # 3) fixed mount: link6 -> Fixed_Jaw (gripper body + fingers unchanged)
        mount = ET.SubElement(robot, "joint", {"name": f"{side}_gripper_mount", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": MOUNT_XYZ, "rpy": MOUNT_RPY})
        ET.SubElement(mount, "parent", {"link": f"{side}_link6"})
        ET.SubElement(mount, "child", {"link": jaw})

        # 4) insert new elements right after joint5 for readable ordering
        children = list(robot)
        idx = children.index(j5)
        for offset, el in enumerate((link5, j6, link6), start=1):
            robot.insert(idx + offset, el)

    robot.set("name", "aloha_mini_pro_v3")
    ET.indent(ET.ElementTree(robot), space="  ")
    OUT.write_bytes(ET.tostring(robot, encoding="utf-8", xml_declaration=True))
    print(f"wrote {OUT}")

    # quick structural audit
    out = ET.parse(OUT).getroot()
    for side in ("left", "right"):
        chain = []
        for jn in (1, 2, 3, 4, 5, 6):
            j = find(out, "joint", f"{side}_joint{jn}")
            chain.append(f"{jn}:{j.find('parent').get('link')}->{j.find('child').get('link')}")
        m = find(out, "joint", f"{side}_gripper_mount")
        chain.append(f"mount:{m.find('parent').get('link')}->{m.find('child').get('link')}")
        print(side, " ".join(chain))


if __name__ == "__main__":
    main()
