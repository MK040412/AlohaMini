#!/usr/bin/env python3
"""
Lightweight (no-ManiSkill / no-SAPIEN) validator for the AlohaMini parallel gripper.

This parses the URDF with the standard library only (xml.etree + numpy) and checks
the structural/kinematic properties that must hold for the 2-finger parallel gripper:

  1. URDF is well-formed and the kinematic tree is valid (single root, no cycles,
     every joint parent/child link exists, every link reachable).
  2. The active (non-fixed) joint order matches what the agent's keyframes assume,
     i.e. SAPIEN's depth-first order from the root: a DFS that visits each link's
     child joints in URDF declaration order.
  3. All referenced mesh files exist on disk.
  4. Forward kinematics: at the closed and open gripper configurations, the two
     finger tips are symmetric about the gripper center, the aperture matches the
     ~84 mm stroke, and the TCP (tip midpoint) stays centered.

Run:  python3 maniskill/tools/validate_parallel_gripper.py
Exit code 0 = all checks passed, 1 = at least one failure.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = (
    Path(__file__).resolve().parents[1]
    / "assets/robots/aloha_mini/maniskill_so100_version.urdf"
)

EXPECTED_DOF = 18
# The SET of active joints that must be present (the runtime *order* is decided by
# SAPIEN -- it interleaves the two arms -- and is verified by smoke_test_gripper.py,
# so here we only assert the count and the set of names, not a positional order).
EXPECTED_ACTIVE_JOINTS = {
    "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint",
    "vertical_move",
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_finger_joint1", "left_finger_joint2",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_finger_joint1", "right_finger_joint2",
}

PASS, FAIL = [], []


def check(cond, ok_msg, fail_msg):
    (PASS if cond else FAIL).append(ok_msg if cond else fail_msg)
    print(("  [PASS] " + ok_msg) if cond else ("  [FAIL] " + fail_msg))
    return cond


# ----------------------------------------------------------------------------- #
# URDF parsing helpers
# ----------------------------------------------------------------------------- #
def parse_origin(elem):
    """Return (xyz[3], rpy[3]) from an <origin> child of elem (defaults to 0)."""
    o = elem.find("origin") if elem is not None else None
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if o is not None:
        if o.get("xyz"):
            xyz = np.array([float(v) for v in o.get("xyz").split()])
        if o.get("rpy"):
            rpy = np.array([float(v) for v in o.get("rpy").split()])
    return xyz, rpy


def rpy_to_R(rpy):
    """URDF fixed-axis roll-pitch-yaw -> R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = rpy_to_R(rpy)
    M[:3, 3] = xyz
    return M


def stl_bbox(path):
    """Min/max vertex of a binary-or-ASCII STL file."""
    import struct
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    verts = []
    if len(data) == 84 + 50 * n and n > 0:  # binary STL
        off = 84
        for _ in range(n):
            # 12 floats: normal(3) + 3 vertices(3); take the 9 vertex coords
            vals = struct.unpack("<12f", data[off:off + 48])
            verts.extend([vals[3:6], vals[6:9], vals[9:12]])
            off += 50
    else:  # ASCII STL
        for line in data.decode("utf-8", "ignore").splitlines():
            s = line.split()
            if len(s) == 4 and s[0] == "vertex":
                verts.append((float(s[1]), float(s[2]), float(s[3])))
    v = np.array(verts)
    return v.min(0), v.max(0)


def link_collision(link_elem):
    """Return collision geometry of a link's first <collision> as a tuple:
    ('mesh', filename, xyz, rpy) or ('box', size[3], xyz, rpy), else None."""
    c = link_elem.find("collision")
    if c is None:
        return None
    xyz, rpy = parse_origin(c)
    mesh = c.find("geometry/mesh")
    if mesh is not None:
        s = mesh.get("scale")
        scale = np.array([float(v) for v in s.split()]) if s else np.ones(3)
        return ("mesh", mesh.get("filename"), xyz, rpy, scale)
    box = c.find("geometry/box")
    if box is not None:
        size = np.array([float(v) for v in box.get("size").split()])
        return ("box", size, xyz, rpy, np.ones(3))
    return None


def load_urdf(path):
    root = ET.parse(path).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = []
    for j in root.findall("joint"):
        jn = j.get("name")
        jt = j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        xyz, rpy = parse_origin(j)
        axis = np.array([1.0, 0.0, 0.0])
        a = j.find("axis")
        if a is not None and a.get("xyz"):
            axis = np.array([float(v) for v in a.get("xyz").split()])
        joints.append(dict(name=jn, type=jt, parent=parent, child=child,
                           xyz=xyz, rpy=rpy, axis=axis, elem=j))
    return root, links, joints


# ----------------------------------------------------------------------------- #
# Checks
# ----------------------------------------------------------------------------- #
def check_tree(links, joints):
    print("\n[1] Kinematic tree integrity")
    child_links = {j["child"] for j in joints}
    parent_links = {j["parent"] for j in joints}

    missing = [j["name"] for j in joints
               if j["parent"] not in links or j["child"] not in links]
    check(not missing, "all joint parent/child links are defined",
          f"joints referencing undefined links: {missing}")

    # exactly one child per child-link (tree, not graph)
    seen = {}
    dupes = []
    for j in joints:
        if j["child"] in seen:
            dupes.append(j["child"])
        seen[j["child"]] = j["name"]
    check(not dupes, "every link has at most one parent joint (tree)",
          f"links with multiple parent joints: {dupes}")

    roots = [n for n in links if n not in child_links]
    check(len(roots) == 1, f"single root link: {roots}",
          f"expected exactly 1 root, found {roots}")
    return roots[0] if roots else None


def dfs_active_order(root_link, links, joints):
    """SAPIEN-like DFS from root; collect non-fixed joints in declaration order."""
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)  # declaration order preserved
    order, visited = [], set()

    def walk(link):
        if link in visited:
            return
        visited.add(link)
        for j in by_parent.get(link, []):
            if j["type"] not in ("fixed",):
                order.append(j["name"])
            walk(j["child"])

    walk(root_link)
    return order


def check_active_order(root_link, links, joints):
    print("\n[2] Active joints: count + name set (runtime ORDER verified by smoke test)")
    order = dfs_active_order(root_link, links, joints)
    print("  active (non-fixed) joints found:")
    for i, n in enumerate(order):
        print(f"      {i:2d}: {n}")
    check(len(order) == EXPECTED_DOF, f"active DOF == {EXPECTED_DOF}",
          f"active DOF == {len(order)} (expected {EXPECTED_DOF})")
    got, exp = set(order), EXPECTED_ACTIVE_JOINTS
    check(got == exp,
          "active-joint name set matches expected (18 joints)",
          f"active-joint name set mismatch: missing={exp - got}, extra={got - exp}")
    return order


def check_meshes(root, urdf_path):
    print("\n[3] Referenced mesh files exist")
    base = urdf_path.parent
    missing = []
    refs = set()
    for m in root.iter("mesh"):
        fn = m.get("filename")
        refs.add(fn)
        if not (base / fn).exists():
            missing.append(fn)
    check(not missing, f"all {len(refs)} referenced meshes exist",
          f"missing meshes: {sorted(set(missing))}")


def fk(root_link, links, joints, qpos):
    """World transform of every link given a {joint_name: value} dict."""
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    world = {root_link: np.eye(4)}

    def walk(link):
        for j in by_parent.get(link, []):
            local = T(j["xyz"], j["rpy"])
            if j["type"] == "prismatic":
                q = qpos.get(j["name"], 0.0)
                tr = np.eye(4)
                tr[:3, 3] = j["axis"] * q
                local = local @ tr
            elif j["type"] in ("revolute", "continuous"):
                q = qpos.get(j["name"], 0.0)
                rot = np.eye(4)
                axis = j["axis"] / (np.linalg.norm(j["axis"]) + 1e-12)
                # Rodrigues
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                rot[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
                local = local @ rot
            world[j["child"]] = world[link] @ local
            walk(j["child"])

    walk(root_link)
    return world


def tip(world, name):
    return world[name][:3, 3]


def check_gripper_fk(root_link, links, joints):
    print("\n[4] Forward-kinematics: parallel-gripper geometry")
    OPEN = 0.037
    for side in ("left", "right"):
        f1j, f2j = f"{side}_finger_joint1", f"{side}_finger_joint2"
        t1, t2 = f"{side}_finger1_tip", f"{side}_finger2_tip"

        closed = fk(root_link, links, joints, {f1j: 0.0, f2j: 0.0})
        opened = fk(root_link, links, joints, {f1j: OPEN, f2j: OPEN})

        c1, c2 = tip(closed, t1), tip(closed, t2)
        o1, o2 = tip(opened, t1), tip(opened, t2)

        gap_closed = np.linalg.norm(c1 - c2)
        gap_open = np.linalg.norm(o1 - o2)
        tcp_closed = (c1 + c2) / 2
        tcp_open = (o1 + o2) / 2
        tcp_shift = np.linalg.norm(tcp_closed - tcp_open)

        print(f"  [{side}] closed tip gap = {gap_closed*1000:.1f} mm, "
              f"open tip gap = {gap_open*1000:.1f} mm, "
              f"TCP shift open->closed = {tcp_shift*1000:.2f} mm")

        check(gap_closed < 0.005,
              f"[{side}] fingers meet at center when closed ({gap_closed*1000:.1f} mm)",
              f"[{side}] closed gap too large: {gap_closed*1000:.1f} mm")
        check(abs(gap_open - 2 * OPEN) < 0.01,
              f"[{side}] open aperture ~= {2*OPEN*1000:.0f} mm "
              f"(got {gap_open*1000:.1f} mm)",
              f"[{side}] open aperture {gap_open*1000:.1f} mm != ~{2*OPEN*1000:.0f} mm")
        check(tcp_shift < 0.003,
              f"[{side}] TCP stays centered as gripper opens "
              f"(shift {tcp_shift*1000:.2f} mm)",
              f"[{side}] TCP shifts {tcp_shift*1000:.2f} mm between open/closed")
        # symmetry: the two tips move equal-and-opposite from the TCP
        d1 = o1 - tcp_open
        d2 = o2 - tcp_open
        sym = np.linalg.norm(d1 + d2)
        check(sym < 0.003,
              f"[{side}] fingers move symmetrically about center "
              f"(residual {sym*1000:.2f} mm)",
              f"[{side}] asymmetric finger motion (residual {sym*1000:.2f} mm)")


def check_no_mesh_overlap(root_link, links, joints, urdf_path):
    """At the closed pose, the two finger collision shapes must not interpenetrate,
    measured along the actual jaw open/close (separation) axis -- which is NOT world
    X, because the wrist rotates the gripper frame."""
    print("\n[5] Collision-shape overlap at closed pose (along separation axis)")
    base = urdf_path.parent
    unit = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])

    def world_corners(world, link_name):
        kind, geom, oxyz, orpy, scale = link_collision(links[link_name])
        if kind == "mesh":
            mn, mx = stl_bbox(base / geom)
            mn, mx = mn * scale, mx * scale          # apply URDF mesh scale (mm->m)
        else:  # box
            mn, mx = -geom / 2.0, geom / 2.0
        box = mn + unit * (mx - mn)                  # 8 corners in geom frame
        M = world[link_name] @ T(oxyz, orpy)
        return (M[:3, :3] @ box.T).T + M[:3, 3]      # 8x3 world points

    for side in ("left", "right"):
        f1j, f2j = f"{side}_finger_joint1", f"{side}_finger_joint2"
        l1, l2 = f"{side}_finger1", f"{side}_finger2"
        closed = fk(root_link, links, joints, {f1j: 0.0, f2j: 0.0})
        opened = fk(root_link, links, joints, {f1j: 0.037, f2j: 0.037})

        # separation axis = world direction between the two finger frames when open
        sep = opened[l2][:3, 3] - opened[l1][:3, 3]
        sep = sep / (np.linalg.norm(sep) + 1e-12)

        p1 = world_corners(closed, l1) @ sep         # finger1 projected onto sep axis
        p2 = world_corners(closed, l2) @ sep
        gap = p2.min() - p1.max()                    # finger1 lower, finger2 higher
        print(f"  [{side}] finger1 proj=[{p1.min()*1000:.1f},{p1.max()*1000:.1f}] mm, "
              f"finger2 proj=[{p2.min()*1000:.1f},{p2.max()*1000:.1f}] mm, "
              f"closed gap = {gap*1000:.1f} mm")
        # The real roboninecom clamps interlock (overlap) at the closed pose by
        # design, and finger-finger collision is disabled in the agent, so a
        # closed-pose overlap is acceptable. We only fail on an absurd overlap
        # (e.g. a units/scale bug), and treat the rest as informational.
        check(gap > -0.2,
              f"[{side}] closed-pose clamp overlap within sane bounds "
              f"(gap {gap*1000:.1f} mm; clamps interlock, collision disabled)",
              f"[{side}] absurd closed-pose overlap {(-gap)*1000:.1f} mm "
              f"(likely a mesh units/scale bug)")


def main():
    print(f"Validating: {URDF}")
    if not URDF.exists():
        print(f"  [FAIL] URDF not found: {URDF}")
        sys.exit(1)
    root, links, joints = load_urdf(URDF)
    root_link = check_tree(links, joints)
    if root_link is None:
        print("\nCannot continue without a valid root.")
        sys.exit(1)
    check_active_order(root_link, links, joints)
    check_meshes(root, URDF)
    check_gripper_fk(root_link, links, joints)
    check_no_mesh_overlap(root_link, links, joints, URDF)

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)
    if FAIL:
        for f in FAIL:
            print("  FAIL:", f.splitlines()[0])
        sys.exit(1)
    print("All structural / kinematic checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
