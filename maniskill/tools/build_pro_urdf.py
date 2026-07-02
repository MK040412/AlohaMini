"""Build the AlohaMini Pro URDF: REPLACE the original gripper with the roboninecom
parallel gripper.

The source aloha_mini.urdf "6-DOF arm" is really a 5-DOF arm + jaw gripper:
  joints 1-4 = arm, joint5 = wrist ROLL (axis Y, along the forearm),
  link5 = original gripper body + serrated static jaw, link6/joint6 = the moving
  hook jaw (verified by rendering the meshes).
So the correct conversion mirrors roboninecom's own SO-101 kit (and our Std SO-100
conversion): DELETE link5+link6+joint6 (the old gripper) and let joint5 (the wrist
roll — roboninecom's `link4_to_link5`) drive the parallel-gripper palm directly.
Result: 5 arm joints + prismatic parallel gripper per arm, on the longer Pro arm.

PALM_RPY (env) composes an extra rotation into joint5's origin for visual mount
tuning (default "0 0 0"; iterate by rendering). Output: aloha_mini_pro_v2.urdf.
"""
import os, shutil, copy
import xml.etree.ElementTree as ET

ROOT = "/home/perelman/AlohaMini/maniskill"
SRC6 = os.path.join(ROOT, "aloha_mini/aloha_mini.urdf")               # 6-DOF arm
SRC5 = os.path.join(ROOT, "assets/robots/aloha_mini/maniskill_so100_version.urdf")  # parallel gripper
OUT = os.path.join(ROOT, "aloha_mini/aloha_mini_pro_v2.urdf")

PALM_RPY = os.environ.get("PALM_RPY", "0 0 0")   # extra rotation on joint5 -> palm

# gripper subtree element names (per arm) transplanted from the 5-DOF URDF
GRIP_LINKS = ["{s}_Fixed_Jaw", "{s}_finger1", "{s}_finger2", "{s}_finger1_tip", "{s}_finger2_tip"]
GRIP_JOINTS = ["{s}_finger_joint1", "{s}_finger_joint2",
               "{s}_finger1_tip_joint", "{s}_finger2_tip_joint"]


def _rescale_link_mass(link, new_mass):
    """Set a link's mass to new_mass and scale its inertia tensor by the mass ratio."""
    inertial = link.find("inertial")
    if inertial is None:
        return
    m = inertial.find("mass")
    if m is None:
        return
    old = float(m.get("value"))
    if old <= 0:
        return
    ratio = new_mass / old
    m.set("value", f"{new_mass:.6f}")
    inertia = inertial.find("inertia")
    if inertia is not None:
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            if inertia.get(k) is not None:
                inertia.set(k, f"{float(inertia.get(k)) * ratio:.8f}")


def main():
    # copy clamp meshes next to the new URDF
    src_meshes = os.path.join(ROOT, "assets/robots/aloha_mini/clamp_meshes")
    dst_meshes = os.path.join(ROOT, "aloha_mini/clamp_meshes")
    os.makedirs(dst_meshes, exist_ok=True)
    for f in os.listdir(src_meshes):
        shutil.copy2(os.path.join(src_meshes, f), os.path.join(dst_meshes, f))

    t6 = ET.parse(SRC6); r6 = t6.getroot()
    t5 = ET.parse(SRC5); r5 = t5.getroot()
    links5 = {l.get("name"): l for l in r5.findall("link")}
    joints5 = {j.get("name"): j for j in r5.findall("joint")}

    # (1) prepend the planar mobile base (root -> root_x -> root_y -> base_link),
    #     matching what base_agent.py's BASE_JOINT_NAMES expects, and freeze wheels.
    for lname in ("root", "root_x_link", "root_y_link"):
        if lname in links5:
            r6.append(copy.deepcopy(links5[lname]))
    for jname in ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"):
        if jname in joints5:
            r6.append(copy.deepcopy(joints5[jname]))   # root_z child == base_link
    for j in r6.findall("joint"):
        if j.get("name", "").startswith("wheel") and j.get("type") == "continuous":
            j.set("type", "fixed")
            for ax in j.findall("axis"):
                j.remove(ax)
    # Widen the mid-arm joint limits. The SolidWorks-derived URDF pinned joints 2-4 at
    # +-90 deg, which is unusually restrictive; use +-135 deg so the arm can grasp from
    # a comfortably bent, well-conditioned posture. joint5 (wrist roll) gets full turn.
    WIDEN = {f"{s}_joint{i}": 2.3562 for s in ("left", "right") for i in range(2, 5)}
    WIDEN.update({f"{s}_joint5": 3.1416 for s in ("left", "right")})
    for j in r6.findall("joint"):
        lim = j.find("limit")
        if j.get("name") in WIDEN and lim is not None:
            lim.set("lower", f"{-WIDEN[j.get('name')]:.4f}")
            lim.set("upper", f"{WIDEN[j.get('name')]:.4f}")

    # DELETE the original gripper: link5 (gripper body + static serrated jaw),
    # link6 (moving hook jaw), joint6 (jaw pivot), and the old fixed ee frames.
    for tag, names in (("joint", ["left_ee_joint", "right_ee_joint",
                                  "left_joint6", "right_joint6"]),
                       ("link", ["left_ee_link", "right_ee_link",
                                 "left_link5", "right_link5",
                                 "left_link6", "right_link6"])):
        for el in list(r6.findall(tag)):
            if el.get("name") in names:
                r6.remove(el)

    for s in ("left", "right"):
        # transplant gripper links + joints verbatim (mesh paths clamp_meshes/* stay valid),
        # but rescale the clamp masses: the roboninecom meshes come out at ~0.5-0.15 kg
        # (SolidWorks density artifact) -> ~0.86 kg dangling at the wrist, which gravity-
        # sags the arm. Real clamp is light plastic/aluminum, so use realistic masses.
        GRIP_MASS = {f"{s}_Fixed_Jaw": 0.08, f"{s}_finger1": 0.03, f"{s}_finger2": 0.03}
        for tmpl in GRIP_LINKS:
            nm = tmpl.format(s=s)
            if nm in links5:
                link = copy.deepcopy(links5[nm])
                if nm in GRIP_MASS:
                    _rescale_link_mass(link, GRIP_MASS[nm])
                r6.append(link)
        for tmpl in GRIP_JOINTS:
            nm = tmpl.format(s=s)
            if nm in joints5:
                r6.append(copy.deepcopy(joints5[nm]))
        # RETARGET joint5 (the wrist roll, roboninecom's link4_to_link5): its child
        # becomes the parallel-gripper palm. PALM_RPY tunes the mount orientation.
        for j in r6.findall("joint"):
            if j.get("name") == f"{s}_joint5":
                j.find("child").set("link", f"{s}_Fixed_Jaw")
                o = j.find("origin")
                if o is not None and PALM_RPY != "0 0 0":
                    o.set("rpy", PALM_RPY)

    ET.indent(t6, space="  ")
    t6.write(OUT, encoding="utf-8", xml_declaration=True)
    nrev = sum(1 for j in r6.findall("joint") if j.get("type") == "revolute")
    npris = sum(1 for j in r6.findall("joint") if j.get("type") == "prismatic")
    print(f"wrote {OUT}")
    print(f"  revolute={nrev} prismatic={npris}  palm rpy='{PALM_RPY}'")
    print(f"  links={len(r6.findall('link'))} joints={len(r6.findall('joint'))}")


if __name__ == "__main__":
    main()
