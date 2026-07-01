"""Slow, realistic-speed, domain-randomized grasp demo (HQ full-RT side view).

Improvements over hq_render.py:
  * SLOW motion: every phase is interpolated in joint space with a per-step velocity
    cap (~0.02 rad/step = ~0.4 rad/s at 20 Hz control), so the arm moves at a speed a
    real SO100 servo could follow -- no instantaneous snap-to-target.
  * Gradual gripper close + slow vertical lift + settle frames.
  * Domain randomization: object XY (and yaw) randomized per seed via object_xy_noise;
    IK targets the ACTUAL randomized object position, proving robustness.
  * (approach angles handled by grasp_demo_angles.py once pose-IK lands.)
"""
import sys, os, json, subprocess, shutil, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")
import sapien.render as R
import mani_skill.envs, data_gen
from data_gen.intern_engine.skills import build_skill
from data_gen.intern_engine.skills.ik import (
    solve_arm_ik_position, solve_arm_ik_pose, solve_arm_ik_full_pose,
    apply_horizontal_jaw_wrist_roll, desired_approach_dir, actor_position, resolve_actor,
)


def _best_position(env, target, arm, lift, seed=None):
    """Position-only IK with a shoulder-lift seed sweep -> lowest position error.
    The 6-DOF Pro IK is seed-sensitive (bad seeds land in 200 mm local minima); the
    sweep finds a reaching config, and its jaws come out ~level already."""
    best = None
    if seed is not None:
        return solve_arm_ik_position(env, target, arm=arm, seed=seed, lift_position=lift)
    for s in (0.5, 1.0, 1.5, -0.5, -1.0, 0.0):
        r = solve_arm_ik_position(env, target, arm=arm, lift_position=lift, shoulder_lift_seed=s)
        if best is None or r.error < best.error:
            best = r
    return best


def _best_full_pose(env, target, approach_dir, jaw_dir, arm, lift, seed=None):
    """Full-pose IK with a shoulder-lift seed sweep -> lowest position error."""
    best = None
    seeds = [seed] if seed is not None else [0.5, 1.0, 1.5, -0.5, 0.0]
    for s in seeds:
        kw = dict(arm=arm, lift_position=lift, max_iters=250)
        if seed is not None:
            kw["seed"] = seed
        else:
            kw["shoulder_lift_seed"] = s
        r = solve_arm_ik_full_pose(env, target, approach_dir, jaw_dir, **kw)
        if best is None or r.error < best.error:
            best = r
    return best

W, H, SPP, PATH_DEPTH, FPS = 1920, 1280, 64, 12, 30
# velocity caps (per 0.05 s control step)
V_ARM = 0.020      # rad/step  (~0.40 rad/s)
V_ARM_DESCEND = 0.013
V_LIFT = 0.0045    # m/step    (~0.09 m/s)
CLOSE_STEPS = 32   # gradual gripper close
SETTLE = 10        # contact settle frames after close
HOLD = 22
OUT = "/home/perelman/AlohaMini/maniskill/data_gen/output/demo_slow"
FRAMES_TMP = ("/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/"
              "scratchpad/frames_slow_" + str(os.getpid()))  # unique per process (parallel-safe)

R.set_ray_tracing_samples_per_pixel(SPP)
R.set_ray_tracing_path_depth(PATH_DEPTH)
try:
    R.set_ray_tracing_denoiser("oidn")
except Exception as e:
    print("denoiser warn:", e)
os.makedirs(OUT, exist_ok=True)


def to_uint8(frame):
    a = frame[0]
    a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    return a.astype(np.uint8) if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)


def encode(frames, path):
    if os.path.isdir(FRAMES_TMP):
        shutil.rmtree(FRAMES_TMP)
    os.makedirs(FRAMES_TMP)
    import imageio.v2 as imageio
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(FRAMES_TMP, f"{i:05d}.png"), f)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(FRAMES_TMP, "%05d.png"),
                    "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(FRAMES_TMP)


def interp(q0, q1, vmax):
    q0 = np.asarray(q0, dtype=np.float32); q1 = np.asarray(q1, dtype=np.float32)
    n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / max(vmax, 1e-6))))
    return [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]


class SlowGrasp:
    """Build a slow, speed-capped pick trajectory using position IK (top-down)."""

    def __init__(self, env):
        self.env = env
        self.skill = build_skill("pick")
        self.arm = "left"
        self.lay = self.skill.arm_layout(env.unwrapped.agent)  # 5-DOF Std / 6-DOF Pro

    def _action(self, arm_qpos, gripper, lift):
        a = self.skill.current_action_template(self.env)
        a[:3] = 0.0
        a[3] = float(lift)
        a[self.lay["right_grip"]] = self.skill.open_gripper  # idle (right) gripper open
        self.skill.set_arm_action(a, self.arm, arm_qpos, gripper)
        return a.astype(np.float32)

    def build(self, object_name, descend_offset=-0.025, approach_h=0.11,
              lift_h=0.16, lift_start=0.0):
        env = self.env
        tgt = actor_position(resolve_actor(env, object_name))
        op = self.skill.open_gripper; cl = self.skill.closed_gripper
        # current arm config (rest keyframe)
        q_now = self.skill.current_action_template(env)[self.lay["left_arm"]].astype(np.float32)
        if self.lay["n"] >= 6:
            # 6-DOF Pro: position-only IK with a seed sweep (the arm's grasp config has
            # near-level jaws already, and joint6 is not a clean tool-roll so leveling it
            # separately would drag the TCP off target). desc seeds from appr for a smooth
            # single-branch descent.
            appr = _best_position(env, tgt + np.array([0, 0, approach_h], np.float32),
                                  self.arm, lift_start)
            desc = _best_position(env, tgt + np.array([0, 0, descend_offset], np.float32),
                                  self.arm, lift_start, seed=appr.arm_qpos)
        else:
            appr = apply_horizontal_jaw_wrist_roll(
                env, solve_arm_ik_position(env, tgt + np.array([0, 0, approach_h], np.float32),
                                           arm=self.arm, lift_position=lift_start))
            desc = apply_horizontal_jaw_wrist_roll(
                env, solve_arm_ik_position(env, tgt + np.array([0, 0, descend_offset], np.float32),
                                           arm=self.arm, seed=appr.arm_qpos, lift_position=lift_start))
        acts = []
        # 1) slow approach above object
        for q in interp(q_now, appr.arm_qpos, V_ARM):
            acts.append(self._action(q, op, lift_start))
        # 2) slow descend onto object
        for q in interp(appr.arm_qpos, desc.arm_qpos, V_ARM_DESCEND):
            acts.append(self._action(q, op, lift_start))
        # 3) gradual close
        for k in range(1, CLOSE_STEPS + 1):
            g = op + (cl - op) * (k / CLOSE_STEPS)
            acts.append(self._action(desc.arm_qpos, g, lift_start))
        # 4) settle
        for _ in range(SETTLE):
            acts.append(self._action(desc.arm_qpos, cl, lift_start))
        # 5) slow lift
        for lz in interp([lift_start], [lift_h], V_LIFT):
            acts.append(self._action(desc.arm_qpos, cl, float(lz[0])))
        # 6) hold
        for _ in range(HOLD):
            acts.append(self._action(desc.arm_qpos, cl, lift_h))
        return acts

    def build_angled(self, object_name, pitch_deg, descend_offset=-0.02, back=0.11,
                     lift_h=0.16, lift_start=0.0):
        """Approach an object along a tilted axis (pitch_deg) and grasp, slow motion."""
        env = self.env
        tgt = actor_position(resolve_actor(env, object_name))
        op = self.skill.open_gripper; cl = self.skill.closed_gripper
        grasp_pt = (tgt + np.array([0, 0, descend_offset], np.float32)).astype(np.float64)
        d = desired_approach_dir(grasp_pt, pitch_deg)           # down-and-forward unit
        pre = (grasp_pt - back * d).astype(np.float32)          # back off along approach
        q_now = self.skill.current_action_template(env)[self.lay["left_arm"]].astype(np.float32)
        appr = apply_horizontal_jaw_wrist_roll(
            env, solve_arm_ik_pose(env, pre, pitch_deg, arm=self.arm, lift_position=lift_start))
        desc = apply_horizontal_jaw_wrist_roll(
            env, solve_arm_ik_pose(env, grasp_pt.astype(np.float32), pitch_deg, arm=self.arm,
                                   seed=appr.arm_qpos, lift_position=lift_start))
        acts = []
        for q in interp(q_now, appr.arm_qpos, V_ARM):
            acts.append(self._action(q, op, lift_start))
        for q in interp(appr.arm_qpos, desc.arm_qpos, V_ARM_DESCEND):
            acts.append(self._action(q, op, lift_start))
        for k in range(1, CLOSE_STEPS + 1):
            acts.append(self._action(desc.arm_qpos, op + (cl - op) * (k / CLOSE_STEPS), lift_start))
        for _ in range(SETTLE):
            acts.append(self._action(desc.arm_qpos, cl, lift_start))
        for lz in interp([lift_start], [lift_h], V_LIFT):
            acts.append(self._action(desc.arm_qpos, cl, float(lz[0])))
        for _ in range(HOLD):
            acts.append(self._action(desc.arm_qpos, cl, lift_h))
        return acts


VIEWS = {
    "side": ([0.48, -0.88, 1.05], [-0.13, -0.45, 0.75]),
    "head": ([-0.35, 0.0, 1.45], [-0.13, -0.45, 0.70]),   # replicates cam_main mast view
}


def run(object_id, seed, noise, tag, view, distractors=None):
    eye, target = VIEWS[view]
    ids = [object_id] + list(distractors or [])
    env = gym.make("AlohaMiniMultiYCB-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", object_ids=ids,
                   object_xy_noise=float(noise), render_eye=eye, render_target=target,
                   human_render_camera_configs=dict(shader_pack="rt", width=W, height=H))
    be = env.unwrapped
    env.reset(seed=seed)
    g = SlowGrasp(env)
    acts = g.build(be.object_actor_names[0])
    frames = [to_uint8(env.render())]
    for a in acts:
        env.step(torch.as_tensor(a)[None])
        frames.append(to_uint8(env.render()))
    succ = bool(be.evaluate()["success"][0])
    objp = actor_position(resolve_actor(env, be.object_actor_names[0]))
    env.close()
    path = os.path.join(OUT, f"{object_id}_{tag}_{view}.mp4")
    encode(frames, path)
    print(f"[{object_id:22s} {tag:9s} {view:4s}] success={succ} frames={len(frames)} "
          f"objXY=({objp[0]:+.3f},{objp[1]:+.3f}) noise={noise} -> {os.path.getsize(path)//1024}KB",
          flush=True)
    return path, succ


def run_angled(object_id, pitch, view, seed=0):
    eye, target = VIEWS[view]
    env = gym.make("AlohaMiniMultiYCB-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", object_ids=[object_id],
                   render_eye=eye, render_target=target,
                   human_render_camera_configs=dict(shader_pack="rt", width=W, height=H))
    be = env.unwrapped
    env.reset(seed=seed)
    g = SlowGrasp(env)
    acts = g.build_angled(be.object_actor_names[0], pitch, descend_offset=-0.025, back=0.10)
    frames = [to_uint8(env.render())]
    for a in acts:
        env.step(torch.as_tensor(a)[None])
        frames.append(to_uint8(env.render()))
    succ = bool(be.evaluate()["success"][0])
    env.close()
    path = os.path.join(OUT, f"{object_id}_pitch{pitch}_{view}.mp4")
    encode(frames, path)
    print(f"[{object_id:22s} pitch{pitch:3d} {view:4s}] success={succ} frames={len(frames)} "
          f"-> {os.path.getsize(path)//1024}KB", flush=True)
    return path, succ


def concat(paths, out_path):
    if not paths:
        return None
    listf = out_path + ".txt"
    with open(listf, "w") as fh:
        for p in paths:
            fh.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy",
                    out_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else None


def hstack(side_path, head_path, out_path):
    """Side-by-side side|head combined video (labels burned in)."""
    if not (os.path.exists(side_path) and os.path.exists(head_path)):
        return None
    subprocess.run([
        "ffmpeg", "-y", "-i", side_path, "-i", head_path, "-filter_complex",
        "[0:v]drawtext=text='SIDE':x=20:y=20:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[a];"
        "[1:v]scale=-1:1280,drawtext=text='HEAD':x=20:y=20:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[b];"
        "[a][b]hstack=inputs=2[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "16",
        "-preset", "slow", "-pix_fmt", "yuv420p", out_path],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else None


def main_angles():
    spec = json.loads(os.environ.get("ANGLE_JSON", "{}"))
    if not spec:
        spec = {"objects": {"062_dice": [60, 45, 30], "058_golf_ball": [60, 45, 30],
                            "012_strawberry": [60, 45, 30], "073-f_lego_duplo": [60, 45, 30]},
                "views": ["side"], "head_object": "062_dice"}
    made = []
    for oid, pitches in spec["objects"].items():
        perobj = []
        for v in spec.get("views", ["side"]):
            for p in pitches:
                path, s = run_angled(oid, p, v)
                if s:
                    made.append(path)
                    if v == "side":
                        perobj.append(path)
        # same object, 3 angles, concatenated
        if perobj:
            c = concat(perobj, os.path.join(OUT, f"{oid}_ANGLES.mp4"))
            if c:
                print("ANGLES_MONTAGE:", c, flush=True)
    # one object also from the head view across angles
    ho = spec.get("head_object")
    if ho:
        for p in spec["objects"].get(ho, []):
            run_angled(ho, p, "head")
    print("ANGLES_DONE made=", len(made))


def main():
    if os.environ.get("MODE") == "angles":
        return main_angles()
    spec = json.loads(os.environ.get("DEMO_JSON", "{}"))
    if not spec:
        spec = {"objects": ["077_rubiks_cube", "062_dice", "058_golf_ball"],
                "dr_objects": ["077_rubiks_cube", "062_dice", "058_golf_ball"],
                "dr_noise": 0.045, "dr_seeds": 3, "views": ["side", "head"]}
    views = spec.get("views", ["side", "head"])
    made = []
    # 1) slow top-down, every object, every requested view
    for oid in spec["objects"]:
        sp = {}
        for v in views:
            p, s = run(oid, 0, 0.0, "slow", v)
            if s:
                made.append(p); sp[v] = p
        if "side" in sp and "head" in sp:
            c = hstack(sp["side"], sp["head"], os.path.join(OUT, f"{oid}_slow_COMBINED.mp4"))
            if c:
                print("COMBINED:", c, flush=True)
    # 2) domain randomization, side view (position robustness)
    for oid in spec.get("dr_objects", []):
        for k in range(int(spec.get("dr_seeds", 3))):
            p, s = run(oid, 100 + k, float(spec.get("dr_noise", 0.045)), f"DR{k}", "side")
            if s:
                made.append(p)
    print("SLOW_DONE made=", len(made))


if __name__ == "__main__":
    main()
