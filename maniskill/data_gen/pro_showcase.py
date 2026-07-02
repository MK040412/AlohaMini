"""6-DOF Pro cube-grasp SHOWCASE — Std-showcase-grade.

HQ RT (1920x1280, spp64, crf15) x multiple camera views (side/head/top) x multiple
approach pitches. Recipe = the validated pro_grasp_test one: desc-first full-pose IK,
Cartesian-line tilted descent, slow speed-capped motion, gradual close, slow lift.

Usage:
  python pro_showcase.py sweep                 # fast no-render pitch sweep
  python pro_showcase.py clip <pitch> <view>   # one HQ clip
"""
import sys, os, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")
sys.path.insert(0, "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad")
import mani_skill.envs, data_gen
from grasp_demo_v2 import (SlowGrasp, _best_full_pose, interp, to_uint8,
                           V_ARM, V_ARM_DESCEND, V_LIFT, CLOSE_STEPS, SETTLE, HOLD,
                           FRAMES_TMP, FPS)
from data_gen.intern_engine.skills.ik import actor_position, resolve_actor, desired_approach_dir

# FAST render settings (user: quality doesn't need to be extreme — prioritize speed):
# rt-fast shader, low spp, 720p, fast ffmpeg preset.
import sapien.render as _R
_R.set_ray_tracing_samples_per_pixel(8)
_R.set_ray_tracing_path_depth(4)
W, H = 1280, 720


def encode(frames, path):
    import shutil, subprocess
    import imageio.v2 as imageio
    if os.path.isdir(FRAMES_TMP):
        shutil.rmtree(FRAMES_TMP)
    os.makedirs(FRAMES_TMP)
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(FRAMES_TMP, f"{i:05d}.png"), f)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                    os.path.join(FRAMES_TMP, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-preset", "veryfast", path],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(FRAMES_TMP, ignore_errors=True)

OBJ = "077_rubiks_cube"
BASE = (-0.29, -0.25)
OUT = "/home/perelman/AlohaMini/maniskill/data_gen/output/demo_pro"
os.makedirs(OUT, exist_ok=True)

# camera views (eye, target) — side/head mirror the Std showcase; top added per user ask
VIEWS = {
    "side": ([0.48, -0.88, 1.05], [-0.15, -0.35, 0.85]),
    "head": ([-0.35, 0.0, 1.45], [-0.13, -0.45, 0.80]),
    "top":  ([-0.13, -0.42, 1.75], [-0.13, -0.45, 0.75]),
}


def build_acts(env, pitch_deg):
    """Validated Pro grasp: desc-first branch, tilted Cartesian-line descent."""
    be = env.unwrapped
    name = be.object_actor_names[0]
    obj0 = actor_position(resolve_actor(env, name)).copy()
    g = SlowGrasp(env)
    skill, lay = g.skill, g.lay
    APPROACH_H, LIFT_H = 0.11, 0.16
    appr_dir = desired_approach_dir(obj0, pitch_deg).astype(np.float32)
    jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
    grasp_pt = obj0.astype(np.float32)
    pre_pt = (grasp_pt - appr_dir * APPROACH_H).astype(np.float32)
    desc = _best_full_pose(env, grasp_pt, appr_dir, jaw_dir, "left", 0.0)
    appr = _best_full_pose(env, pre_pt, appr_dir, jaw_dir, "left", 0.0, seed=desc.arm_qpos)
    # Cartesian-line descent waypoints along the tilted axis
    descent_qs = [appr.arm_qpos]; seedq = appr.arm_qpos
    for s in np.linspace(APPROACH_H, 0.0, 9)[1:]:
        w = _best_full_pose(env, (grasp_pt - appr_dir * s).astype(np.float32),
                            appr_dir, jaw_dir, "left", 0.0, seed=seedq)
        descent_qs.append(w.arm_qpos); seedq = w.arm_qpos
    desc_q = descent_qs[-1]
    op, cl = skill.open_gripper, skill.closed_gripper
    q_now = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
    acts = []
    for q in interp(q_now, appr.arm_qpos, V_ARM):
        acts.append(g._action(q, op, 0.0))
    for q0, q1 in zip(descent_qs[:-1], descent_qs[1:]):
        for q in interp(q0, q1, V_ARM_DESCEND):
            acts.append(g._action(q, op, 0.0))
    for k in range(1, CLOSE_STEPS + 1):
        acts.append(g._action(desc_q, op + (cl - op) * k / CLOSE_STEPS, 0.0))
    for _ in range(SETTLE):
        acts.append(g._action(desc_q, cl, 0.0))
    for lz in interp([0.0], [LIFT_H], V_LIFT):
        acts.append(g._action(desc_q, cl, float(lz[0])))
    for _ in range(HOLD):
        acts.append(g._action(desc_q, cl, LIFT_H))
    return acts, desc.error, appr.error


def make_env(render_hq, view):
    kw = dict(num_envs=1, obs_mode="state", control_mode="pd_joint_pos_fixed_base",
              render_mode="rgb_array", reward_mode="none", sim_backend="physx_cpu",
              object_ids=[OBJ], robot_uid="aloha_mini_pro_v2", base_xy=BASE)
    if render_hq:
        eye, target = VIEWS[view]
        kw.update(render_eye=eye, render_target=target,
                  human_render_camera_configs=dict(shader_pack="rt-fast", width=W, height=H))
    return gym.make("AlohaMiniMultiYCB-v1", **kw)


def sweep(pitches):
    for p in pitches:
        env = make_env(False, "side"); be = env.unwrapped
        env.reset(seed=0)
        try:
            acts, de, ae = build_acts(env, p)
            for a in acts:
                env.step(torch.as_tensor(a)[None])
            succ = bool(be.evaluate()["success"][0])
            obj = actor_position(resolve_actor(env, be.object_actor_names[0]))
            print(f"PITCH {p:5.1f}  success={succ}  obj_z={obj[2]:.3f}  ik_err d={de:.4f} a={ae:.4f}", flush=True)
        except Exception as e:
            print(f"PITCH {p:5.1f}  ERROR {e}", flush=True)
        env.close()


def clip(pitch, view):
    env = make_env(True, view); be = env.unwrapped
    env.reset(seed=0)
    acts, _, _ = build_acts(env, pitch)
    frames = [to_uint8(env.render())]
    for a in acts:
        env.step(torch.as_tensor(a)[None])
        frames.append(to_uint8(env.render()))
    succ = bool(be.evaluate()["success"][0])
    env.close()
    path = os.path.join(OUT, f"pro_{OBJ}_pitch{int(pitch)}_{view}.mp4")
    encode(frames, path)
    print(f"CLIP pitch={pitch} view={view} success={succ} frames={len(frames)} "
          f"-> {path} ({os.path.getsize(path)//1024}KB)", flush=True)
    return succ


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sweep":
        sweep([45.0, 55.0, 60.0, 65.0, 70.0])
    elif mode == "clip":
        clip(float(sys.argv[2]), sys.argv[3])
