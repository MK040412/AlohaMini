"""High-quality side-view RT renderer for AlohaMini parallel-gripper grasp demos.

Captures every env.render() frame at high resolution + full ray tracing, then
encodes with ffmpeg at a high bitrate (CRF 15, slow preset) -- far better quality
than RecordEpisode's default imageio encode. Side view shows the whole grasp.
"""
import sys, os, json, subprocess, shutil, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")
import sapien.render as R
import mani_skill.envs, data_gen
from data_gen.intern_engine.skills import build_skill

# ---- quality knobs ----------------------------------------------------------
W, H, SPP, PATH_DEPTH = 1920, 1280, 64, 12
FPS = 30
OUT = "/home/perelman/AlohaMini/maniskill/data_gen/output/demo_hq"
FRAMES_TMP = "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad/hqframes"

# graspable objects (filled from slot0 sweep): {object_id: descend_offset}.
# Each target is placed at slot 0 (the tuned grasp sweet spot) with distractors behind.
GRASP = json.loads(os.environ.get("GRASP_JSON", "{}"))
if not GRASP:
    GRASP = {"077_rubiks_cube": -0.025}
DISTRACTORS = ["058_golf_ball", "057_racquetball", "063-b_marbles",
               "012_strawberry", "073-f_lego_duplo"]

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
    if a.dtype != np.uint8:
        a = np.clip(a * (255 if a.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return a


def encode(frames, path):
    d = FRAMES_TMP
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    import imageio.v2 as imageio
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(d, f"{i:05d}.png"), f)
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(d, "%05d.png"),
           "-c:v", "libx264", "-crf", "15", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(d)


def render_one(object_id, offset, with_distractors):
    # target at index 0 -> slot 0 (tuned sweet spot, yaw 0 at episode_index 0)
    ids = [object_id]
    if with_distractors:
        ids += [d for d in DISTRACTORS if d != object_id][:4]
    env = gym.make("AlohaMiniMultiYCB-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", object_ids=ids,
                   human_render_camera_configs=dict(shader_pack="rt", width=W, height=H))
    be = env.unwrapped
    pick = build_skill("pick")
    env.reset(seed=0)
    acts = pick.plan(env, {"object_actor": be.object_actor_names[0], "descend_offset": offset,
                           "lift_height": 0.16, "lift_steps": 70, "hold_steps": 22,
                           "approach_steps": 25, "descend_steps": 30, "close_steps": 40})
    frames = [to_uint8(env.render())]               # initial pose
    for a in acts:
        env.step(torch.as_tensor(np.asarray(a, dtype=np.float32))[None])
        frames.append(to_uint8(env.render()))
    succ = bool(be.evaluate()["success"][0])
    env.close()
    return frames, succ


def main():
    made = []
    for oid, off in GRASP.items():
        for tag, distr in (("solo", False), ("scene", True)):
            frames, succ = render_one(oid, off, distr)
            path = os.path.join(OUT, f"{oid}_{tag}.mp4")
            encode(frames, path)
            sz = os.path.getsize(path)
            print(f"[{oid:24s} {tag:5s}] success={succ} frames={len(frames)} "
                  f"{W}x{H} spp{SPP} -> {sz//1024}KB {path}", flush=True)
            if succ:
                made.append(path)
    # montage of successful scene videos
    if made:
        listf = os.path.join(OUT, "concat.txt")
        with open(listf, "w") as fh:
            for p in made:
                fh.write(f"file '{p}'\n")
        montage = os.path.join(OUT, "MONTAGE_all.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
                        "-c", "copy", montage], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("MONTAGE:", montage, os.path.getsize(montage) // 1024, "KB")
    print("HQ_DONE made=", made)


if __name__ == "__main__":
    main()
