"""Assemble category reels + a master showcase from the rendered demo clips.

Globs data_gen/output/demo_slow and groups by kind (slow side/head, DR, angles),
burns a short title card label on each clip via drawtext, and concatenates.
Re-encodes (not stream-copy) so clips with differing frame counts concatenate cleanly.
"""
import os, glob, subprocess

OUT = "/home/perelman/AlohaMini/maniskill/data_gen/output/demo_slow"
SHOW = os.path.join(OUT, "showcase")
os.makedirs(SHOW, exist_ok=True)
W, H = 1920, 1280


def label_clip(src, text, dst):
    safe = text.replace(":", "\\:").replace("'", "")
    subprocess.run([
        "ffmpeg", "-y", "-i", src, "-vf",
        f"scale={W}:{H},drawtext=text='{safe}':x=(w-text_w)/2:y=h-90:fontsize=46:"
        f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=14",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", dst],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst if os.path.exists(dst) else None


def reel(labeled, out_name):
    labeled = [p for p in labeled if p]
    if not labeled:
        return None
    lf = os.path.join(SHOW, out_name + ".txt")
    with open(lf, "w") as fh:
        for p in labeled:
            fh.write(f"file '{p}'\n")
    out = os.path.join(SHOW, out_name)
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lf, "-c", "copy", out],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if os.path.exists(out) else None


def nice(oid):
    return oid.split("_", 1)[-1].replace("_", " ") if "_" in oid else oid


def combined_side_head():
    """Per-object SIDE|HEAD side-by-side clips."""
    outs = []
    for sp in sorted(glob.glob(os.path.join(OUT, "*_slow_side.mp4"))):
        oid = os.path.basename(sp).replace("_slow_side.mp4", "")
        hp = os.path.join(OUT, f"{oid}_slow_head.mp4")
        if not os.path.exists(hp):
            continue
        dst = os.path.join(OUT, f"{oid}_slow_COMBINED.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", sp, "-i", hp, "-filter_complex",
            "[0:v]scale=960:640,drawtext=text='SIDE':x=18:y=14:fontsize=30:fontcolor=white:"
            "box=1:boxcolor=black@0.5[a];"
            "[1:v]scale=960:640,drawtext=text='HEAD':x=18:y=14:fontsize=30:fontcolor=white:"
            "box=1:boxcolor=black@0.5[b];[a][b]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
            "-pix_fmt", "yuv420p", dst],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(dst):
            outs.append(dst)
            print(f"COMBINED {os.path.basename(dst)} ({os.path.getsize(dst)//1024}KB)", flush=True)
    return outs


def main():
    combined_side_head()
    tmp = []
    # 1) slow top-down, side view, all objects
    slow_side = sorted(glob.glob(os.path.join(OUT, "*_slow_side.mp4")))
    r1 = []
    for p in slow_side:
        oid = os.path.basename(p).replace("_slow_side.mp4", "")
        r1.append(label_clip(p, f"SLOW GRASP  -  {nice(oid)}  (side, full-RT)",
                              os.path.join(SHOW, f"_l_{oid}_slowside.mp4")))
    reel_slow = reel(r1, "REEL_1_slow_side.mp4")
    # 2) head view
    r2 = []
    for p in sorted(glob.glob(os.path.join(OUT, "*_slow_head.mp4"))):
        oid = os.path.basename(p).replace("_slow_head.mp4", "")
        r2.append(label_clip(p, f"HEAD CAMERA  -  {nice(oid)}",
                             os.path.join(SHOW, f"_l_{oid}_slowhead.mp4")))
    reel_head = reel(r2, "REEL_2_head_view.mp4")
    # 3) domain randomization
    r3 = []
    for p in sorted(glob.glob(os.path.join(OUT, "*_DR*_side.mp4"))):
        base = os.path.basename(p).replace(".mp4", "")
        oid = base.split("_DR")[0]
        r3.append(label_clip(p, f"DOMAIN RANDOMIZATION  -  {nice(oid)}  (random position)",
                             os.path.join(SHOW, f"_l_{base}.mp4")))
    reel_dr = reel(r3, "REEL_3_domain_random.mp4")
    # 4) approach angles
    r4 = []
    for p in sorted(glob.glob(os.path.join(OUT, "*_pitch*_side.mp4"))):
        base = os.path.basename(p).replace(".mp4", "")
        oid = base.split("_pitch")[0]
        pitch = base.split("_pitch")[1].split("_")[0]
        r4.append(label_clip(p, f"APPROACH ANGLE {pitch} deg  -  {nice(oid)}",
                             os.path.join(SHOW, f"_l_{base}.mp4")))
    reel_ang = reel(r4, "REEL_4_approach_angles.mp4")
    # master = all reels back to back
    master = reel([reel_slow, reel_head, reel_dr, reel_ang], "MASTER_showcase.mp4")
    for nm, pth in [("slow", reel_slow), ("head", reel_head), ("DR", reel_dr),
                    ("angles", reel_ang), ("MASTER", master)]:
        if pth:
            print(f"{nm:8s} {os.path.getsize(pth)//1024:6d}KB  {pth}", flush=True)
    print("SHOWCASE_DONE")


if __name__ == "__main__":
    main()
