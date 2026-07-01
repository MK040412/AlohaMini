"""Sharded parallel renderer: one process per shard renders its slice of the job
list, skipping clips already on disk. Run N copies with SHARD=0..N-1, NSHARD=N to
use multiple vCPUs (sim) + overlap CPU sim with GPU ray tracing across processes.
"""
import os, sys
sys.path.insert(0, "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad")
import grasp_demo_v2 as G

OUT = G.OUT
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))

SLOW_OBJS = ["077_rubiks_cube", "062_dice", "009_gelatin_box", "070-a_colored_wood_blocks",
             "012_strawberry", "058_golf_ball", "073-f_lego_duplo", "065-a_cups", "061_foam_brick"]
DR_OBJS = ["077_rubiks_cube", "062_dice", "058_golf_ball"]
ANGLE = {"062_dice": [60, 45, 30], "058_golf_ball": [60, 45, 30],
         "012_strawberry": [60, 45, 30], "073-f_lego_duplo": [60, 45, 30]}


def jobs():
    js = []
    for oid in SLOW_OBJS:
        for v in ("side", "head"):
            js.append(("slow", oid, v))
    for oid in DR_OBJS:
        for k in range(3):
            js.append(("dr", oid, k))
    for oid, pitches in ANGLE.items():
        for p in pitches:
            js.append(("angle", oid, p, "side"))
    for p in ANGLE["062_dice"]:
        js.append(("angle", "062_dice", p, "head"))
    return js


def out_path(job):
    if job[0] == "slow":
        return os.path.join(OUT, f"{job[1]}_slow_{job[2]}.mp4")
    if job[0] == "dr":
        return os.path.join(OUT, f"{job[1]}_DR{job[2]}_side.mp4")
    if job[0] == "angle":
        return os.path.join(OUT, f"{job[1]}_pitch{job[2]}_{job[3]}.mp4")
    raise ValueError(job)


def run_job(job):
    if job[0] == "slow":
        return G.run(job[1], 0, 0.0, "slow", job[2])
    if job[0] == "dr":
        return G.run(job[1], 100 + job[2], 0.045, f"DR{job[2]}", "side")
    if job[0] == "angle":
        return G.run_angled(job[1], job[2], job[3])
    raise ValueError(job)


def main():
    all_jobs = jobs()
    mine = [j for i, j in enumerate(all_jobs) if i % NSHARD == SHARD]
    done = 0
    for job in mine:
        p = out_path(job)
        if os.path.exists(p) and os.path.getsize(p) > 50_000:
            print(f"[shard{SHARD}] SKIP exists {os.path.basename(p)}", flush=True)
            continue
        run_job(job)
        done += 1
    print(f"[shard{SHARD}] DONE rendered={done}/{len(mine)}", flush=True)


if __name__ == "__main__":
    main()
