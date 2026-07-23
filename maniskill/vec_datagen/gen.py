#!/usr/bin/env python3
"""GPU-parallel task data generation for AlohaMini 2 — spec-driven front-end.

All data generation runs BATCHED on the GPU (physx_cuda, N environments stepped
in lock-step) with CuRobo batched motion planning; this wrapper only translates
a task spec into the env-var + argv interface of vec_pick_gen_mo.py.

Usage (from maniskill/):

    python -m vec_datagen.gen --colors red,green,blue --batches 3
    python -m vec_datagen.gen --spec my_tasks.json

Spec JSON = the CLI flags as keys (see --help):

    {"colors": ["red", "yellow", "green"], "batches": 3, "envs": 16,
     "episode_base": 700000, "station_noise": 0.02, "yaw_rand": true,
     "grasp_dart": 2, "arm_init_noise": 0.0, "out_dir": "instr_out"}

Each batch attempts `envs` episodes in parallel ("pick up the {color} cube",
per-env random target color); only successful grasps+lifts are saved as .npz
episodes into out_dir. Requires CuRobo and a CUDA GPU (N=16 fits in 8 GB).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, help="JSON file with the same keys as the flags below")
    ap.add_argument("--colors", default="red,green,blue",
                    help="comma-separated cube colors on the table (3-4); target is per-env random")
    ap.add_argument("--batches", type=int, default=3, help="GPU batches to run (episodes = batches x envs attempts)")
    ap.add_argument("--envs", type=int, default=16, help="parallel envs per batch (16 = 8 GB GPU ceiling)")
    ap.add_argument("--episode-base", type=int, default=500000, help="starting episode id for output files")
    ap.add_argument("--station-noise", type=float, default=0.0, help="base-station XY noise, m (diversity)")
    ap.add_argument("--yaw-rand", action="store_true", help="randomize grasp yaw over the 4 verified yaws")
    ap.add_argument("--grasp-dart", type=int, default=0, help="DART perturb->replan recovery rounds per grasp")
    ap.add_argument("--arm-init-noise", type=float, default=0.0, help="initial arm qpos noise, rad (proprio DR)")
    ap.add_argument("--out-dir", default=None, help="output dir for episode .npz (default: vec_datagen/instr_out)")
    args = ap.parse_args()

    if args.spec:
        spec = json.loads(args.spec.read_text())
        for k, v in spec.items():
            setattr(args, k.replace("-", "_"), v)
    colors = args.colors.split(",") if isinstance(args.colors, str) else list(args.colors)

    env = os.environ.copy()
    env["STATION_NOISE"] = str(args.station_noise)
    env["YAW_RAND"] = "1" if args.yaw_rand else "0"
    env["GRASP_DART_N"] = str(args.grasp_dart)
    env["ARM_INIT_NOISE"] = str(args.arm_init_noise)
    if args.out_dir:
        env["INSTR_OUT_DIR"] = str(Path(args.out_dir).resolve())

    cmd = [sys.executable, str(HERE / "vec_pick_gen_mo.py"),
           str(args.batches), str(args.envs), str(args.episode_base), *colors]
    print(f"[GEN] physx_cuda x{args.envs} envs, {args.batches} batches, colors={colors}", flush=True)
    print(f"[GEN] diversity: station_noise={args.station_noise} yaw_rand={env['YAW_RAND']} "
          f"dart={args.grasp_dart} arm_init_noise={args.arm_init_noise}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=HERE.parent)
    out = Path(env.get("INSTR_OUT_DIR", HERE / "instr_out"))
    n = len(list(out.glob("*.npz")))
    print(f"[GEN] done (exit {r.returncode}) — {n} episode files now in {out}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
