#!/usr/bin/env python3
"""AlohaMini demo runner — one command per feature, GUI or headless.

Run from the maniskill/ directory:

    python demo.py empty                    # robot in an empty scene (headless smoke)
    python demo.py empty --render           # ... in the GUI viewer
    python demo.py table --render           # tabletop cube-pick scene (AlohaMini 1)
    python demo.py ycb --render             # multi-object YCB table   [needs YCB assets]
    python demo.py replicacad --render      # apartment scene          [needs ReplicaCAD assets]

Options:
    --robot mini1|mini2   which robot, where the scene allows it (default: scene's native robot)
    --render              open the SAPIEN viewer (default: headless, steps N times and exits)
    --steps N             headless step count (default 100)
    --shader default|rt-fast|rt   render quality for --render (default: default)

Missing optional assets are detected and the exact download command is printed.
"""
import argparse
import sys
from pathlib import Path

DATA_DIR = Path.home() / ".maniskill" / "data"

# scene key -> (env_id, allowed robots, default robot, required asset)
SCENES = {
    "empty": ("Empty-v1", ("mini1", "mini2"), "mini2", None),
    "table": ("AlohaMiniTablePick-v1", ("mini1",), "mini1", None),
    "ycb": ("AlohaMiniMultiYCB-v1", ("mini1",), "mini1", "ycb"),
    "replicacad": ("ReplicaCAD_SceneManipulation-v1", ("mini1", "mini2"), "mini1", "replicacad"),
}
ROBOT_UIDS = {"mini1": "aloha_mini_1", "mini2": "aloha_mini_2"}
ASSETS = {
    "ycb": (DATA_DIR / "assets" / "mani_skill2_ycb",
            "python -m mani_skill.utils.download_asset ycb"),
    "replicacad": (DATA_DIR / "scene_datasets" / "replica_cad_dataset",
                   "python -m mani_skill.utils.download_asset ReplicaCAD"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=SCENES)
    ap.add_argument("--robot", choices=ROBOT_UIDS, default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--shader", choices=("default", "rt-fast", "rt"), default="default")
    args = ap.parse_args()

    env_id, allowed, default_robot, asset = SCENES[args.scene]
    robot = args.robot or default_robot
    if robot not in allowed:
        ap.error(f"scene '{args.scene}' supports --robot {'|'.join(allowed)} only")

    if asset is not None:
        path, cmd = ASSETS[asset]
        if not path.exists():
            print(f"[DEMO] '{args.scene}' needs assets that are not installed ({path}).")
            print(f"[DEMO] download them once with:\n    {cmd}")
            return 1

    import gymnasium as gym
    try:
        import mani_skill.agents.robots.aloha_mini  # noqa: installed via install.py
    except ImportError:
        import agents.aloha_mini  # noqa: fallback — repo checkout without install.py
    import mani_skill.envs  # noqa
    if args.scene in ("table", "ycb"):
        import data_gen.tasks  # noqa: registers the AlohaMini* task envs

    kwargs = dict(obs_mode="state", reward_mode="none", sim_backend="physx_cpu")
    if len(allowed) > 1:
        # the repo task envs pin their own robot; only pass a choice where one exists
        kwargs["robot_uids"] = ROBOT_UIDS[robot]
    if args.render:
        kwargs["render_mode"] = "human"
        kwargs["human_render_camera_configs"] = dict(shader_pack=args.shader)

    print(f"[DEMO] {args.scene}: gym.make({env_id!r}, robot={ROBOT_UIDS[robot]})", flush=True)
    env = gym.make(env_id, **kwargs)
    reset_options = dict(reconfigure=True) if args.scene == "replicacad" else None
    env.reset(seed=0, options=reset_options)
    hold = env.action_space.sample() * 0

    if args.render:
        print("[DEMO] close the SAPIEN window to exit", flush=True)
        while True:
            env.step(hold)
            env.render()
    for _ in range(args.steps):
        env.step(hold)
    env.close()
    print(f"[DEMO] {args.scene}: OK ({args.steps} steps, headless)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
