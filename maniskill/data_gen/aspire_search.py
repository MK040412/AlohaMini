"""ASPIRE-style evolutionary search for the AlohaMini Pro NAV->MANIP grasp.

This intentionally reuses the validated code paths from pro_nav_pick_place.py:
feasibility-gated base station selection, root-relative base joint conversion,
the pick skill action layout, and Cartesian-line tilted descent using the live
left arm base for desired_approach_dir().
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import traceback
from copy import deepcopy
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill envs
import data_gen  # noqa: F401 - registers AlohaMiniMultiYCB-v1
from data_gen.intern_engine.skills.ik import (
    actor_position,
    desired_approach_dir,
    resolve_actor,
    solve_arm_ik_full_pose,
)
from grasp_demo_v2 import (
    HOLD,
    SETTLE,
    V_ARM,
    V_ARM_DESCEND,
    V_LIFT,
    SlowGrasp,
    _best_full_pose,
    interp,
)


RESULTS_PATH = (
    "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/"
    "scratchpad/aspire_search_results.json"
)

OBJ = "077_rubiks_cube"
V_BASE = 0.010
LIFT_H = 0.16

K = 4
T = 3
DEBUG_SEEDS = [0, 1]
VALIDATION_SEEDS = [2, 3, 4]
SEARCH_RNG_SEED = 20260702

PITCH_RANGE = (45.0, 75.0)
APPROACH_H_RANGE = (0.08, 0.14)
DESCEND_OFFSET_RANGE = (-0.01, 0.01)
JAW_DIRS = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
CLOSE_STEPS_CHOICES = [16, 24, 32]
STATION_BY_CHOICES = [0.01, 0.04, 0.08]

DEFAULT_CANDIDATE = {
    "pitch_deg": 60.0,
    "approach_h": 0.11,
    "descend_offset": 0.0,
    "jaw_dir": [1.0, 0.0, 0.0],
    "close_steps": 32,
    "station_by": 0.01,
}


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0])


def _tensor_action(action: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(action, dtype=torch.float32)[None]


def _canonical_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "pitch_deg": float(candidate["pitch_deg"]),
        "approach_h": float(candidate["approach_h"]),
        "descend_offset": float(candidate["descend_offset"]),
        "jaw_dir": [float(x) for x in candidate["jaw_dir"]],
        "close_steps": int(candidate["close_steps"]),
        "station_by": float(candidate["station_by"]),
    }


def candidate_summary(candidate: dict[str, Any]) -> str:
    c = _canonical_candidate(candidate)
    jaw = "x" if np.allclose(c["jaw_dir"], [1.0, 0.0, 0.0]) else "y"
    return (
        f"p={c['pitch_deg']:.1f},h={c['approach_h']:.3f},"
        f"off={c['descend_offset']:+.3f},jaw={jaw},"
        f"close={c['close_steps']},by={c['station_by']:.2f}"
    )


def make_env(seed: int):
    kwargs = dict(
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos_fixed_base",
        render_mode="rgb_array",
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="none",
        object_ids=[OBJ],
        robot_uid="aloha_mini_pro_v2",
        base_xy=(-0.40, 0.18),
        slot_override_xy=[(-0.13, -0.31)],
    )
    try:
        env = gym.make("AlohaMiniMultiYCB-v1", object_xy_noise=0.02, **kwargs)
    except TypeError:
        env = gym.make("AlohaMiniMultiYCB-v1", **kwargs)
    env.reset(seed=int(seed), options={"episode_index": 0})
    return env


def execute(candidate: dict[str, Any], seed: int) -> float:
    """Run one no-render NAV->MANIP grasp rollout and return its scalar score."""

    candidate = _canonical_candidate(candidate)
    summary = candidate_summary(candidate)
    ik_details: list[dict[str, Any]] = []
    env = None
    try:
        env = make_env(seed)
        be = env.unwrapped
        robot = be.agent.robot
        names = [j.name for j in robot.active_joints]
        idx = {n: i for i, n in enumerate(names)}
        base_ids = [
            idx["root_x_axis_joint"],
            idx["root_y_axis_joint"],
            idx["root_z_rotation_joint"],
        ]

        object_name = be.object_actor_names[0]
        obj0 = actor_position(resolve_actor(env, object_name)).copy()

        grasp = SlowGrasp(env)
        skill, lay = grasp.skill, grasp.lay
        op, cl = skill.open_gripper, skill.closed_gripper

        def qnow() -> np.ndarray:
            q = robot.get_qpos()
            return (
                q[0].detach().cpu().numpy()
                if hasattr(q, "detach")
                else np.asarray(q).reshape(-1)
            ).copy()

        root_p = robot.pose.p
        root_xy = (
            root_p[0].detach().cpu().numpy()
            if hasattr(root_p, "detach")
            else np.asarray(root_p).reshape(-1)
        )[:2].astype(np.float64)

        def world_to_joint(world_xy_yaw: tuple[float, float, float] | np.ndarray) -> np.ndarray:
            out = np.array(world_xy_yaw, dtype=np.float64).copy()
            out[0] -= root_xy[0]
            out[1] -= root_xy[1]
            return out

        def set_q(q: np.ndarray) -> None:
            robot.set_qpos(torch.as_tensor(q[None], dtype=torch.float32))

        def act(base_t: np.ndarray, arm_q: np.ndarray, grip: float, lift: float) -> np.ndarray:
            """base_t is WORLD (x, y) or (x, y, yaw), converted to root-relative joints."""

            a = skill.current_action_template(env)
            yaw = float(base_t[2]) if len(base_t) > 2 else 0.0
            j = world_to_joint((float(base_t[0]), float(base_t[1]), yaw))
            a[0], a[1], a[2] = float(j[0]), float(j[1]), float(j[2])
            a[3] = float(lift)
            a[lay["right_grip"]] = op
            skill.set_arm_action(a, "left", arm_q, grip)
            return a.astype(np.float32)

        def run(actions: list[np.ndarray]) -> None:
            for a in actions:
                env.step(_tensor_action(a))

        def arm_base_xy() -> np.ndarray:
            for link in be.agent.robot.get_links():
                if link.name == "left_base":
                    p = link.pose.p
                    arr = (
                        p[0].detach().cpu().numpy()
                        if hasattr(p, "detach")
                        else np.asarray(p).reshape(-1)
                    )
                    return arr[:2].astype(np.float64)
            raise RuntimeError("Could not find left_base link for live approach direction.")

        def select_station(target_pt: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
            jaw_dir = np.asarray(candidate["jaw_dir"], dtype=np.float32)
            q0 = qnow()
            rest = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
            best: tuple[float, float, float, float, np.ndarray, float] | None = None
            arm_offsets = {
                0.0: (0.156, -0.041),
                -np.pi / 2: (-0.041, -0.156),
            }
            try:
                for yaw, (ox, _oy) in arm_offsets.items():
                    for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
                        by = float(candidate["station_by"])
                        bx = float(target_pt[0] - ox + dx)
                        j = world_to_joint((bx, by, yaw))
                        q = q0.copy()
                        q[base_ids[0]], q[base_ids[1]], q[base_ids[2]] = j[0], j[1], j[2]
                        set_q(q)

                        hold = act(np.array([bx, by, yaw], np.float32), rest, op, 0.0)
                        for _ in range(6):
                            env.step(_tensor_action(hold))

                        qs = qnow()
                        base_err = float(
                            np.hypot(qs[base_ids[0]] - j[0], qs[base_ids[1]] - j[1])
                        )
                        detail = {
                            "label": label,
                            "yaw_deg": float(np.degrees(yaw)),
                            "bx": bx,
                            "by": by,
                            "dx": float(dx),
                            "base_err": base_err,
                        }
                        if base_err > 0.02:
                            detail["rejected"] = "base_settle"
                            ik_details.append(detail)
                            continue

                        ab = arm_base_xy()
                        appr_dir = desired_approach_dir(
                            target_pt,
                            candidate["pitch_deg"],
                            base_xy=tuple(ab),
                        ).astype(np.float32)
                        r = solve_arm_ik_full_pose(
                            env,
                            target_pt,
                            appr_dir,
                            jaw_dir,
                            arm="left",
                            lift_position=0.0,
                            shoulder_lift_seed=1.0,
                            max_iters=120,
                        )
                        dist = float(np.linalg.norm(ab - target_pt[:2]))
                        comfort = abs(dist - 0.20)
                        score = float(r.error + 0.02 * comfort)
                        detail.update(
                            {
                                "arm_base_xy": [float(ab[0]), float(ab[1])],
                                "target": [float(x) for x in target_pt],
                                "ik_error": float(r.error),
                                "ik_success": bool(r.success),
                                "ik_iters": int(r.iterations),
                                "ik_ori_error": (
                                    None
                                    if r.wrist_roll_score is None
                                    else float(r.wrist_roll_score)
                                ),
                                "comfort": comfort,
                                "score": score,
                            }
                        )
                        ik_details.append(detail)
                        if best is None or score < best[0]:
                            best = (score, bx, by, yaw, r.arm_qpos.copy(), float(r.error))
            finally:
                set_q(q0)

            if best is None:
                raise RuntimeError(f"no physically-valid station found for {label}")

            print(
                f"    [FEAS {label}] station=({best[1]:+.3f},{best[2]:+.3f},"
                f"yaw={np.degrees(best[3]):.0f}deg) ik_err={best[5]:.4f}",
                flush=True,
            )
            return np.array([best[1], best[2], best[3]], np.float32), best[4]

        rest_q = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
        base0 = qnow()[base_ids].astype(np.float32)
        base0[0] += root_xy[0]
        base0[1] += root_xy[1]

        grasp_pt = obj0.astype(np.float32)
        station, station_arm_seed = select_station(grasp_pt, "pick")

        nav = [act(b, rest_q, op, 0.0) for b in interp(base0, station, V_BASE)]
        nav += [nav[-1]] * 20
        run(nav)

        grasp_pt = actor_position(resolve_actor(env, object_name)).astype(np.float32)
        lb_xy = arm_base_xy()
        target_pt = (
            grasp_pt + np.array([0.0, 0.0, candidate["descend_offset"]], np.float32)
        ).astype(np.float32)
        appr_dir = desired_approach_dir(
            target_pt,
            candidate["pitch_deg"],
            base_xy=tuple(lb_xy),
        ).astype(np.float32)
        jaw_dir = np.asarray(candidate["jaw_dir"], dtype=np.float32)
        pre_pt = (target_pt - appr_dir * candidate["approach_h"]).astype(np.float32)

        desc = _best_full_pose(
            env,
            target_pt,
            appr_dir,
            jaw_dir,
            "left",
            0.0,
            seed=station_arm_seed,
        )
        ik_details.append(
            {
                "label": "pick_desc",
                "target": [float(x) for x in target_pt],
                "ik_error": float(desc.error),
                "ik_success": bool(desc.success),
                "ik_iters": int(desc.iterations),
                "ik_ori_error": (
                    None if desc.wrist_roll_score is None else float(desc.wrist_roll_score)
                ),
            }
        )
        appr = _best_full_pose(
            env,
            pre_pt,
            appr_dir,
            jaw_dir,
            "left",
            0.0,
            seed=desc.arm_qpos,
        )
        ik_details.append(
            {
                "label": "pick_approach",
                "target": [float(x) for x in pre_pt],
                "ik_error": float(appr.error),
                "ik_success": bool(appr.success),
                "ik_iters": int(appr.iterations),
                "ik_ori_error": (
                    None if appr.wrist_roll_score is None else float(appr.wrist_roll_score)
                ),
            }
        )
        print(f"    [PICK] ik appr={appr.error:.4f} desc={desc.error:.4f}", flush=True)

        descent = [appr.arm_qpos]
        seed_q = appr.arm_qpos
        for s in np.linspace(candidate["approach_h"], 0.0, 9)[1:]:
            w = _best_full_pose(
                env,
                (target_pt - appr_dir * s).astype(np.float32),
                appr_dir,
                jaw_dir,
                "left",
                0.0,
                seed=seed_q,
            )
            ik_details.append(
                {
                    "label": "pick_cartesian",
                    "s": float(s),
                    "ik_error": float(w.error),
                    "ik_success": bool(w.success),
                    "ik_iters": int(w.iterations),
                    "ik_ori_error": (
                        None if w.wrist_roll_score is None else float(w.wrist_roll_score)
                    ),
                }
            )
            descent.append(w.arm_qpos)
            seed_q = w.arm_qpos
        desc_q = descent[-1]

        acts: list[np.ndarray] = []
        for q in interp(rest_q, appr.arm_qpos, V_ARM):
            acts.append(act(station, q, op, 0.0))
        for q0, q1 in zip(descent[:-1], descent[1:]):
            for q in interp(q0, q1, V_ARM_DESCEND):
                acts.append(act(station, q, op, 0.0))
        close_steps = int(candidate["close_steps"])
        for k in range(1, close_steps + 1):
            grip = op + (cl - op) * k / close_steps
            acts.append(act(station, desc_q, grip, 0.0))
        for _ in range(SETTLE):
            acts.append(act(station, desc_q, cl, 0.0))
        for lift_z in interp([0.0], [LIFT_H], V_LIFT):
            acts.append(act(station, desc_q, cl, float(lift_z[0])))
        for _ in range(min(6, HOLD)):
            acts.append(act(station, desc_q, cl, LIFT_H))
        run(acts)

        info = be.evaluate()
        success = _scalar_bool(info["success"])
        objf = actor_position(resolve_actor(env, object_name)).copy()
        if success:
            score = 1.0
        else:
            xy_push_dist = float(np.linalg.norm(objf[:2] - obj0[:2]))
            score = float(0.5 * math.exp(-xy_push_dist))

        print(
            f"  [seed={seed}] {summary} -> score={score:.2f}",
            flush=True,
        )
        return score
    except Exception:
        print(f"  [seed={seed}] {summary} crashed; traceback and IK details follow:", flush=True)
        traceback.print_exc()
        print("  IK_DETAILS " + json.dumps(ik_details, indent=2), flush=True)
        print(f"  [seed={seed}] {summary} -> score=0.00", flush=True)
        return 0.0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def random_candidate(rng: random.Random) -> dict[str, Any]:
    return {
        "pitch_deg": rng.uniform(*PITCH_RANGE),
        "approach_h": rng.uniform(*APPROACH_H_RANGE),
        "descend_offset": rng.uniform(*DESCEND_OFFSET_RANGE),
        "jaw_dir": list(rng.choice(JAW_DIRS)),
        "close_steps": int(rng.choice(CLOSE_STEPS_CHOICES)),
        "station_by": float(rng.choice(STATION_BY_CHOICES)),
    }


def _jitter(value: float, lo: float, hi: float, rng: random.Random) -> float:
    sigma = 0.10 * (hi - lo)
    return float(min(hi, max(lo, value + rng.gauss(0.0, sigma))))


def _maybe_flip(current: Any, choices: list[Any], rng: random.Random) -> Any:
    if rng.random() >= 0.20:
        return deepcopy(current)
    options = [deepcopy(c) for c in choices if c != current]
    return rng.choice(options) if options else deepcopy(current)


def mutate(parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    c = _canonical_candidate(parent)
    c["pitch_deg"] = _jitter(c["pitch_deg"], *PITCH_RANGE, rng)
    c["approach_h"] = _jitter(c["approach_h"], *APPROACH_H_RANGE, rng)
    c["descend_offset"] = _jitter(c["descend_offset"], *DESCEND_OFFSET_RANGE, rng)
    c["jaw_dir"] = list(_maybe_flip(c["jaw_dir"], [list(j) for j in JAW_DIRS], rng))
    c["close_steps"] = int(_maybe_flip(c["close_steps"], CLOSE_STEPS_CHOICES, rng))
    c["station_by"] = float(_maybe_flip(c["station_by"], STATION_BY_CHOICES, rng))
    return c


def write_results(rounds: list[dict[str, Any]], best: dict[str, Any], validation: list[float]) -> None:
    payload = {
        "rounds": rounds,
        "best": best,
        "validation_scores": [float(x) for x in validation],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    rng = random.Random(SEARCH_RNG_SEED)
    rounds: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    total_rollouts = 0
    early_stop = False

    for round_idx in range(T):
        if round_idx == 0:
            candidates = [random_candidate(rng) for _ in range(K - 1)]
            candidates.append(deepcopy(DEFAULT_CANDIDATE))
        else:
            survivors = sorted(evaluated, key=lambda r: r["mean_score"], reverse=True)[:3]
            candidates = [mutate(survivors[i % len(survivors)]["candidate"], rng) for i in range(K)]

        print(f"\n[ROUND {round_idx + 1}] evaluating {len(candidates)} candidates", flush=True)
        round_candidates: list[dict[str, Any]] = []
        round_scores: list[float] = []

        for cand_idx, candidate in enumerate(candidates):
            candidate = _canonical_candidate(candidate)
            print(f"[CAND {cand_idx + 1}/{len(candidates)}] {candidate_summary(candidate)}", flush=True)
            seed_scores = []
            for seed in DEBUG_SEEDS:
                seed_scores.append(execute(candidate, seed))
                total_rollouts += 1
            mean_score = float(np.mean(seed_scores))
            record = {
                "candidate": candidate,
                "seed_scores": [float(x) for x in seed_scores],
                "mean_score": mean_score,
            }
            evaluated.append(record)
            round_candidates.append(candidate)
            round_scores.append(mean_score)
            if best_record is None or mean_score > best_record["mean_score"]:
                best_record = record

            print(
                f"[CAND {cand_idx + 1}] mean={mean_score:.3f} "
                f"seeds={[round(float(s), 3) for s in seed_scores]}",
                flush=True,
            )
            if all(float(s) >= 1.0 for s in seed_scores):
                print("[EARLY STOP] candidate succeeded on all debug seeds", flush=True)
                early_stop = True
                break

        rounds.append({"candidates": round_candidates, "scores": round_scores})
        interim_best = best_record or {
            "candidate": deepcopy(DEFAULT_CANDIDATE),
            "seed_scores": [],
            "mean_score": 0.0,
        }
        write_results(
            rounds,
            {
                **interim_best["candidate"],
                "score": float(interim_best["mean_score"]),
                "debug_scores": [float(x) for x in interim_best["seed_scores"]],
                "total_rollouts": total_rollouts,
            },
            [],
        )
        if early_stop:
            break

    assert best_record is not None
    best_candidate = best_record["candidate"]
    print(f"\n[BEST DEBUG] {candidate_summary(best_candidate)} score={best_record['mean_score']:.3f}", flush=True)
    print(f"[VALIDATION] held-out seeds {VALIDATION_SEEDS}", flush=True)
    validation_scores = []
    for seed in VALIDATION_SEEDS:
        validation_scores.append(execute(best_candidate, seed))
        total_rollouts += 1

    best_json = {
        **best_candidate,
        "score": float(best_record["mean_score"]),
        "debug_scores": [float(x) for x in best_record["seed_scores"]],
        "total_rollouts": total_rollouts,
    }
    write_results(rounds, best_json, validation_scores)
    print(
        f"\n[DONE] best={candidate_summary(best_candidate)} "
        f"debug_score={best_record['mean_score']:.3f} "
        f"validation={[round(float(s), 3) for s in validation_scores]} "
        f"rollouts={total_rollouts}",
        flush=True,
    )
    print(f"[RESULTS] {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
