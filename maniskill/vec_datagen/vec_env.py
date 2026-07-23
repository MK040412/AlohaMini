"""Vectorized AM2 Pro table pick-place env for physx_gpu batch data generation.
Cube on a WIDE static table + a place target, per-env domain randomization. Left
arm only (right arm folded). Designed for N-env GPU batch from the ground up.

The support is a WIDE static table (not the old 2 cm pedestal) that spans the whole
DR workspace, so the cube AND the target rest on it anywhere in the DR range:
  * table half-size [0.30, 0.18, 0.35], center [-0.015, -0.45, 0.35] -> top z=0.70.
Cube DR : cube_xy_noise=0.06 around (-0.13, -0.45)  -> x in [-0.19,-0.07], y in [-0.51,-0.39].
Target  : x in [0.02, 0.18], y in [-0.51, -0.39]    (within fixed-base reach after re-station).
Cube color + size are jittered per-episode for visual DR (state-only: harmless to
physics/plan; applied at _load_scene so they need a reconfigure to change)."""
from __future__ import annotations
import numpy as np, torch, sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose as MSPose
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents.aloha_mini  # noqa: registers aloha_mini_2

TABLE_TOP_Z = 0.70
CUBE_HALF = 0.02
CUBE_XY = (-0.13, -0.45)
LIFT_SUCCESS = 0.06

# Wide static table spanning the whole DR workspace (top at TABLE_TOP_Z).
TABLE_HALF = (0.30, 0.18, TABLE_TOP_Z / 2)        # [0.30, 0.18, 0.35]
TABLE_CENTER = (-0.015, -0.45, TABLE_TOP_Z / 2)   # [-0.015, -0.45, 0.35]

# Target DR box (on the table, reachable after a re-station to the target).
TARGET_X = (0.02, 0.18)
TARGET_Y = (-0.51, -0.39)


@register_env("AM2VecPickPlace-v1", max_episode_steps=200)
class AM2VecPickPlace(BaseEnv):
    SUPPORTED_ROBOTS = ["aloha_mini_2"]

    def __init__(self, *args, cube_xy_noise=0.06, target_xy_noise=0.10,
                 cube_half=CUBE_HALF, cube_color=(1.0, 0.0, 0.0, 1.0), **kwargs):
        self.cube_xy_noise = cube_xy_noise
        self.target_xy_noise = target_xy_noise      # kept for back-compat (unused for target box)
        self._cube_half = float(cube_half)          # read at _load_scene (reconfigure to change)
        self._cube_color = list(cube_color)
        super().__init__(*args, robot_uids="aloha_mini_2", **kwargs)

    @property
    def _default_sensor_configs(self):
        # The observation sensors are exactly the FIVE real AM2 Pro cameras, which
        # the robot agent (aloha_mini_2) provides via its own _sensor_configs
        # (front / back / chest + left/right wrist). Add NO extra env camera here,
        # so the sensor set matches the physical 5-camera rig.
        return []

    @property
    def _default_human_render_camera_configs(self):
        # A fixed 3rd-person camera used only for render()/mp4 previews and debug —
        # NOT part of the policy observation (that is the 5 robot cameras above).
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 1.05],
                                    target=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z))
        return CameraConfig("render_camera", pose, 640, 480, np.pi / 2.2, 0.01, 100)

    def _load_scene(self, options: dict):
        # WIDE static table: cube + target rest on it anywhere in the DR range.
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=list(TABLE_HALF))
        b.add_box_visual(half_size=list(TABLE_HALF),
                         material=sapien.render.RenderMaterial(base_color=[0.55, 0.45, 0.35, 1]))
        b.initial_pose = sapien.Pose(p=list(TABLE_CENTER))
        self.platform = b.build_static(name="table")
        self.cube = actors.build_cube(self.scene, half_size=self._cube_half, color=self._cube_color,
            name="cube", initial_pose=sapien.Pose(p=[CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + self._cube_half]))
        # visual-only target marker
        self.target = actors.build_sphere(self.scene, radius=0.02, color=[0, 1, 0, 0.6],
            name="target", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0.1, -0.45, TABLE_TOP_Z]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
            self.agent.robot.set_pose(sapien.Pose())
            dx = (torch.rand(b) * 2 - 1) * self.cube_xy_noise
            dy = (torch.rand(b) * 2 - 1) * self.cube_xy_noise
            cxy = torch.tensor(CUBE_XY) + torch.stack([dx, dy], 1)
            ch = self._cube_half
            cp = torch.zeros(b, 3); cp[:, :2] = cxy; cp[:, 2] = TABLE_TOP_Z + ch
            q = torch.zeros(b, 4); q[:, 0] = 1
            self.cube.set_pose(MSPose.create_from_pq(p=cp, q=q))
            tx = TARGET_X[0] + torch.rand(b) * (TARGET_X[1] - TARGET_X[0])
            ty = TARGET_Y[0] + torch.rand(b) * (TARGET_Y[1] - TARGET_Y[0])
            tp = torch.zeros(b, 3); tp[:, 0] = tx; tp[:, 1] = ty; tp[:, 2] = TABLE_TOP_Z
            self.target.set_pose(MSPose.create_from_pq(p=tp, q=q))

    def evaluate(self):
        cube_z = self.cube.pose.p[:, 2]
        lifted = cube_z > (TABLE_TOP_Z + self._cube_half + LIFT_SUCCESS)
        return {"success": lifted, "cube_z": cube_z}

    # ---- dense reward for RL (bypasses BC covariate shift: learns from own rollouts) ----
    def compute_dense_reward(self, obs, action, info):
        tcp = self.agent.robot.links_map["left_Fixed_Jaw"].pose.p          # (n,3)
        d = torch.linalg.norm(tcp - self.cube.pose.p, dim=1)
        reach = 1.0 - torch.tanh(5.0 * d)                                  # reach the cube
        grasped = self.agent.is_grasping(self.cube, arm_id=1).float()      # close + hold
        lift = torch.clamp((self.cube.pose.p[:, 2] - (TABLE_TOP_Z + self._cube_half)) / 0.10, 0, 1)
        rew = reach + grasped + 3.0 * lift * grasped
        rew = torch.where(info["success"], rew + 5.0, rew)
        return rew

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 9.0

    def _get_obs_extra(self, info):
        return dict(cube_pos=self.cube.pose.p, target_pos=self.target.pose.p)


# ============================ multi-object instruction following =============
# Named colours (vivid, well-separated in hue) so an instruction "pick up the
# <color> cube" grounds unambiguously to exactly one object on the table.
NAMED_COLORS = {
    "red":    [0.85, 0.10, 0.10, 1.0],
    "green":  [0.10, 0.70, 0.15, 1.0],
    "blue":   [0.12, 0.25, 0.85, 1.0],
    "yellow": [0.92, 0.80, 0.10, 1.0],
    "orange": [0.95, 0.45, 0.05, 1.0],
    "purple": [0.60, 0.15, 0.78, 1.0],
}
# Cube-centre placement region on the table (all reachable after a re-station).
MO_X = (-0.18, 0.13)
MO_Y = (-0.52, -0.39)
MO_MIN_SEP = 0.085          # min centre-to-centre spacing so a grasp clears neighbours


@register_env("AM2MultiObject-v1", max_episode_steps=400)
class AM2MultiObject(BaseEnv):
    """Multiple distinct-colour cubes on the table + a neutral place marker. Each
    episode's instruction names ONE cube by colour; the data-gen must pick THAT
    cube (treating the others as obstacles) and place it on the marker. This is the
    instruction-following task that grounds language -> the correct object."""
    SUPPORTED_ROBOTS = ["aloha_mini_2"]

    def __init__(self, *args, obj_colors=("red", "green", "blue"),
                 cube_half=CUBE_HALF, **kwargs):
        self._obj_colors = list(obj_colors)     # names, spawn order (index = object id)
        self._cube_half = float(cube_half)
        super().__init__(*args, robot_uids="aloha_mini_2", **kwargs)

    @property
    def _default_sensor_configs(self):
        return []                               # the 5 robot cameras are the sensors

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 1.05],
                                    target=(0.0, -0.45, TABLE_TOP_Z))
        return CameraConfig("render_camera", pose, 640, 480, np.pi / 2.2, 0.01, 100)

    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=list(TABLE_HALF))
        b.add_box_visual(half_size=list(TABLE_HALF),
                         material=sapien.render.RenderMaterial(base_color=[0.55, 0.45, 0.35, 1]))
        b.initial_pose = sapien.Pose(p=list(TABLE_CENTER))
        self.platform = b.build_static(name="table")

        self.cubes = []
        for i, cname in enumerate(self._obj_colors):
            col = NAMED_COLORS[cname]
            # spread initial poses so no two share a spot before _initialize_episode
            init = sapien.Pose(p=[MO_X[0] + 0.06 * i, MO_Y[0], TABLE_TOP_Z + self._cube_half])
            self.cubes.append(actors.build_cube(
                self.scene, half_size=self._cube_half, color=col,
                name=f"cube_{i}_{cname}", initial_pose=init))
        # neutral place marker (flat gray disk, visual only) — not a coloured cube,
        # so it is never confused with an instruction target object.
        self.target = actors.build_cylinder(
            self.scene, radius=0.03, half_length=0.002, color=[0.7, 0.7, 0.72, 0.9],
            name="marker", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0.10, -0.45, TABLE_TOP_Z]))

    def _sample_positions(self, b, n, rng_seed):
        """Rejection-sample n cube centres + a marker, all >= MO_MIN_SEP apart."""
        g = torch.Generator().manual_seed(int(rng_seed))
        pts = []
        tries = 0
        while len(pts) < n + 1 and tries < 2000:
            tries += 1
            x = MO_X[0] + torch.rand(1, generator=g, device="cpu").item() * (MO_X[1] - MO_X[0])
            y = MO_Y[0] + torch.rand(1, generator=g, device="cpu").item() * (MO_Y[1] - MO_Y[0])
            if all((x - px) ** 2 + (y - py) ** 2 >= MO_MIN_SEP ** 2 for px, py in pts):
                pts.append((x, y))
        while len(pts) < n + 1:                 # fallback: relax if rejection stalled
            pts.append((MO_X[0] + 0.05 * len(pts), MO_Y[1]))
        return pts

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
            self.agent.robot.set_pose(sapien.Pose())
            b = len(env_idx)
            ch = self._cube_half
            K = len(self.cubes)
            # Sample every env's layout first, then set each actor ONCE with a
            # (b, 3) batch. Setting poses inside a per-env loop with a (1, 3)
            # tensor broadcasts to ALL envs in the reset mask — every env ended
            # up with the LAST env's layout (one layout per batch).
            # Layout seeds come from the episode RNG, which env.reset(seed=...)
            # seeds — layouts differ per env AND per reset, reproducibly.
            cube_xy = np.zeros((b, K, 2), np.float32)
            marker_xy = np.zeros((b, 2), np.float32)
            for j in range(b):
                seed = int(self._episode_rng.randint(2 ** 31))
                pts = self._sample_positions(1, K, seed)
                cube_xy[j] = np.asarray(pts[:K], np.float32)
                marker_xy[j] = pts[-1]
            q = torch.zeros(b, 4); q[:, 0] = 1
            for i, cube in enumerate(self.cubes):
                p = torch.zeros(b, 3)
                p[:, 0] = torch.from_numpy(cube_xy[:, i, 0])
                p[:, 1] = torch.from_numpy(cube_xy[:, i, 1])
                p[:, 2] = TABLE_TOP_Z + ch
                cube.set_pose(MSPose.create_from_pq(p=p, q=q))
            mp = torch.zeros(b, 3)
            mp[:, 0] = torch.from_numpy(marker_xy[:, 0])
            mp[:, 1] = torch.from_numpy(marker_xy[:, 1])
            mp[:, 2] = TABLE_TOP_Z + 0.002
            self.target.set_pose(MSPose.create_from_pq(p=mp, q=q))

    def evaluate(self):
        return {"cube0_z": self.cubes[0].pose.p[:, 2]}

    def _get_obs_extra(self, info):
        # flatten cube positions to (num_envs, k*3) so state-obs concatenation works;
        # the data-gen reads cube poses directly off the actors, not from this vector.
        cps = torch.stack([c.pose.p for c in self.cubes], dim=1)   # (n, k, 3)
        return dict(
            cube_positions=cps.reshape(cps.shape[0], -1),
            target_pos=self.target.pose.p,
        )


if __name__ == "__main__":
    import gymnasium as gym, time
    n = 128
    env = gym.make("AM2VecPickPlace-v1", num_envs=n, sim_backend="physx_cuda",
                   obs_mode="state", control_mode="pd_joint_pos_fixed_base", render_mode=None)
    obs, _ = env.reset(seed=0)
    ee = env.unwrapped.agent.robot.links_map
    print("[ENV] created", n, "envs; cube_pos[0]=", env.unwrapped.cube.pose.p[0].tolist())
    print("[ENV] TCP link present:", "left_Fixed_Jaw" in ee)
    act = torch.zeros(n, env.action_space.shape[-1] if hasattr(env.action_space,'shape') else 18, device="cuda")
    t0 = time.time()
    for _ in range(20): env.step(act)
    torch.cuda.synchronize()
    print(f"[ENV] 20 steps ok, {20*n/(time.time()-t0):.0f} env-steps/s")
