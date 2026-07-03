"""Find an epilogue camera pose that actually SHOWS the shaker resting on the
middle shelf (final pose from v2 run14): teleport doors+shaker to the final
state and snap stills from candidate eyes (cameras don't collide, so interior
placements are fair game)."""
import os
import numpy as np, torch
import hab_fridge2 as hf
import hab_scene
from hab_fridge2 import FridgeDemo, OUT_DIR
from hab_pick_place import actor_info, save_png, to_u8
import sapien

FINAL = np.array([-1.8811, -3.0696, 0.7114])
FINAL_Q = np.array([0.96701, 0.17393, 0.07592, -0.16994])  # run15 settled quat (w,x,y,z)

demo = FridgeDemo(norender=False)
t, obstacles, bounds, grid, sgrid, fz = demo.setup()
art = hf.find_fridge_articulation(demo.env)
art.set_qpos(torch.tensor([[0.0, 0.953]], dtype=torch.float32))
demo.set_door_drive([(5.0, 2.0, 5.0), (30.0, 8.0, 30.0)], [0.0, 0.953])
t.actor.set_pose(sapien.Pose(FINAL.tolist(), FINAL_Q.tolist()))
try:
    t.actor.set_linear_velocity(torch.zeros((1, 3)))
    t.actor.set_angular_velocity(torch.zeros((1, 3)))
except Exception:
    pass
demo.hold_base(demo.current_base_world(), lift=0.0, steps=12, record=False)
s = actor_info(t.name, t.actor)
print(f"[PROBE] shaker={np.round(s.center,3).tolist()} bottom={s.bottom:.3f}", flush=True)
c = np.asarray(s.center, np.float64)

cam = demo.be.unwrapped._human_render_cameras["closeup_camera"].camera
CANDS = {
    "A_inside_west": (c + np.array([-0.33, 0.02, 0.10]), c),
    "B_inside_west_high": (c + np.array([-0.28, 0.00, 0.22]), c),
    "C_slot_overhead": (c + np.array([-0.02, 0.00, 0.30]), c),
    "D_slot_ne": (c + np.array([0.04, 0.10, 0.28]), c),
    "E_mouth_diag": (c + np.array([0.06, 0.22, 0.42]), c),
    "F_inside_nw": (c + np.array([-0.25, 0.16, 0.18]), c),
    "G_inside_sw": (c + np.array([-0.25, -0.16, 0.18]), c),
}
outdir = OUT_DIR / "epi_cam_probe"
outdir.mkdir(exist_ok=True)
for name, (eye, tgt) in CANDS.items():
    pose = hab_scene.camera_pose_list(list(map(float, eye)), list(map(float, tgt)))
    cam.set_local_pose(sapien.Pose(pose[:3], pose[3:]))
    demo.hold_base(demo.current_base_world(), lift=0.0, steps=2, record=False)
    fr = to_u8(demo.be.render_rgb_array(camera_name="closeup_camera"))
    save_png(fr, outdir / f"{name}.png")
    print(f"[PROBE] saved {name} eye={np.round(eye,3).tolist()}", flush=True)
print("DONE", flush=True)
