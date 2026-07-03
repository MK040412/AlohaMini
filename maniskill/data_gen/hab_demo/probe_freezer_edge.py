"""Where is the freezer-floor (divider) FRONT edge on the y=-3.05 lane?"""
import numpy as np, torch
import hab_fridge2 as hf
from hab_fridge2 import FridgeDemo
from hab_pick_place import actor_info
demo = FridgeDemo(norender=True)
t, obstacles, bounds, grid, sgrid, fz = demo.setup()
art = hf.find_fridge_articulation(demo.env)
art.set_qpos(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
demo.set_door_drive([(30.0,8.0,30.0),(5.0,2.0,5.0)],[1.0,0.0])
demo.hold_base(demo.current_base_world(), lift=0.0, steps=8, record=False)
for x in (-1.88,-1.90,-1.92,-1.94,-1.96,-1.98,-2.00):
    demo.set_actor_center(t.actor, t.name, np.array([x,-3.05,1.10]))
    demo.hold_base(demo.current_base_world(), lift=0.0, steps=8, record=False)
    s = actor_info(t.name, t.actor)
    print(f"EDGE x={x:+.2f} settled={np.round(s.center,3).tolist()} bottom={s.bottom:.3f} "
          f"on_freezer={abs(s.bottom-1.0227)<0.02}", flush=True)
print("DONE", flush=True)
