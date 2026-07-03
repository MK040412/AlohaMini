"""Convert the AlohaMini (5-DOF Std) URDF to USD for InternDataEngine / Isaac Sim,
and dump the articulation joint order + link prim paths needed for aloha_mini.yaml.
"""
import os
from isaacsim import SimulationApp
kit = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import UsdPhysics

URDF = os.environ.get("SRC_URDF", "/home/perelman/.maniskill/data/robots/aloha_mini/maniskill_so100_version.urdf")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad/InternDataEngine/workflows/simbox/assets/aloha_mini")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_USD = os.path.join(OUT_DIR, "robot.usd")

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False      # keep Fixed_Jaw / finger tips as prims
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = True                 # fixed-base robot (no mobile base drift)
import_config.distance_scale = 1.0
import_config.make_default_prim = True
import_config.self_collision = False

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=URDF, import_config=import_config,
    get_articulation_root=True,
)
print("IMPORT status:", status, "prim_path:", prim_path, flush=True)

stage = omni.usd.get_context().get_stage()
# save USD
stage.Export(OUT_USD)
print("SAVED USD:", OUT_USD, flush=True)

# dump articulation joints (order = joint index in the USD articulation)
print("=== ARTICULATION JOINTS (index : name : type) ===", flush=True)
i = 0
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        tj = prim.GetTypeName()
        # only movable joints (revolute/prismatic) get articulation DOF indices
        if tj in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            print(f"DOF {i:2d} : {prim.GetName():28s} : {tj}", flush=True)
            i += 1
print("=== END JOINTS ===", flush=True)
# dump key link prim paths (palm/fingers/tips)
print("=== KEY LINK PRIMS ===", flush=True)
for prim in stage.Traverse():
    n = prim.GetName()
    if any(k in n for k in ("Fixed_Jaw", "finger1", "finger2", "finger1_tip", "finger2_tip", "Base", "base_link")):
        print("PRIM:", prim.GetPath(), flush=True)
print("=== END PRIMS ===", flush=True)
kit.close()
