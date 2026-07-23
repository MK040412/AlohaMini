"""Single source of truth for the AlohaMini robot lineup.

Every user-facing tool (view_urdf.py, demo.py) resolves robot keys and agent
registration through this module, so adding/renaming a robot is a one-file
change here plus the agent class itself.
"""

# short CLI key -> ManiSkill agent uid
ROBOTS = {
    "mini1": "aloha_mini_1",   # SO100 arms + parallel gripper (repo assets)
    "mini2": "aloha_mini_2",   # official AM2 Pro arms + parallel gripper (Release zip)
}


def register_agents() -> None:
    """Import the agent package so both uids register with ManiSkill.

    Prefers the copy install.py placed inside mani_skill (works from any cwd);
    falls back to the repo-local package for checkouts that skipped install.py.
    """
    try:
        import mani_skill.agents.robots.aloha_mini  # noqa: F401
    except ImportError:
        import agents.aloha_mini  # noqa: F401


def ensure_assets(uid: str) -> bool:
    """Check the robot's URDF is installed; if not, print how to get it.

    Keeps user tools from dying in a raw loader traceback when the AlohaMini 2
    release zip (a separate download) was skipped.
    """
    from pathlib import Path

    urdf = {
        "aloha_mini_1": Path.home() / ".maniskill/data/robots/aloha_mini/aloha_mini_1.urdf",
        "aloha_mini_2": Path.home() / ".maniskill/data/robots/aloha_mini_2/aloha_mini_2.urdf",
    }[uid]
    if urdf.exists():
        return True
    print(f"[ROBOTS] {uid} assets are not installed ({urdf}).")
    if uid == "aloha_mini_1":
        print("[ROBOTS] install them with:\n    python install.py")
    else:
        print("[ROBOTS] download them once with:")
        print("    wget https://github.com/MK040412/AlohaMini/releases/download/"
              "urdf-assets-v1/aloha_mini_2_urdf.zip")
        print("    unzip aloha_mini_2_urdf.zip -d ~/.maniskill/data/robots/")
    return False
