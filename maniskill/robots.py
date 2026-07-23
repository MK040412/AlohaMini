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
