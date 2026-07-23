#!/usr/bin/env python3
"""AlohaMini -> ManiSkill3 installer.

Usage:
    python install.py              install (idempotent — safe to re-run)
    python install.py --check      only verify the current installation
    python install.py --uninstall  remove everything install.py added

What install does:
    1. Copy the agent package to  <mani_skill>/agents/robots/aloha_mini/
       (registers robot uids "aloha_mini_1" and "aloha_mini_2")
    2. Copy the AlohaMini 1 URDF + meshes to ~/.maniskill/data/robots/aloha_mini/
    3. Register the agents in  <mani_skill>/agents/robots/__init__.py
    4. Patch the ReplicaCAD scene builder (original is backed up)

AlohaMini 2's URDF + meshes are NOT in this repo (too large) — download the
release zip separately; `--check` tells you if it is missing. See README.md.
"""

import shutil
import sys
from pathlib import Path

AM2_ZIP_URL = (
    "https://github.com/MK040412/AlohaMini/releases/download/"
    "urdf-assets-v1/aloha_mini_2_urdf.zip"
)
REGISTER_LINE = "from .aloha_mini import AlohaMini1, AlohaMini2, AlohaMiniBaseAgent"


def find_maniskill_path() -> Path:
    try:
        import mani_skill
        return Path(mani_skill.__file__).parent
    except ImportError:
        print("Error: ManiSkill not installed. Install with: pip install mani-skill")
        sys.exit(1)


def data_dir() -> Path:
    return Path.home() / ".maniskill" / "data"


def register_agents(robots_init: Path) -> None:
    """Idempotently (re)write our import line in mani_skill's robots/__init__.py.

    Old installs may carry an import of since-renamed classes, which would crash
    every `import mani_skill.agents.robots` — so stale aloha_mini lines are
    dropped, not skipped.
    """
    lines = [
        ln for ln in robots_init.read_text().splitlines()
        if "aloha_mini" not in ln
    ]
    robots_init.write_text("\n".join([REGISTER_LINE] + lines) + "\n")
    print(f"  Registered agents in {robots_init}")


def install() -> None:
    script_dir = Path(__file__).parent.resolve()
    maniskill_path = find_maniskill_path()

    print(f"ManiSkill installation: {maniskill_path}")
    print(f"ManiSkill data directory: {data_dir()}")

    # 1. Agent package (replace wholesale — the directory is owned by us, and
    #    leftover modules from an older install would shadow the new ones)
    agent_src = script_dir / "agents" / "aloha_mini"
    agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"
    print(f"\nInstalling agent files to {agent_dst}...")
    if agent_dst.exists():
        shutil.rmtree(agent_dst)
    shutil.copytree(agent_src, agent_dst, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"  Copied {sum(1 for _ in agent_dst.glob('*.py'))} agent files")

    # 2. AlohaMini 1 assets (merge-copy: never wipe the destination — it may
    #    hold locally-installed extras such as the AlohaMini 2 release assets)
    asset_src = script_dir / "assets" / "robots" / "aloha_mini"
    asset_dst = data_dir() / "robots" / "aloha_mini"
    print(f"\nInstalling AlohaMini 1 URDF + meshes to {asset_dst}...")
    shutil.copytree(asset_src, asset_dst, dirs_exist_ok=True)
    print("  Done")

    # 3. Agent registration
    robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
    print("\nRegistering agents...")
    register_agents(robots_init)

    # 4. ReplicaCAD scene builder patch (optional feature; original backed up)
    sb_src = script_dir / "scene_builder" / "replicacad" / "scene_builder.py"
    sb_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"
    if sb_src.exists() and sb_dst.exists():
        backup = sb_dst.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(sb_dst, backup)
        shutil.copy2(sb_src, sb_dst)
        print(f"\nPatched ReplicaCAD scene builder (backup: {backup.name})")

    print("\n" + "=" * 50)
    print("Installation complete!")
    print("=" * 50)
    check()


def check() -> bool:
    """Verify the install; say exactly what is missing and how to fix it."""
    ok = True
    print("\n[check] importing registered agents...")
    try:
        from mani_skill.agents.robots import AlohaMini1, AlohaMini2  # noqa: F401
        print("  OK  aloha_mini_1 + aloha_mini_2 agents import")
    except Exception as exc:
        ok = False
        print(f"  FAIL agents do not import ({exc!r}) — re-run: python install.py")

    am1 = data_dir() / "robots" / "aloha_mini" / "aloha_mini_1.urdf"
    print(f"[check] AlohaMini 1 assets: {am1}")
    print("  OK  present" if am1.exists() else "  FAIL missing — re-run: python install.py")
    ok = ok and am1.exists()

    am2 = data_dir() / "robots" / "aloha_mini_2" / "aloha_mini_2.urdf"
    print(f"[check] AlohaMini 2 assets: {am2}")
    if am2.exists():
        print("  OK  present")
    else:
        ok = False
        print("  MISSING — AlohaMini 2 assets are a separate download:")
        print(f"    wget {AM2_ZIP_URL}")
        print("    unzip aloha_mini_2_urdf.zip -d ~/.maniskill/data/robots/")

    print("\nNext steps:")
    print("  python view_urdf.py mini1 --headless   # load-check AlohaMini 1")
    print("  python view_urdf.py mini2 --headless   # load-check AlohaMini 2")
    print("  python view_urdf.py mini1              # open the GUI viewer")
    return ok


def uninstall() -> None:
    maniskill_path = find_maniskill_path()

    agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"
    if agent_dst.exists():
        shutil.rmtree(agent_dst)
        print(f"Removed {agent_dst}")

    robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
    if robots_init.exists():
        lines = [ln for ln in robots_init.read_text().splitlines() if "aloha_mini" not in ln]
        robots_init.write_text("\n".join(lines) + "\n")
        print(f"Deregistered agents in {robots_init}")

    for name in ("aloha_mini", "aloha_mini_2"):
        asset_dst = data_dir() / "robots" / name
        if asset_dst.exists():
            shutil.rmtree(asset_dst)
            print(f"Removed {asset_dst}")

    sb_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"
    backup = sb_dst.with_suffix(".py.bak")
    if backup.exists():
        shutil.copy2(backup, sb_dst)
        backup.unlink()
        print("Restored original scene_builder.py")

    print("\nUninstallation complete!")


if __name__ == "__main__":
    if "--uninstall" in sys.argv[1:]:
        uninstall()
    elif "--check" in sys.argv[1:]:
        sys.exit(0 if check() else 1)
    else:
        install()
