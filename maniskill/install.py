#!/usr/bin/env python3
"""
AlohaMini ManiSkill Integration Installer

This script installs the AlohaMini robot agent and assets into your ManiSkill installation.

Usage:
    python install.py

This will:
1. Copy the AlohaMini agent files to mani_skill/agents/robots/aloha_mini/
2. Copy the URDF and mesh files to ~/.maniskill/data/robots/aloha_mini/
3. Update the ReplicaCAD scene builder to support AlohaMini
"""

import os
import shutil
import sys
from pathlib import Path


def find_maniskill_path():
    """Find the ManiSkill installation path."""
    try:
        import mani_skill
        return Path(mani_skill.__file__).parent
    except ImportError:
        print("Error: ManiSkill not installed. Install with: pip install mani-skill")
        sys.exit(1)


def get_maniskill_data_dir():
    """Get the ManiSkill data directory."""
    return Path.home() / ".maniskill" / "data"


def install():
    """Install AlohaMini into ManiSkill."""
    script_dir = Path(__file__).parent.resolve()

    # Find ManiSkill paths
    maniskill_path = find_maniskill_path()
    data_dir = get_maniskill_data_dir()

    print(f"ManiSkill installation: {maniskill_path}")
    print(f"ManiSkill data directory: {data_dir}")

    # 1. Install agent files
    agent_src = script_dir / "agents" / "aloha_mini"
    agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"

    if agent_src.exists():
        print(f"\nInstalling agent files to {agent_dst}...")
        agent_dst.mkdir(parents=True, exist_ok=True)
        for f in agent_src.glob("*.py"):
            shutil.copy2(f, agent_dst / f.name)
            print(f"  Copied {f.name}")
    else:
        print(f"Warning: Agent source directory not found: {agent_src}")

    # 2. Install URDF and mesh files
    asset_src = script_dir / "assets" / "robots" / "aloha_mini"
    asset_dst = data_dir / "robots" / "aloha_mini"

    if asset_src.exists():
        print(f"\nInstalling URDF and mesh files to {asset_dst}...")
        if asset_dst.exists():
            shutil.rmtree(asset_dst)
        shutil.copytree(asset_src, asset_dst)
        print(f"  Copied all files from {asset_src}")
    else:
        print(f"Warning: Asset source directory not found: {asset_src}")

    # 3. Update scene builder (optional - backup first)
    scene_builder_src = script_dir / "scene_builder" / "replicacad" / "scene_builder.py"
    scene_builder_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"

    if scene_builder_src.exists() and scene_builder_dst.exists():
        print(f"\nUpdating ReplicaCAD scene builder...")
        # Backup original
        backup = scene_builder_dst.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(scene_builder_dst, backup)
            print(f"  Backed up original to {backup.name}")
        shutil.copy2(scene_builder_src, scene_builder_dst)
        print(f"  Updated scene_builder.py")

    # 4. Register the agent in __init__.py
    robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
    if robots_init.exists():
        content = robots_init.read_text()
        if "aloha_mini" not in content:
            print(f"\nRegistering AlohaMini agent...")
            # Add import
            new_import = 'from .aloha_mini import AlohaMini, AlohaMiniFixed, AlohaMiniVirtual, ALOHA_MINI_BASE_COLLISION_BIT, ALOHA_MINI_WHEELS_COLLISION_BIT'
            if "# Robot imports" in content:
                content = content.replace("# Robot imports", f"# Robot imports\n{new_import}")
            else:
                content = f"{new_import}\n{content}"
            robots_init.write_text(content)
            print(f"  Added import to {robots_init}")

    print("\n" + "="*50)
    print("Installation complete!")
    print("="*50)
    print("\nYou can now use the AlohaMini robot in ManiSkill:")
    print('  robot_uids="aloha_mini"          # With wheels')
    print('  robot_uids="aloha_mini_fixed"    # Fixed base')
    print('  robot_uids="aloha_mini_virtual"  # Virtual mobile base')
    print("\nExample:")
    print("  python maniskill/examples/demo_virtual_base.py --render")


def uninstall():
    """Uninstall AlohaMini from ManiSkill."""
    maniskill_path = find_maniskill_path()
    data_dir = get_maniskill_data_dir()

    # Remove agent files
    agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"
    if agent_dst.exists():
        shutil.rmtree(agent_dst)
        print(f"Removed {agent_dst}")

    # Remove asset files
    asset_dst = data_dir / "robots" / "aloha_mini"
    if asset_dst.exists():
        shutil.rmtree(asset_dst)
        print(f"Removed {asset_dst}")

    # Restore scene builder backup
    scene_builder_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"
    backup = scene_builder_dst.with_suffix(".py.bak")
    if backup.exists():
        shutil.copy2(backup, scene_builder_dst)
        backup.unlink()
        print(f"Restored original scene_builder.py")

    print("\nUninstallation complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        uninstall()
    else:
        install()
