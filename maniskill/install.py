#!/usr/bin/env python3
"""
AlohaMini ManiSkill Integration Installer

This script installs the AlohaMini robot agent and assets into your ManiSkill installation.

Usage:
    python install.py           # Install
    python install.py --uninstall   # Uninstall
    python install.py --check       # Check installation status

This will:
1. Copy the AlohaMini agent files to mani_skill/agents/robots/aloha_mini/
2. Copy the URDF and mesh files to ~/.maniskill/data/robots/aloha_mini/
3. Update the ReplicaCAD scene builder to support AlohaMini
"""

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def get_venv_info():
    """Get information about the current virtual environment."""
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        venv_path = Path(venv_path)
        # Detect if this is a uv venv
        is_uv = venv_path.name == '.venv' or (venv_path / 'pyvenv.cfg').exists()
        return {
            'active': True,
            'path': venv_path,
            'type': 'uv' if is_uv else 'standard',
            'site_packages': Path(sysconfig.get_path('purelib'))
        }
    return {
        'active': False,
        'path': None,
        'type': None,
        'site_packages': Path(sysconfig.get_path('purelib'))
    }


def find_maniskill_in_venv(venv_path: Path):
    """
    Explicitly search for mani_skill in a virtual environment's site-packages.

    Args:
        venv_path: Path to the virtual environment (e.g., .venv)

    Returns:
        Path to mani_skill package or None
    """
    # Try to find any python version in lib/
    lib_path = venv_path / 'lib'
    if lib_path.exists():
        for python_dir in lib_path.iterdir():
            if python_dir.name.startswith('python'):
                possible_path = python_dir / 'site-packages' / 'mani_skill'
                if possible_path.exists():
                    return possible_path

    # Windows style
    windows_path = venv_path / 'Lib' / 'site-packages' / 'mani_skill'
    if windows_path.exists():
        return windows_path

    return None


def find_maniskill_path():
    """
    Find the ManiSkill installation path.

    Works with both standard pip and uv virtual environments.
    """
    venv_info = get_venv_info()

    # Method 1: Try to import mani_skill directly (works if venv is activated)
    try:
        import mani_skill
        maniskill_path = Path(mani_skill.__file__).parent
        print(f"Found ManiSkill at: {maniskill_path}")
        return maniskill_path
    except ImportError:
        pass

    # Method 2: If venv is active, search in its site-packages
    if venv_info['active']:
        maniskill_path = find_maniskill_in_venv(venv_info['path'])
        if maniskill_path:
            print(f"Found ManiSkill at: {maniskill_path}")
            return maniskill_path

    # Method 3: Check for .venv in current directory
    cwd_venv = Path.cwd() / '.venv'
    if cwd_venv.exists():
        maniskill_path = find_maniskill_in_venv(cwd_venv)
        if maniskill_path:
            print(f"Found ManiSkill in .venv at: {maniskill_path}")
            return maniskill_path

    # If all methods fail, provide helpful error message
    error_msg = """
============================================================
ERROR: ManiSkill not found!
============================================================

IMPORTANT: The package name is 'mani-skill' (with HYPHEN), NOT 'mani_skill'.

To install with uv:
    uv pip install mani-skill

To install with pip:
    pip install mani-skill

"""

    if venv_info['active']:
        error_msg += f"""
Current virtual environment: {venv_info['path']}
Environment type: {venv_info['type']}
Site-packages location: {venv_info['site_packages']}
"""
        # Check if user accidentally installed wrong package
        site_packages = venv_info['site_packages']
        if site_packages.exists():
            installed = list(site_packages.glob('mani*'))
            if installed:
                error_msg += f"\nFound packages matching 'mani*': {[p.name for p in installed]}"
    else:
        error_msg += """
WARNING: No virtual environment detected!

It's strongly recommended to use a virtual environment:
    uv venv
    source .venv/bin/activate  # Linux/Mac
    .venv\\Scripts\\activate   # Windows

Then install ManiSkill:
    uv pip install mani-skill
"""

    print(error_msg, file=sys.stderr)
    sys.exit(1)


def get_maniskill_data_dir():
    """Get the ManiSkill data directory."""
    # Check for custom data directory
    custom_dir = os.environ.get('MS_ASSET_DIR')
    if custom_dir:
        return Path(custom_dir)
    return Path.home() / ".maniskill" / "data"


def check_write_permissions(path: Path) -> bool:
    """Check if we have write permissions to a path."""
    try:
        if path.exists():
            return os.access(path, os.W_OK)
        else:
            # Check parent directory
            parent = path.parent
            while not parent.exists():
                parent = parent.parent
            return os.access(parent, os.W_OK)
    except Exception:
        return False


def install():
    """Install AlohaMini into ManiSkill."""
    script_dir = Path(__file__).parent.resolve()

    print("=" * 60)
    print("AlohaMini ManiSkill Integration Installer")
    print("=" * 60)

    # Get virtual environment info
    venv_info = get_venv_info()
    if venv_info['active']:
        print(f"\nVirtual environment: {venv_info['path']}")
        print(f"Environment type: {venv_info['type']}")
    else:
        print("\nWARNING: No virtual environment detected!")
        print("It's recommended to use a virtual environment.")
        print("Continuing anyway...")

    # Find ManiSkill paths
    maniskill_path = find_maniskill_path()
    data_dir = get_maniskill_data_dir()

    print(f"\nManiSkill installation: {maniskill_path}")
    print(f"ManiSkill data directory: {data_dir}")

    # Check write permissions
    if not check_write_permissions(maniskill_path):
        print(f"\nERROR: No write permission to {maniskill_path}")
        print("Try running with appropriate permissions or check your virtual environment.")
        sys.exit(1)

    # 1. Install agent files
    agent_src = script_dir / "agents" / "aloha_mini"
    agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"

    if agent_src.exists():
        print(f"\n[1/4] Installing agent files to {agent_dst}...")
        agent_dst.mkdir(parents=True, exist_ok=True)
        for f in agent_src.glob("*.py"):
            shutil.copy2(f, agent_dst / f.name)
            print(f"  Copied {f.name}")
    else:
        print(f"\nWARNING: Agent source directory not found: {agent_src}")

    # 2. Install URDF and mesh files
    asset_src = script_dir / "assets" / "robots" / "aloha_mini"
    asset_dst = data_dir / "robots" / "aloha_mini"

    if asset_src.exists():
        print(f"\n[2/4] Installing URDF and mesh files to {asset_dst}...")
        if asset_dst.exists():
            shutil.rmtree(asset_dst)
        shutil.copytree(asset_src, asset_dst, symlinks=True)
        print(f"  Copied all files from {asset_src}")
    else:
        print(f"\nWARNING: Asset source directory not found: {asset_src}")

    # 3. Update scene builder (optional - backup first)
    scene_builder_src = script_dir / "scene_builder" / "replicacad" / "scene_builder.py"
    scene_builder_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"

    print(f"\n[3/4] Updating ReplicaCAD scene builder...")
    if scene_builder_src.exists() and scene_builder_dst.exists():
        # Backup original
        backup = scene_builder_dst.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(scene_builder_dst, backup)
            print(f"  Backed up original to {backup.name}")
        shutil.copy2(scene_builder_src, scene_builder_dst)
        print(f"  Updated scene_builder.py")
    elif scene_builder_src.exists():
        print(f"  WARNING: Target scene_builder.py not found at {scene_builder_dst}")
        print(f"  This may indicate ManiSkill version mismatch. Skipping scene builder update.")
    else:
        print(f"  Skipped (source not found)")

    # 4. Register the agent in __init__.py
    robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
    print(f"\n[4/4] Registering AlohaMini agent...")
    if robots_init.exists():
        content = robots_init.read_text()
        if "aloha_mini" not in content:
            # Add import for AlohaMini agents
            new_import = 'from .aloha_mini import AlohaMiniSO100V2, AlohaMiniBaseAgent'

            # Find the best place to insert
            if "# Robot imports" in content:
                content = content.replace("# Robot imports", f"# Robot imports\n{new_import}")
            else:
                # Add at the beginning after any existing imports
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('from .') or line.startswith('import '):
                        insert_idx = i + 1
                lines.insert(insert_idx, new_import)
                content = '\n'.join(lines)

            robots_init.write_text(content)
            print(f"  Added import to {robots_init}")
        else:
            print(f"  AlohaMini already registered")
    else:
        print(f"  WARNING: robots __init__.py not found at {robots_init}")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nYou can now use the AlohaMini robot in ManiSkill:")
    print('  robot_uids="aloha_mini_so100_v2"')
    print("\nVerify installation:")
    print("  python install.py --check")
    print("\nTest import:")
    print('  python -c "from mani_skill.agents.robots.aloha_mini import AlohaMiniSO100V2; print(\'Success!\')"')
    print("\nTeleoperation demo:")
    print("  cd teleop && python demo_teleop.py --render")


def check_installation():
    """Check if AlohaMini is properly installed."""
    print("=" * 60)
    print("AlohaMini Installation Check")
    print("=" * 60)

    all_ok = True

    venv_info = get_venv_info()
    print(f"\n[1] Virtual Environment")
    print(f"    Active: {'Yes' if venv_info['active'] else 'No'}")
    if venv_info['active']:
        print(f"    Path: {venv_info['path']}")
        print(f"    Type: {venv_info['type']}")

    # Check ManiSkill
    print(f"\n[2] ManiSkill Package")
    try:
        import mani_skill
        maniskill_path = Path(mani_skill.__file__).parent
        print(f"    Status: INSTALLED")
        print(f"    Path: {maniskill_path}")
        print(f"    Version: {getattr(mani_skill, '__version__', 'unknown')}")
    except ImportError:
        print(f"    Status: NOT FOUND")
        print(f"    Install with: uv pip install mani-skill")
        all_ok = False
        maniskill_path = None

    if maniskill_path:
        # Check agent files
        print(f"\n[3] Agent Files")
        agent_path = maniskill_path / "agents" / "robots" / "aloha_mini"
        if agent_path.exists():
            print(f"    Status: INSTALLED")
            print(f"    Path: {agent_path}")
            files = list(agent_path.glob("*.py"))
            print(f"    Files: {[f.name for f in files]}")
        else:
            print(f"    Status: NOT INSTALLED")
            all_ok = False

        # Check registration
        print(f"\n[4] Agent Registration")
        robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
        if robots_init.exists():
            content = robots_init.read_text()
            if "aloha_mini" in content:
                print(f"    Status: REGISTERED")
            else:
                print(f"    Status: NOT REGISTERED")
                all_ok = False
        else:
            print(f"    Status: __init__.py not found")
            all_ok = False

    # Check data files
    print(f"\n[5] Asset Files")
    data_dir = get_maniskill_data_dir()
    asset_path = data_dir / "robots" / "aloha_mini"
    if asset_path.exists():
        print(f"    Status: INSTALLED")
        print(f"    Path: {asset_path}")
    else:
        print(f"    Status: NOT INSTALLED")
        all_ok = False

    # Try to import
    print(f"\n[6] Import Test")
    try:
        from mani_skill.agents.robots.aloha_mini import AlohaMiniSO100V2
        print(f"    Status: SUCCESS")
        print(f"    AlohaMiniSO100V2 imported successfully")
    except ImportError as e:
        print(f"    Status: FAILED")
        print(f"    Error: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed! AlohaMini is ready to use.")
    else:
        print("Some checks failed. Run 'python install.py' to install.")
    print("=" * 60)

    return all_ok


def uninstall():
    """Uninstall AlohaMini from ManiSkill."""
    print("=" * 60)
    print("AlohaMini Uninstaller")
    print("=" * 60)

    try:
        maniskill_path = find_maniskill_path()
    except SystemExit:
        print("\nManiSkill not found. Continuing to clean up data files...")
        maniskill_path = None

    data_dir = get_maniskill_data_dir()

    if maniskill_path:
        # Remove agent files
        agent_dst = maniskill_path / "agents" / "robots" / "aloha_mini"
        if agent_dst.exists():
            shutil.rmtree(agent_dst)
            print(f"\nRemoved agent files: {agent_dst}")

        # Restore scene builder backup
        scene_builder_dst = maniskill_path / "utils" / "scene_builder" / "replicacad" / "scene_builder.py"
        backup = scene_builder_dst.with_suffix(".py.bak")
        if backup.exists():
            shutil.copy2(backup, scene_builder_dst)
            backup.unlink()
            print(f"Restored original scene_builder.py")

        # Remove registration from __init__.py
        robots_init = maniskill_path / "agents" / "robots" / "__init__.py"
        if robots_init.exists():
            content = robots_init.read_text()
            if "aloha_mini" in content:
                lines = content.split('\n')
                lines = [l for l in lines if 'aloha_mini' not in l.lower()]
                robots_init.write_text('\n'.join(lines))
                print("Removed registration from robots/__init__.py")

    # Remove asset files
    asset_dst = data_dir / "robots" / "aloha_mini"
    if asset_dst.exists():
        shutil.rmtree(asset_dst)
        print(f"Removed asset files: {asset_dst}")

    print("\n" + "=" * 60)
    print("Uninstallation complete!")
    print("=" * 60)


def print_help():
    """Print help message."""
    print(__doc__)
    print("""
Options:
    (no args)       Install AlohaMini into ManiSkill
    --uninstall     Remove AlohaMini from ManiSkill
    --check         Check installation status
    --help, -h      Show this help message

Example workflow with uv:
    # 1. Create and activate virtual environment
    uv venv
    source .venv/bin/activate

    # 2. Install ManiSkill (note: use hyphen, not underscore!)
    uv pip install mani-skill pygame websockets Pillow

    # 3. Install AlohaMini
    python install.py

    # 4. Verify installation
    python install.py --check

    # 5. Run demo
    cd teleop && python demo_teleop.py --render
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--uninstall":
            uninstall()
        elif arg == "--check":
            check_installation()
        elif arg in ("--help", "-h"):
            print_help()
        else:
            print(f"Unknown option: {arg}")
            print("Usage: python install.py [--uninstall|--check|--help]")
            sys.exit(1)
    else:
        install()
