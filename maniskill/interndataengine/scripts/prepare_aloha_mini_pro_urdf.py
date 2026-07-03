#!/usr/bin/env python3
"""Prepare AlohaMini Pro v3 URDF assets for CuRobo.

CuRobo expects axis-aligned joint axes and finite limits for non-fixed joints. The
source ManiSkill URDF has a continuous root yaw joint and a few near-principal
axes, so this script copies the URDF/mesh assets into InternDataEngine's CuRobo
content tree and normalizes those fields.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_SOURCE_DIR = Path.home() / ".maniskill/data/robots/aloha_mini"
DEFAULT_OUTPUT_REL = Path(
    "workflows/simbox/curobo/src/curobo/content/assets/robot/aloha_mini_pro"
)


def _snap_axis(axis_text: str) -> str:
    vals = [float(v) for v in axis_text.split()]
    if len(vals) != 3:
        return axis_text
    dominant = max(range(3), key=lambda i: abs(vals[i]))
    snapped = [0.0, 0.0, 0.0]
    snapped[dominant] = 1.0 if vals[dominant] >= 0 else -1.0
    return " ".join(str(int(v)) if v in (-1.0, 0.0, 1.0) else str(v) for v in snapped)


def prepare_urdf(source_dir: Path, output_dir: Path) -> Path:
    source_urdf = source_dir / "aloha_mini_pro_v3.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    output_dir.mkdir(parents=True, exist_ok=True)
    for asset_dir in ("meshes", "clamp_meshes"):
        shutil.copytree(source_dir / asset_dir, output_dir / asset_dir, dirs_exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.get("type") == "continuous":
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.set("lower", "-3.14159")
            limit.set("upper", "3.14159")
            limit.set("effort", limit.get("effort") or "500")
            limit.set("velocity", limit.get("velocity") or "1.0")

        axis = joint.find("axis")
        if axis is not None and axis.get("xyz"):
            axis.set("xyz", _snap_axis(axis.get("xyz")))

        limit = joint.find("limit")
        if limit is not None:
            if float(limit.get("effort", "0") or 0.0) == 0.0:
                limit.set("effort", "100")
            if float(limit.get("velocity", "0") or 0.0) == 0.0:
                limit.set("velocity", "3.14159")

    ET.indent(tree, space="  ")
    output_urdf = output_dir / "aloha_mini_pro.urdf"
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.repo_root / DEFAULT_OUTPUT_REL
    output_urdf = prepare_urdf(args.source_dir, output_dir)
    print(output_urdf)


if __name__ == "__main__":
    main()
