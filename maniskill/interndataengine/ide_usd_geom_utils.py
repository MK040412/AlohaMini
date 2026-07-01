# pylint: skip-file
# flake8: noqa
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom


def to_list(data: Sequence):
    """Convert sequence-like data to a list, returning an empty list for None."""
    if data is None:
        return []
    return list(data)


def recursive_parse_new(prim: Usd.Prim) -> Tuple[list, list, list]:
    """Recursively collect mesh vertices and face indices from a prim subtree in world coordinates."""
    points_total: List = []
    faceVertexCounts_total: List[int] = []
    faceVertexIndices_total: List[int] = []

    if prim.IsA(UsdGeom.Mesh):
        prim_imageable = UsdGeom.Imageable(prim)
        xform_world_transform = np.array(prim_imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        points = prim.GetAttribute("points").Get()
        faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
        faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
        faceVertexCounts = to_list(faceVertexCounts)
        faceVertexIndices = to_list(faceVertexIndices)
        points = to_list(points)

        points = np.array(points)  # Nx3
        ones = np.ones((points.shape[0], 1))  # Nx1
        points_h = np.hstack([points, ones])  # Nx4
        points_transformed_h = np.dot(points_h, xform_world_transform)  # Nx4
        points_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3][:, np.newaxis]  # Nx3
        points = points_transformed.tolist()

        base_num = len(points_total)
        faceVertexIndices = np.array(faceVertexIndices)
        faceVertexIndices_total.extend((base_num + faceVertexIndices).tolist())
        faceVertexCounts_total.extend(faceVertexCounts)
        points_total.extend(points)

    children = prim.GetChildren()
    for child in children:
        child_points, child_faceVertexCounts, child_faceVertexIndices = recursive_parse_new(child)
        base_num = len(points_total)
        child_faceVertexIndices = np.array(child_faceVertexIndices)
        faceVertexIndices_total.extend((base_num + child_faceVertexIndices).tolist())
        faceVertexCounts_total.extend(child_faceVertexCounts)
        points_total.extend(child_points)

    return (
        points_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
    )


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    """Build an Open3D triangle mesh from USD mesh point and face data."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def sample_points_from_mesh(mesh, num_points: int = 1000):
    """Uniformly sample points from an Open3D mesh."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def sample_points_from_prim(prim: Usd.Prim, num_points: int = 1000) -> np.ndarray:
    """Sample points from a prim subtree by converting meshes to a sampled point cloud."""
    points, faceVertexCounts, faceVertexIndices = recursive_parse_new(prim)
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    pcd = sample_points_from_mesh(mesh, num_points)
    return np.asarray(pcd.points)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """Compute an axis-aligned world-space bounding box for a prim subtree.

    The fast path (Imageable.ComputeWorldBound on the subtree root) returns an
    EMPTY range for robots whose mesh subtrees are loaded as USD *instances* in
    the live stage (URDF-imported robots): ComputeWorldBound on the instance root
    doesn't see geometry that lives in the instance prototype, so min=+FLT_MAX.
    That empty box made the A_on_B_region_sampler place the robot at Z=-inf (it
    fell to -inf, every grasp target became FLT_MAX, and no CuRobo plan could
    converge). Fall back to a manual union over every descendant mesh, traversing
    instance proxies, which is robust to instancing and to render/default purpose.
    """
    time = Usd.TimeCode.Default()
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]

    def _extent_union(root):
        # Union world-space corners of every descendant mesh's local extent/points.
        # Reads extent directly so it ignores visibility (Isaac hides robot visual
        # meshes) and traverses instance proxies (URDF meshes are instanceable).
        gmin = np.full(3, np.inf)
        gmax = np.full(3, -np.inf)
        n = 0
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if not p.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(p)
            ext = mesh.GetExtentAttr().Get()
            if ext:
                lmin = np.array(ext[0], dtype=float)
                lmax = np.array(ext[1], dtype=float)
            else:
                pts = mesh.GetPointsAttr().Get()
                if not pts:
                    continue
                arr = np.array(pts, dtype=float)
                lmin, lmax = arr.min(0), arr.max(0)
            xf = np.array(UsdGeom.Imageable(p).ComputeLocalToWorldTransform(time), dtype=float)
            corners = np.array(
                [[x, y, z, 1.0] for x in (lmin[0], lmax[0]) for y in (lmin[1], lmax[1]) for z in (lmin[2], lmax[2])]
            )
            world_h = corners @ xf  # USD row-major: world_row = local_row @ M
            world = world_h[:, :3] / world_h[:, 3:4]
            gmin = np.minimum(gmin, world.min(0))
            gmax = np.maximum(gmax, world.max(0))
            n += 1
        return gmin, gmax, n

    bound_range = UsdGeom.Imageable(prim).ComputeWorldBound(time, *purposes).ComputeAlignedBox()
    if not bound_range.IsEmpty() and np.isfinite(bound_range.GetMin()[0]):
        return bound_range

    # A physics-articulation robot's Robot.prim is the ArticulationRoot *joint* prim,
    # whose SIBLINGS (not children) hold the link meshes — so both the fast path and a
    # subtree union over `prim` find zero meshes and the box stays empty (=> Z=-inf, the
    # robot fell to -inf, all grasp targets FLT_MAX, no CuRobo plan converged). Walk up
    # to the ancestor that actually contains geometry.
    search = prim
    for _ in range(5):
        gmin, gmax, n_mesh = _extent_union(search)
        if n_mesh > 0 and np.all(np.isfinite(gmin)):
            return Gf.Range3d(Gf.Vec3d(*gmin.tolist()), Gf.Vec3d(*gmax.tolist()))
        parent = search.GetParent()
        if not parent or not parent.IsValid() or parent.IsPseudoRoot():
            break
        search = parent
    return bound_range


if __name__ == "__main__":
    # Simple manual test/debug hook kept from original peizhou_prim.py
    dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    models_folder = os.path.join(dir_path, "minidump_usd/0001/dest_usd/models_z/models/")
    subfolders = [f.name for f in os.scandir(models_folder) if f.is_dir()]
    for subfolder in subfolders:
        print(subfolder)
        usd_path = os.path.join(models_folder, subfolder, "instance.usd")
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetPrimAtPath("/Root/Instance")
        points, faceVertexCounts, faceVertexIndices = recursive_parse_new(prim)
        mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
        o3d.visualization.draw_geometries([mesh])
        pcd = sample_points_from_mesh(mesh, num_points=10000)
        o3d.visualization.draw_geometries([pcd])
