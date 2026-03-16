"""Align two point clouds with ICP and save the merged result.

Used to compare a good and damaged version of an object: after alignment,
a high ICP fitness score indicates the point clouds overlap well (geometrically
similar objects), while a low score suggests structural differences exist.
"""

import tyro
import os
import numpy as np
import open3d as o3d
from dataclasses import dataclass


@dataclass
class Args:
    pcd1: str
    """Path to the first point cloud PLY file (e.g. the reference/good object)"""
    pcd2: str
    """Path to the second point cloud PLY file (e.g. the damaged object)"""
    output_dir: str = "point_clouds"
    """Directory to save the merged combined.ply"""
    max_correspondence_distance: float = 0.01
    """ICP correspondence distance threshold in metres"""


def compute_icp(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, args: Args):
    for pcd in [pcd1, pcd2]:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )

    reg = o3d.pipelines.registration.registration_colored_icp(
        pcd2,
        pcd1,
        max_correspondence_distance=args.max_correspondence_distance,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=5000
        ),
    )

    np.set_printoptions(precision=6, suppress=True)
    print(f"\n  ICP result:")
    print(f"    Fitness:  {reg.fitness:.6f}  "
          f"(fraction of inlier correspondences — higher = better overlap)")
    print(f"    RMSE:     {reg.inlier_rmse:.6f}  "
          f"(root-mean-square error of inlier pairs — lower = tighter alignment)")
    print(f"    Transformation:\n{reg.transformation}")

    if reg.fitness < 0.5:
        print("\n  Warning: low fitness (<0.5). The two point clouds may differ significantly,")
        print("  or the initial poses are misaligned. Consider running load_pipeline.py first")
        print("  to extract and centre the clouds before comparison.")

    pcd2.transform(reg.transformation)
    merged = pcd1 + pcd2
    out_path = os.path.join(args.output_dir, "combined.ply")
    os.makedirs(args.output_dir, exist_ok=True)
    o3d.io.write_point_cloud(out_path, merged.uniform_down_sample(10))
    print(f"\n  Merged point cloud saved to: {out_path}")
    return reg.transformation.copy()


def main(args: Args):
    print(f"\nLoading point clouds:")
    print(f"  pcd1 (reference): {args.pcd1}")
    print(f"  pcd2 (query):     {args.pcd2}")
    pcd1 = o3d.io.read_point_cloud(args.pcd1)
    pcd2 = o3d.io.read_point_cloud(args.pcd2)
    print(f"  pcd1: {len(pcd1.points)} points")
    print(f"  pcd2: {len(pcd2.points)} points")
    compute_icp(pcd1, pcd2, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
