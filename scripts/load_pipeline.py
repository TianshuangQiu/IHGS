"""Extract point clouds from trained IHGS models and compare them with ICP.

Loads two trained pipelines, extracts Gaussian point clouds, and aligns them
using colored ICP. Multiple rotation initializations are tried to avoid local
minima caused by rotational symmetry in the objects.

The best alignment is selected by comparing per-pixel render differences
after transforming pipeline2's cameras into pipeline1's coordinate frame.
Results (frames, depths, accumulations) are saved for downstream analysis
with scripts/analyze_from_jupyter.py.
"""

from inhand.ih_config import ihgs_full_merged
from inhand.ihgs import IHGSModelConfig, IHGSModel
from inhand.ih_pipeline import IHGSPipelineConfig, IHGSPipeline
from inhand.ih_datamanager import IHDataManagerConfig, IHDataManager
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from glob import glob
from tqdm import tqdm
import open3d as o3d
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
import numpy as np
from nerfstudio.cameras.cameras import Cameras
import torch
import cv2
import os
import tyro
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation as R
from skimage.metrics import structural_similarity as ssim


@dataclass
class Args:
    pipeline_name1: str
    """Data path substring identifying the first (reference/good) pipeline"""
    pipeline_name2: str
    """Data path substring identifying the second (query/damaged) pipeline"""
    folder_name: str
    """Output subfolder under defect_detection/ for this comparison run"""
    outputs_dir: str = "outputs/combined/ihgs-full-merged"
    """Directory to search for trained config files"""
    output_dir: str = "defect_detection"
    """Root directory for all output files"""
    n_frames: int = 100
    """Number of frames to render and compare"""
    n_gaussians: int = 10000
    """Top-N Gaussians (by opacity) to include in the extracted point cloud"""
    init_transform: str = ""
    """Path to a txt file containing a 4x4 initial transform matrix (row-major),
    or a comma-separated 16-element string. Defaults to identity if not provided.
    Use this when you have a good prior on the relative pose between the two objects."""
    rotations: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0, 0, 0), (180, 0, 0), (0, 180, 0), (0, 0, 180)]
    )
    """Euler angles (degrees, XYZ) for rotation initializations to try before ICP.
    Multiple initializations help escape local minima from rotationally symmetric objects.
    Each is composed with init_transform before being passed to ICP."""
    max_correspondence_distance: float = 0.001
    """ICP inlier correspondence distance threshold in metres"""
    crop_height: int = 1080
    """Rendered image height after cropping (pixels)"""
    crop_width: int = 2048
    """Rendered image width after cropping (pixels)"""


def load_init_transform(args: Args) -> np.ndarray:
    """Load the 4x4 initial transform from file, inline string, or default to identity."""
    if not args.init_transform:
        return np.eye(4)
    path = Path(args.init_transform)
    if path.exists():
        tsfm = np.loadtxt(str(path))
        print(f"  Loaded initial transform from: {path}")
    else:
        vals = [float(v) for v in args.init_transform.replace(",", " ").split()]
        tsfm = np.array(vals).reshape(4, 4)
        print(f"  Parsed initial transform from inline string")
    return tsfm


def extract_pointcloud(model, n_gaussians: int) -> o3d.geometry.PointCloud:
    means = model.gauss_params["means"].detach().cpu().numpy()
    colors = SH2RGB(model.gauss_params["features_dc"]).detach().cpu().numpy()
    opacities = model.gauss_params["opacities"].detach().cpu().numpy()
    top_opacities = np.argsort(opacities.squeeze())[::-1]
    means = np.delete(means, top_opacities[n_gaussians:], axis=0)
    colors = np.delete(colors, top_opacities[n_gaussians:], axis=0)
    centroid = means.mean(axis=0)
    means -= centroid
    model.gauss_params["means"] = (
        model.gauss_params["means"] - torch.from_numpy(centroid).to("cuda").float()
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_pipeline(target: str, outputs_dir: str):
    print(f"  Searching for '{target}' in {outputs_dir}...")
    yamls = sorted(glob(f"{outputs_dir}/*/config.yml"), key=lambda x: x.split("/")[-2], reverse=True)
    for path in tqdm(yamls, leave=False):
        if not os.path.isdir(Path(path).parent / "nerfstudio_models"):
            continue
        with open(path, "r") as f:
            text = f.readlines()
        if any(target in t.split() for t in text):
            print(f"  Found: {path}")
            _, pipeline, _, _ = eval_setup(Path(path))
            return pipeline
    raise ValueError(f"Could not find pipeline matching '{target}' in {outputs_dir}")


def compute_icp(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
    args: Args,
    init_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run colored ICP with multiple rotation initializations, return best colored and colorless transforms."""
    for pcd in [pcd1, pcd2]:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )

    print(f"\n  Trying {len(args.rotations)} rotation initializations to handle rotational symmetry...")
    print(f"  (Rotations: {[f'({rx},{ry},{rz})' for rx,ry,rz in args.rotations]})")

    colored_regs, colorless_regs = [], []
    for rx, ry, rz in tqdm(args.rotations, desc="ICP initializations"):
        rot_mat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        tsfm = np.eye(4)
        tsfm[:3, :3] = rot_mat
        init = tsfm @ init_transform

        colored_regs.append(
            o3d.pipelines.registration.registration_colored_icp(
                pcd2, pcd1,
                max_correspondence_distance=args.max_correspondence_distance,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=10000
                ),
                init=init,
            )
        )
        colorless_regs.append(
            o3d.pipelines.registration.registration_colored_icp(
                pcd2, pcd1,
                max_correspondence_distance=args.max_correspondence_distance,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100000
                ),
                init=init,
            )
        )

    best_colored = sorted(colored_regs, key=lambda x: x.fitness)[-1]
    best_colorless = sorted(colorless_regs, key=lambda x: x.fitness)[-1]

    print(f"\n  Best colored ICP   — fitness: {best_colored.fitness:.6f}  RMSE: {best_colored.inlier_rmse:.6f}")
    print(f"  Best colorless ICP — fitness: {best_colorless.fitness:.6f}  RMSE: {best_colorless.inlier_rmse:.6f}")
    print(f"  (Fitness = fraction of inlier correspondences; RMSE = alignment error in metres)")

    out_dir = f"{args.output_dir}/{args.folder_name}"
    pcd2.transform(best_colored.transformation)
    o3d.io.write_point_cloud(f"{out_dir}/merged_colored.ply", pcd1 + pcd2)
    pcd2.transform(np.linalg.inv(best_colored.transformation))
    pcd2.transform(best_colorless.transformation)
    o3d.io.write_point_cloud(f"{out_dir}/merged_colorless.ply", pcd1 + pcd2)

    return best_colored.transformation.copy(), best_colorless.transformation.copy()


def crop_camera(camera, args: Args):
    camera.width = torch.tensor([[args.crop_width]])
    camera.height = torch.tensor([[args.crop_height]])
    camera.cx = torch.tensor([[
        camera.cx.squeeze().item() + float(camera.width.squeeze().item() - 1) / 2 - args.crop_width
    ]])
    camera.cy = torch.tensor([[
        camera.cy.squeeze().item() + float(camera.height.squeeze().item() - 1) / 2 - args.crop_height
    ]])
    return camera


def transform_camera(camera, transformation):
    c2w = camera.camera_to_worlds.squeeze().detach().cpu().numpy()
    c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)])
    c2w = np.linalg.inv(transformation) @ c2w
    camera.camera_to_worlds = torch.from_numpy(c2w)[..., :3, :].unsqueeze(0).to("cuda").float()
    return camera


def render(pipeline, camera):
    outputs = pipeline.model.get_outputs(camera.to("cuda"))
    return outputs["rgb"], outputs["depth"], outputs["accumulation"]


def select_transform(pipeline1, pipeline2, transform1, transform2, args: Args) -> np.ndarray:
    """Render a test frame with both transforms and pick the one with better overlap."""
    out_dir = f"{args.output_dir}/{args.folder_name}"
    camera = pipeline1.datamanager.train_dataset.cameras[0:1]
    camera = crop_camera(camera, args)

    def eval_transform(tsfm, suffix):
        rgb1, _, acc1 = render(pipeline1, camera)
        cam2 = transform_camera(camera, tsfm)
        rgb2, _, acc2 = render(pipeline2, cam2)
        transform_camera(cam2, np.linalg.inv(tsfm))  # restore

        acc1 = acc1.detach().cpu().numpy()
        acc2 = acc2.detach().cpu().numpy()
        acc_overlap = (acc1 * acc2) / max(np.sum(acc1), np.sum(acc2))
        abs_diff = np.linalg.norm(
            rgb1.detach().cpu().numpy() - rgb2.detach().cpu().numpy(), axis=-1
        )
        img1 = cv2.cvtColor((rgb1.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor((rgb2.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out_dir}/icp_test_{suffix}_ref.jpg", img1)
        cv2.imwrite(f"{out_dir}/icp_test_{suffix}_query.jpg", img2)
        return float(np.sum(acc_overlap)), float(np.mean(abs_diff))

    overlap1, diff1 = eval_transform(transform1, "colored")
    overlap2, diff2 = eval_transform(transform2, "colorless")

    print(f"\n  Transform selection (based on rendered overlap at frame 0):")
    print(f"    Colored   ICP — overlap: {overlap1:.4f}  mean pixel diff: {diff1:.4f}")
    print(f"    Colorless ICP — overlap: {overlap2:.4f}  mean pixel diff: {diff2:.4f}")

    if overlap1 < 0.9 and overlap2 < 0.9:
        raise ValueError(
            f"Both transforms have low overlap (<0.9). "
            f"Try providing a better --init-transform or adding more --rotations."
        )
    elif overlap1 < 0.8:
        print("    → Selecting colorless ICP (colored had insufficient overlap)")
        return transform2
    elif overlap2 < 0.8:
        print("    → Selecting colored ICP (colorless had insufficient overlap)")
        return transform1
    elif diff1 <= diff2:
        print("    → Selecting colored ICP (lower per-pixel difference after alignment)")
        return transform1
    else:
        print("    → Selecting colorless ICP (lower per-pixel difference after alignment)")
        return transform2


def main(args: Args):
    out_dir = f"{args.output_dir}/{args.folder_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading pipelines:")
    print(f"  Reference (pipeline1): '{args.pipeline_name1}'")
    print(f"  Query     (pipeline2): '{args.pipeline_name2}'")
    pipeline1 = get_pipeline(args.pipeline_name1, args.outputs_dir)
    pipeline2 = get_pipeline(args.pipeline_name2, args.outputs_dir)

    with open(f"{out_dir}/dataset.txt", "w") as f:
        f.writelines([args.pipeline_name1 + "\n", args.pipeline_name2 + "\n"])

    for p in [pipeline1, pipeline2]:
        p.cuda()
        p.model.eval()
        p.model.set_background(torch.tensor([1.0, 1.0, 1.0]).to("cuda"))

    print(f"\nExtracting point clouds (top {args.n_gaussians} Gaussians by opacity)...")
    pcd1 = extract_pointcloud(pipeline1.model, args.n_gaussians)
    pcd2 = extract_pointcloud(pipeline2.model, args.n_gaussians)
    o3d.io.write_point_cloud(f"{out_dir}/pcd1.ply", pcd1)
    o3d.io.write_point_cloud(f"{out_dir}/pcd2.ply", pcd2)
    print(f"  pcd1: {len(pcd1.points)} pts  pcd2: {len(pcd2.points)} pts")

    tsfm_path = f"{out_dir}/transformation.txt"
    if not os.path.exists(tsfm_path):
        init_transform = load_init_transform(args)
        transform1, transform2 = compute_icp(pcd1, pcd2, args, init_transform)
        transformation = select_transform(pipeline1, pipeline2, transform1, transform2, args)
        np.savetxt(tsfm_path, transformation)
        print(f"\n  Transformation saved to: {tsfm_path}")
    else:
        transformation = np.loadtxt(tsfm_path)
        print(f"\n  Loaded cached transformation from: {tsfm_path}")
        print(f"  (Delete {tsfm_path} to recompute)")

    print(f"\nRendering {args.n_frames} frame pairs and computing SSIM differences...")
    all_frames, all_depths, all_accs, ssim_scores = [], [], [], []
    for i in tqdm(range(args.n_frames)):
        camera = pipeline1.datamanager.train_dataset.cameras[i : i + 1]
        camera = crop_camera(camera, args)
        rgb1, depth1, acc1 = render(pipeline1, camera)
        camera = transform_camera(camera, transformation)
        rgb2, depth2, acc2 = render(pipeline2, camera)

        img1 = (rgb1.detach().cpu().numpy() * 255).astype(np.uint8)
        img2 = (rgb2.detach().cpu().numpy() * 255).astype(np.uint8)
        score, diff = ssim(img1, img2, full=True, multichannel=True, channel_axis=2)
        ssim_scores.append(score)

        cv2.imwrite(f"{out_dir}/{i:04d}_rgb1.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{out_dir}/{i:04d}_rgb2.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        diff_map = cv2.applyColorMap((np.mean(diff, axis=-1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imwrite(f"{out_dir}/{i:04d}_diff.jpg", diff_map)

        all_frames.append(np.stack([rgb1.detach().cpu().numpy(), rgb2.detach().cpu().numpy()]))
        all_depths.append(np.stack([depth1.detach().cpu().numpy(), depth2.detach().cpu().numpy()]))
        all_accs.append(np.stack([acc1.detach().cpu().numpy(), acc2.detach().cpu().numpy()]))

    np.save(f"{out_dir}/frames.npy", np.stack(all_frames))
    np.save(f"{out_dir}/depths.npy", np.stack(all_depths))
    np.save(f"{out_dir}/accs.npy", np.stack(all_accs))

    mean_ssim = float(np.mean(ssim_scores))
    print(f"\nResults saved to: {out_dir}/")
    print(f"  Mean SSIM across {args.n_frames} frames: {mean_ssim:.4f}")
    print(f"  (SSIM = 1.0 means pixel-perfect match; lower values indicate visible differences)")
    if mean_ssim < 0.8:
        print(f"  → Low SSIM ({mean_ssim:.4f}): the two objects have significant visual differences,")
        print(f"    which may indicate damage, wear, or geometric/appearance changes.")
    else:
        print(f"  → High SSIM ({mean_ssim:.4f}): the two objects are visually similar after alignment.")
    print(f"\n  Run scripts/analyze_from_jupyter.py to compute per-pixel difference statistics.")


if __name__ == "__main__":
    main(tyro.cli(Args))
