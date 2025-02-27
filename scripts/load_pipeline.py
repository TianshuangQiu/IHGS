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
import time
import torch
import torchvision
import cv2
import os
from matplotlib import pyplot as plt


def extract_pointcloud(model):
    means = model.gauss_params["means"].detach().cpu().numpy()
    colors = SH2RGB(model.gauss_params["features_dc"]).detach().cpu().numpy()
    opacities = model.gauss_params["opacities"].detach().cpu().numpy()
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    # center pointcloud
    top_opacities = np.argsort(opacities.squeeze())[::-1]
    means = np.delete(means, top_opacities[50000:], axis=0)
    colors = np.delete(colors, top_opacities[50000:], axis=0)
    centroid = means.mean(axis=0)
    means -= centroid
    model.gauss_params["means"] = (
        model.gauss_params["means"] - torch.from_numpy(centroid).to("cuda").float()
    )
    pcd.points = o3d.utility.Vector3dVector(means)  # Set positions
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Set colors

    return pcd


def get_pipeline(target):
    print("Searching for", target)
    found_path = None
    for path in tqdm(glob("outputs/combined/ihgs-full-merged/*/config.yml")):
        # load as str
        if not os.path.isdir(Path(path).parent / "nerfstudio_models"):
            continue
        with open(path, "r") as f:
            text = f.readlines()
        for t in text:
            if target in t:
                print(path)
                found_path = path
                break
        if found_path:
            break
    if not found_path:
        raise ValueError(f"Could not find {target}")
    train_config, pipeline, _, _ = eval_setup(Path(found_path))
    return pipeline


def compute_icp(pcd1, pcd2):
    pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    pcd2.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        pcd2,
        pcd1,
        max_correspondence_distance=0.001,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=100000
        ),
    )

    np.set_printoptions(precision=8, suppress=True)
    print("Transformation Matrix after ICP:\n", reg_p2p.transformation)

    pcd2.transform(reg_p2p.transformation)
    merged_pcd = pcd1 + pcd2
    o3d.io.write_point_cloud(
        f"defect_detection/combined.ply", merged_pcd.uniform_down_sample(10)
    )
    return reg_p2p.transformation.copy()


os.makedirs("defect_detection", exist_ok=True)
# pipeline1 = get_pipeline("expo-marker-bad")
# pipeline2 = get_pipeline("expo-marker-good")
pipeline1 = get_pipeline("pipeline-realsense-damaged")
pipeline2 = get_pipeline("pipeline-realsense-good")
pipeline1.cuda()
pipeline1.model.eval()

pipeline2.cuda()
pipeline2.model.eval()

pcd1 = extract_pointcloud(pipeline1.model)
pcd2 = extract_pointcloud(pipeline2.model)
o3d.io.write_point_cloud("defect_detection/pcd1.ply", pcd1)
o3d.io.write_point_cloud("defect_detection/pcd2.ply", pcd2)
transformation = compute_icp(pcd1, pcd2)


for i in range(0, 100, 1):
    camera = pipeline1.datamanager.train_dataset.cameras[i : i + 1]
    crop_ci = 1080
    crop_cj = 2048
    camera.width = torch.tensor([[crop_cj]])
    camera.height = torch.tensor([[crop_ci]])
    cx = (
        camera.cx.squeeze().item()
        + float(camera.width.squeeze().item() - 1) / 2
        - crop_cj
    )
    cy = (
        camera.cy.squeeze().item()
        + float(camera.height.squeeze().item() - 1) / 2
        - crop_ci
    )
    camera.cx = torch.tensor([[cx]])
    camera.cy = torch.tensor([[cy]])

    camera = camera.to("cuda")
    outputs = pipeline1.model.get_outputs(camera)
    rgb = outputs["rgb"]
    acc1 = outputs["accumulation"]
    img1 = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    wimg1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"defect_detection/rgb1_{i}.jpg", wimg1)

    camera_transform = camera.camera_to_worlds.squeeze().detach().cpu().numpy()
    camera_transform = np.concatenate(
        [camera_transform, np.array([[0, 0, 0, 1]], dtype=np.float32)]
    )
    camera_transform = np.linalg.inv(transformation) @ camera_transform
    camera.camera_to_worlds = (
        torch.from_numpy(camera_transform)[..., :3, :].unsqueeze(0).to("cuda")
    ).float()
    outputs = pipeline2.model.get_outputs(camera)
    rgb = outputs["rgb"]
    acc2 = outputs["accumulation"]
    img2 = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    wimg2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"defect_detection/rgb2_{i}.jpg", wimg2)

    from skimage.metrics import structural_similarity as ssim

    # img1 = cv2.imread("defect_detection/rgb1.jpg")
    # img2 = cv2.imread("defect_detection/rgb2.jpg")
    (score, diff) = ssim(img1, img2, full=True, multichannel=True, channel_axis=2)
    print("SSIM: {}".format(score))
    cv2.imwrite(f"defect_detection/diff_{i}.jpg", diff)

    abs_diff = np.linalg.norm(
        img1.astype(np.float32) - img2.astype(np.float32), axis=-1
    )
    diff = np.mean(diff, axis=-1)
    # figure with colorbar
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img1)
    ax[0].set_title("Damaged")
    ax[0].axis("off")
    ax[1].imshow(img2)
    ax[1].set_title("Good")
    ax[1].axis("off")
    im = ax[2].imshow(abs_diff, cmap="hot")
    ax[2].axis("off")
    # add color bar
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    acc = torch.sum(torch.min(acc1, acc2)).detach().cpu().numpy()
    ax[2].set_title("Difference")
    fig.suptitle(f"average_diff: {(np.mean(abs_diff/acc) * 1e4):.5f}")
    plt.savefig(f"defect_detection/diff_{i}_heatmap.jpg", bbox_inches="tight", dpi=300)
    plt.close()
