import tyro
import os
import json
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
import torch
import numpy as np
from autolab_core import RigidTransform
import open3d as o3d
from glob import glob
import shutil
from tqdm import tqdm
import cv2


def compute_icp(data_path):
    left_grasp = np.load(f"{data_path}/handover/robot_pose.npy")
    right_grasp = np.load(f"{data_path}/handover/best_grasp.npy")
    left_grasp = left_grasp[0, :16].reshape(4, 4)
    # np.savetxt(f"data/{dataset}/left_handover.txt", left_hand_matrix)
    # np.savetxt(f"data/{dataset}/right_handover.txt", right_hand)
    # left_grasp = np.loadtxt(os.path.join(data_path, "left_handover.txt"))
    # right_grasp = np.loadtxt(os.path.join(data_path, "right_handover.txt"))
    lr_transform = np.linalg.inv(left_grasp) @ right_grasp
    if not os.path.exists(os.path.join(data_path, "left", "gaussians.ply")):
        return lr_transform
    if not os.path.exists(os.path.join(data_path, "right", "gaussians.ply")):
        return lr_transform
    pcd1 = o3d.io.read_point_cloud(os.path.join(data_path, "left", "gaussians.ply"))
    pcd2 = o3d.io.read_point_cloud(os.path.join(data_path, "right", "gaussians.ply"))
    pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    pcd2.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        pcd2,
        pcd1,
        max_correspondence_distance=0.01,
        init=lr_transform,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        ),
    )

    np.set_printoptions(precision=8, suppress=True)
    print("Transformation Matrix after ICP:\n", reg_p2p.transformation)

    pcd2.transform(reg_p2p.transformation)
    merged_pcd = pcd1 + pcd2
    o3d.io.write_point_cloud(
        f"{data_path}/combined/merged.ply", merged_pcd.uniform_down_sample(10)
    )

    return reg_p2p.transformation.copy()


def optimize_masks(data_path, side):
    acc = torch.load(os.path.join(data_path, side, "pipe_save.pt"))
    shutil.copytree(
        os.path.join(data_path, side, "masks"),
        os.path.join(data_path, side, "pre_masks"),
        dirs_exist_ok=True,
    )
    remove_idx = []
    for k, v in tqdm(acc.items()):
        v = v.numpy()
        k = int(k)
        mask = cv2.imread(
            f"{data_path}/{side}/pre_masks/frame_{k:04d}.jpg",
            cv2.IMREAD_GRAYSCALE,
        )
        gripper_mask = cv2.imread(
            f"{data_path}/{side}/gripper_masks/frame_{k:04d}.jpg",
            cv2.IMREAD_GRAYSCALE,
        )
        # dilate then erode to remove small holes
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=4
        )
        for i in range(1, num_labels):
            if np.mean(v[labels == i]) < 0.2:
                mask[labels == i] = 0
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        v[v < 0.8] = 0
        v[v >= 0.8] = 1
        # if overlap is less than 80% of accumulation, remove
        if np.sum(np.maximum(mask, v.squeeze())) / np.sum(v) < 0.8:
            remove_idx.append(k)
        output_mask = np.zeros_like(mask)
        output_mask = np.maximum(mask, v.squeeze())
        output_mask = np.minimum(output_mask, (1 - gripper_mask))
        output_mask = output_mask * 255

        cv2.imwrite(
            f"{data_path}/{side}/masks/frame_{k:04d}.jpg",
            output_mask.astype(np.uint8),
        )
    return remove_idx


def main(data_path: str, second_run: bool = False):
    data_path_left = os.path.join(data_path, "left")
    data_path_right = os.path.join(data_path, "right")

    left_remove_idx = []
    right_remove_idx = []
    if not second_run:
        if os.path.exists(os.path.join(data_path_left, "pipe_save.pt")):
            left_remove_idx = optimize_masks(data_path, "left")
        if os.path.exists(os.path.join(data_path_right, "pipe_save.pt")):
            right_remove_idx = optimize_masks(data_path, "right")

    # load camera adjustment
    left_adjustment = torch.load(os.path.join(data_path_left, "camera_adjustment.pt"))
    right_adjustment = torch.load(os.path.join(data_path_right, "camera_adjustment.pt"))

    left_adjustment = exp_map_SO3xR3(left_adjustment)
    bottom_row = (
        torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    left_adjustment = torch.cat(
        [left_adjustment, bottom_row.repeat(left_adjustment.shape[0], 1, 1)], dim=1
    )
    right_adjustment = exp_map_SO3xR3(right_adjustment)
    right_adjustment = torch.cat(
        [right_adjustment, bottom_row.repeat(right_adjustment.shape[0], 1, 1)], dim=1
    )

    with open(os.path.join(data_path_left, "transforms.json"), "r") as f:
        left_trans = json.load(f)
    left_transforms = []
    for f in left_trans["frames"]:
        left_transforms.append(np.array(dict(f)["transform_matrix"]))
    with open(os.path.join(data_path_right, "transforms.json"), "r") as f:
        right_trans = json.load(f)
    right_transforms = []
    for f in right_trans["frames"]:
        right_transforms.append(np.array(dict(f)["transform_matrix"]))
    left_transforms = torch.from_numpy(np.array(left_transforms)).float()
    right_transforms = torch.from_numpy(np.array(right_transforms)).float()

    final_left = torch.cat(
        [
            torch.bmm(left_adjustment[..., :3, :3], left_transforms[..., :3, :3]),
            left_adjustment[..., :3, 3:] + left_transforms[..., :3, 3:],
        ],
        dim=-1,
    )
    final_right = torch.cat(
        [
            torch.bmm(right_adjustment[..., :3, :3], right_transforms[..., :3, :3]),
            right_adjustment[..., :3, 3:] + right_transforms[..., :3, 3:],
        ],
        dim=-1,
    )

    # load fast trained global adjustment
    if os.path.exists(os.path.join(data_path, "combined", "global.pt")):
        right_learned = (
            torch.load(os.path.join(data_path, "combined", "global.pt")).cpu().float()
        )
        right_learned = exp_map_SO3xR3(right_learned)
        right_learned = right_learned.repeat(final_right.shape[0], 1, 1)
        final_right = torch.cat(
            [
                torch.bmm(right_learned[..., :3, :3], final_right[..., :3, :3]),
                right_learned[..., :3, 3:] + final_right[..., :3, 3:],
            ],
            dim=-1,
        )

    out_json = {}
    out_json["fl_x"] = left_trans["fl_x"]
    out_json["fl_y"] = left_trans["fl_y"]
    out_json["cx"] = left_trans["cx"]
    out_json["cy"] = left_trans["cy"]
    out_json["w"] = left_trans["w"]
    out_json["h"] = left_trans["h"]
    out_json["ply_file_path"] = f"merged.ply"
    out_json["frames"] = []

    # estimate handover pose
    lr_transform = compute_icp(data_path)

    lr_transform = torch.from_numpy(lr_transform).unsqueeze(0)
    lr_transform = lr_transform.repeat(final_right.shape[0], 1, 1)

    final_right = torch.cat(
        [final_right, bottom_row.repeat(final_right.shape[0], 1, 1)], dim=1
    )
    final_right = torch.bmm(lr_transform.float(), final_right)
    final_left = torch.cat(
        [final_left, bottom_row.repeat(final_left.shape[0], 1, 1)], dim=1
    )
    for i, f in enumerate(left_trans["frames"]):
        cur_dict = dict(f)
        cur_dict["transform_matrix"] = final_left[i].tolist()
        out_json["frames"].append(cur_dict)

    for i, f in enumerate(right_trans["frames"]):
        cur_dict = dict(f)
        cur_dict["transform_matrix"] = final_right[i].tolist()
        out_json["frames"].append(cur_dict)

    os.makedirs(os.path.join(data_path, "combined", "gripper_masks"), exist_ok=True)
    # copy gripper masks
    idx = 0
    for d in [data_path_left, data_path_right]:
        gripper_masks = glob(os.path.join(d, "gripper_masks", "*.jpg"))
        gripper_masks.sort()
        for i, mask in enumerate(gripper_masks):
            shutil.copy(
                mask,
                os.path.join(
                    data_path, "combined", "gripper_masks", f"frame_{idx:04d}.jpg"
                ),
            )
            # m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            # cv2.imwrite(
            #     os.path.join(
            #         data_path, "combined", "gripper_masks", f"frame_{idx:04d}.jpg"
            #     ),
            #     cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=5),
            # )
            idx += 1
    with open(os.path.join(data_path, "combined", "transforms.json"), "w") as f:
        json.dump(out_json, f)


if __name__ == "__main__":
    tyro.cli(main)
