import tyro
import os
import json
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
import torch
import numpy as np
from autolab_core import RigidTransform


def main(data_path: str):
    data_path_left = os.path.join(data_path, "left")
    data_path_right = os.path.join(data_path, "right")

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
    out_json["frames"] = []

    left_grasp = np.loadtxt(os.path.join(data_path, "left_handover.txt"))
    right_grasp = np.loadtxt(os.path.join(data_path, "right_handover.txt"))
    lr_transform = np.linalg.inv(left_grasp) @ right_grasp
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

    with open(os.path.join(data_path, "combined", "transforms.json"), "w") as f:
        json.dump(out_json, f)


if __name__ == "__main__":
    tyro.cli(main)
