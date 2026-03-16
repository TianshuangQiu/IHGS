"""Remove outlier frames from transforms.json based on sudden mask size changes.

During data capture, occasional frames may have anomalous object masks (e.g. due
to occlusion, motion blur, or segmentation failure). This script detects frames
where the mask size changes abruptly relative to neighbouring frames using a
rate-of-change heuristic, and removes them from the transforms.json so they are
excluded from training.
"""

import os
import json
import cv2
import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    data_path: str
    """Root dataset directory containing left/ and right/ subdirectories"""
    n_frames: int = 100
    """Total number of frames per view to inspect"""
    group_size: int = 20
    """Frames are checked in groups of this size; the two largest rate-of-change
    jumps within each group are flagged as outliers"""
    sides: list[str] = None  # type: ignore
    """Which sides to filter. Defaults to ['left'] if not specified."""

    def __post_init__(self):
        if self.sides is None:
            self.sides = ["left"]


def find_outliers_rate_of_change(data: list[float]) -> np.ndarray:
    """Return indices of the two frames with the largest mask-size rate-of-change."""
    data = np.array(data, dtype=np.float64)
    diff = np.diff(data) / np.mean(data)
    diff = np.append(diff, (data[0] - data[-1]) / np.mean(data))
    return np.argsort(np.abs(diff))[-2:]


def filter_side(data_path: str, side: str, n_frames: int, group_size: int) -> int:
    tsfm_path = os.path.join(data_path, side, "transforms.json")
    tsfm = json.load(open(tsfm_path))
    frames = tsfm["frames"]
    remove_idx = set()

    for i in range(0, n_frames, group_size):
        group = frames[i : i + group_size]
        mask_sizes = []
        for frame in group:
            mask = cv2.imread(frame["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask_sizes.append(0.0)
            else:
                mask_sizes.append(float(np.sum(mask)))

        outliers = find_outliers_rate_of_change(mask_sizes)
        for idx in outliers:
            global_idx = i + int(idx)
            remove_idx.add(global_idx)
            size = mask_sizes[int(idx)]
            prev_size = mask_sizes[max(0, int(idx) - 1)]
            print(f"    Frame {global_idx}: mask size {size:.0f} px "
                  f"(prev: {prev_size:.0f} px, "
                  f"rate-of-change: {abs(size - prev_size) / max(np.mean(mask_sizes), 1):.3f})")

    tsfm["frames"] = [f for i, f in enumerate(frames) if i not in remove_idx]
    json.dump(tsfm, open(tsfm_path, "w"))
    return len(remove_idx)


def main(args: Args):
    print(f"\nFiltering outlier frames in: {args.data_path}")
    print(f"  Strategy: rate-of-change of mask size across groups of {args.group_size} frames.")
    print(f"  Frames where the mask grows or shrinks abruptly relative to neighbours are removed.")
    print(f"  Inspecting {args.n_frames} frames per side in groups of {args.group_size}.\n")

    for side in args.sides:
        tsfm_path = os.path.join(args.data_path, side, "transforms.json")
        if not os.path.exists(tsfm_path):
            print(f"  [{side}] transforms.json not found at {tsfm_path}, skipping.")
            continue
        print(f"  [{side}] Flagged outlier frames:")
        n_removed = filter_side(args.data_path, side, args.n_frames, args.group_size)
        if n_removed == 0:
            print(f"    (none)")
        print(f"  [{side}] Removed {n_removed} frame(s) from transforms.json\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
