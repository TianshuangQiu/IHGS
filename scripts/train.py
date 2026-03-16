"""Run the full IHGS training pipeline for a stereo dataset.

Pipeline overview
-----------------
Stage 1-2  Train independent Gaussian Splatting models on the left and right
           camera views in parallel. During training, the per-camera transforms
           are optimised so that the recovered point clouds are as accurate as
           possible.

Stage 3    Run the cross-view alignment step (first pass).  merge_dataset_with_opt.py
           reads the trained left/right outputs, aligns the two point clouds with
           ICP, and writes a combined dataset under {data_path}/combined/.

Stage 4-5  Re-train the left and right models a second time.  The point clouds
           from Stage 1-2 are already reasonable, but a second training pass with
           the cross-view constraint produces tighter alignment and cleaner
           geometry.

Stage 6    Run the cross-view alignment step (second pass, --second_run flag).
           A second ICP alignment on the improved point clouds yields a more
           accurate combined transform.

Stage 7    Train a single merged model (ihgs-full-merged) on the combined
           dataset.  This is the final output used for defect detection and
           evaluation.
"""

import os
import subprocess
import shutil
import tyro
from glob import glob
from subprocess import Popen
from dataclasses import dataclass, field


@dataclass
class Args:
    data_path: str
    """Root dataset directory.  Must contain left/ and right/ subdirectories,
    each with an images/ folder and a masks/ folder."""
    gpu_0: int = 0
    """GPU index used for the left-view model (Stages 1, 4, 7)."""
    gpu_1: int = 1
    """GPU index used for the right-view model (Stages 2, 5)."""
    skip_clean: bool = False
    """Skip the folder-cleaning step at startup.  By default, any previously
    generated outputs (except images/, masks/, and gripper_masks/) are removed
    so the run starts from a clean state."""


def clean_folder(folder: str) -> None:
    """Remove all subdirectories except images/, masks/, and gripper_masks/."""
    for subfolder in glob(os.path.join(folder, "*")):
        if os.path.isdir(subfolder):
            if os.path.basename(subfolder) not in {"masks", "images", "gripper_masks"}:
                shutil.rmtree(subfolder)


def main(args: Args) -> None:
    if not args.skip_clean:
        print("\n[Setup] Cleaning previous outputs from left/ and right/ ...")
        clean_folder(f"{args.data_path}/left")
        clean_folder(f"{args.data_path}/right")

    # ------------------------------------------------------------------
    # Stages 1-2: train left and right views in parallel
    # ------------------------------------------------------------------
    print("\n[Stages 1-2] Training left and right Gaussian Splatting models in parallel ...")
    print("  Left  model → GPU", args.gpu_0)
    print("  Right model → GPU", args.gpu_1)
    print("  During training the per-camera transforms are optimised; the resulting")
    print("  point clouds will be used for cross-view alignment in Stage 3.\n")

    train_cmds = [
        f"CUDA_VISIBLE_DEVICES={args.gpu_0} ns-train ihgs --data {args.data_path}/left",
        f"CUDA_VISIBLE_DEVICES={args.gpu_1} ns-train ihgs --data {args.data_path}/right",
    ]
    procs = [Popen(cmd, shell=True) for cmd in train_cmds]
    for p in procs:
        p.wait()

    # ------------------------------------------------------------------
    # Stage 3: cross-view alignment (first pass)
    # ------------------------------------------------------------------
    print("\n[Stage 3] Cross-view alignment — first pass ...")
    print("  Aligns the left and right point clouds with ICP and writes the")
    print("  combined dataset to {}/combined/.\n".format(args.data_path))

    subprocess.call(
        f"python3 inhand/merge_dataset_with_opt.py --data_path {args.data_path}",
        shell=True,
    )

    # ------------------------------------------------------------------
    # Stages 4-5: re-train left and right views in parallel
    # ------------------------------------------------------------------
    print("\n[Stages 4-5] Re-training left and right models (second pass) ...")
    print("  A second training run uses the cross-view constraint from Stage 3")
    print("  to produce tighter alignment and cleaner per-view geometry.\n")

    procs = [Popen(cmd, shell=True) for cmd in train_cmds]
    for p in procs:
        p.wait()

    # ------------------------------------------------------------------
    # Stage 6: cross-view alignment (second pass)
    # ------------------------------------------------------------------
    print("\n[Stage 6] Cross-view alignment — second pass ...")
    print("  A second ICP alignment on the improved point clouds yields a more")
    print("  accurate combined transform for the final merged training.\n")

    subprocess.call(
        f"python3 inhand/merge_dataset_with_opt.py --data_path {args.data_path} --second_run",
        shell=True,
    )

    # ------------------------------------------------------------------
    # Stage 7: train merged model
    # ------------------------------------------------------------------
    print("\n[Stage 7] Training merged model (ihgs-full-merged) on combined dataset ...")
    print("  This is the final model used for defect detection and evaluation.\n")

    merged_cmds = [
        f"CUDA_VISIBLE_DEVICES={args.gpu_0} ns-train ihgs-full-merged --data {args.data_path}/combined",
    ]
    procs = [Popen(cmd, shell=True) for cmd in merged_cmds]
    for p in procs:
        p.wait()

    print("\n[Done] Full IHGS pipeline complete.")
    print(f"  Final model outputs are under outputs/combined/ihgs-full-merged/")


if __name__ == "__main__":
    main(tyro.cli(Args))
